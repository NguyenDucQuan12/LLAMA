from __future__ import annotations
"""
Điều phối toàn bộ pipeline RAG:

DOCUMENTS:
    Question
    -> DenseDocumentRetriever lấy top N
    -> CrossEncoderDocumentReranker lấy top K từ top N (Chọn lọc lại 1 lần nữa)
    -> AnswerContextBuilder tạo D1...DK
    -> RagAnswerGenerator gọi Llama
    -> QuestionResponse

SQL:
    Question
    -> QuestionIntentRouter chọn predefined query
    -> SafeSqlServerService chạy SELECT an toàn
    -> AnswerContextBuilder tạo S1
    -> RagAnswerGenerator
    -> QuestionResponse

HYBRID:
    Nhánh document và nhánh SQL chạy song song.
    Kết quả được ghép thành D1...DK + S1 rồi gửi cho Llama.

Chạy test thật:

    python3 -m app.services.question_answering_service \
        --question "Làm thế nào để gọi robot?" \
        --tenant-id "wms" \
        --document-id "fabric-warehouse-guide" \
        --mode documents \
        --include-debug

    python3 -m app.services.question_answering_service \
        --question "Robot AGV-01 đang ở trạng thái nào?" \
        --tenant-id "wms" \
        --mode sql \
        --sql-query-key "get_robot_status" \
        --sql-parameters '{"robot_id":"AGV-01"}'

    python3 -m app.services.question_answering_service \
        --question "AGV-01 đang lỗi; quy trình kiểm tra là gì?" \
        --tenant-id "wms" \
        --mode hybrid \
        --sql-query-key "get_robot_status" \
        --sql-parameters '{"robot_id":"AGV-01"}' \
        --include-debug
"""

import argparse
import asyncio
import inspect
import json
import logging
import sys
from collections.abc import Mapping
from typing import Any, TypeVar
import time
# Mở comment 3 dòng bên dưới mỗi khi test (Chạy trực tiếp hàm if __main__)
import os,sys
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from clients.ollama_client import OllamaClient
from clients.qdrant_repository import QdrantRepository
from config import Settings, get_settings
from generation.answer_generator import RagAnswerGenerator
from generation.context_builder import AnswerContextBuilder,GenerationSource
from retrieval.models import RetrievedChunk, RerankedChunk
from retrieval.reranker import CrossEncoderDocumentReranker
from retrieval.retriever import DenseDocumentRetriever
from schemas import AnswerCitation, QuestionMode, QuestionRequest, QuestionResponse, RetrievedChunkResponse, SqlExecutionResponse
from sql.intent_router import QuestionIntentRouter, RouteDecision
from sql.query_registry import PredefinedSqlQueryRegistry
from sql.sql_service import SafeSqlServerService


logger = logging.getLogger(__name__)

# Chỉ ba route này được service xử lý.
_ALLOWED_ROUTES = {"documents", "sql", "hybrid"}


class QuestionAnsweringService:
    """
    Orchestrator cấp ứng dụng.

    Class này không tự tìm Qdrant, không tự viết SQL và không tự gọi HTTP.
    Nó chỉ phối hợp các component chuyên trách.
    """

    def __init__(self, settings: Settings, retriever: DenseDocumentRetriever, reranker: CrossEncoderDocumentReranker, intent_router: QuestionIntentRouter, sql_server_service: SafeSqlServerService, context_builder: AnswerContextBuilder, answer_generator: RagAnswerGenerator) -> None:
        # Lưu cấu hình chung.
        self.settings = settings
        # Tầng dense retrieval.
        self.retriever = retriever
        # Tầng cross-encoder rerank.
        self.reranker = reranker
        # Router chọn documents/sql/hybrid.
        self.intent_router = intent_router
        # Chỉ chạy predefined SELECT.
        self.sql_server_service = sql_server_service
        # Ghép nguồn thành context D1/S1.
        self.context_builder = context_builder
        # Gọi Llama và validate structured output.
        self.answer_generator = answer_generator
        # Phát hiện cấu hình sai ngay khi dựng service.
        self._validate_configuration()

    async def answer_question(self, request: QuestionRequest) -> QuestionResponse:
        """
        Public method xử lý một câu hỏi.

        Nếu Settings có question_answer_timeout_seconds,
        toàn bộ pipeline bị giới hạn trong khoảng thời gian đó.
        """

        # Chặn request sai trước khi gọi model hoặc database.
        self._validate_request(request)

        timeout_value = getattr(self.settings, "question_answer_timeout_seconds", None)

        # Không cấu hình timeout.
        if timeout_value is None:
            return await self._answer_question_impl(request)

        try:
            timeout_seconds = float(timeout_value)
        except (TypeError, ValueError) as exc:
            raise TypeError("question_answer_timeout_seconds phải là số hoặc None.") from exc

        if timeout_seconds <= 0:
            raise ValueError("question_answer_timeout_seconds phải lớn hơn 0.")

        # Toàn bộ quá trình xử lý dưới đây phải được hoàn thành trong timeout_seconds
        try:
            # asyncio.timeout có từ Python 3.11.
            async with asyncio.timeout(timeout_seconds):
                return await self._answer_question_impl(request)
        except TimeoutError as exc:
            raise RuntimeError(
                f"Đã quá thời gian {timeout_seconds}s nhưng mô hình vẫn chưa thể tìm được câu trả lời"
            ) from exc

    async def _answer_question_impl(self, request: QuestionRequest) -> QuestionResponse:
        """
        Quy trình chính:
        1. Router.
        2. Document/SQL.
        3. Build context.
        4. Llama.
        5. Citation.
        6. Response.
        """

        # ====================================================
        # BƯỚC 1: ROUTE CÂU HỎI
        # ====================================================

        # sql_query_key="get_robot_status" là tên của sql
        # sql_parameters={"robot_id": "AGV-01",}  là tham số truyền vào cho sql

        route_decision = await self.intent_router.decide(
            question=request.question,
            mode=request.mode,
            explicit_query_key=request.sql_query_key,
            explicit_parameters=request.sql_parameters,
        )

        # Chuẩn hóa và kiểm tra route.
        route = self._validate_route_decision(route_decision)

        logger.info("Route=%s, query_key=%s", route, route_decision.query_key)

        # Kết quả mặc định của hai nhánh.
        top_documents: list[RetrievedChunk] = []
        reranked_documents: list[RerankedChunk] = []
        sql_result: SqlExecutionResponse | None = None

        # Các thông báo khi hybrid chỉ hoàn thành một phần.
        notices: list[str] = []

        should_search_documents = route in {"documents", "hybrid"}

        should_execute_sql = (
            route in {"sql", "hybrid"}
            and route_decision.query_key is not None
        )

        # SQL thuần nhưng router không chọn được query an toàn.
        if route == "sql" and not should_execute_sql:
            return self._build_early_response(
                request=request,
                route=route,
                answer=(
                    "Tôi chưa xác định được predefined SQL query "
                    "phù hợp với câu hỏi."
                ),
                insufficient_context=True,
                top_documents=[],
                reranked_documents=[],
                sql_result=None,
            )

        # Hybrid vẫn chạy tài liệu khi SQL chưa có query key.
        if route == "hybrid" and not should_execute_sql:
            notices.append(
                "Chưa chạy được nhánh SQL vì router chưa xác định "
                "được query_key an toàn."
            )

        # ====================================================
        # BƯỚC 2: CHẠY CÁC NHÁNH
        # ====================================================

        if should_search_documents and should_execute_sql:
            # Hai nhánh độc lập nên chạy song song.
            document_outcome, sql_outcome = await asyncio.gather(
                self._run_document_pipeline(request),
                self._run_sql_pipeline(route_decision),
                return_exceptions=True,
            )

            # Xử lý nhánh document.
            if isinstance(document_outcome, Exception):
                logger.error(
                    "Nhánh document thất bại trong hybrid.",
                    exc_info=(
                        type(document_outcome),
                        document_outcome,
                        document_outcome.__traceback__,
                    ),
                )

                if not self._allow_partial_hybrid():
                    raise document_outcome

                notices.append(
                    "Nhánh tài liệu gặp lỗi; câu trả lời chỉ dùng SQL nếu có."
                )
            else:
                top_documents, reranked_documents = document_outcome

            # Xử lý nhánh SQL.
            if isinstance(sql_outcome, Exception):
                logger.error(
                    "Nhánh SQL thất bại trong hybrid.",
                    exc_info=(
                        type(sql_outcome),
                        sql_outcome,
                        sql_outcome.__traceback__,
                    ),
                )

                if not self._allow_partial_hybrid():
                    raise sql_outcome

                notices.append(
                    "Nhánh SQL gặp lỗi; câu trả lời chỉ dùng tài liệu nếu có."
                )
            else:
                sql_result = sql_outcome

        elif should_search_documents:
            top_documents, reranked_documents = (
                await self._run_document_pipeline(request)
            )

        elif should_execute_sql:
            sql_result = await self._run_sql_pipeline(route_decision)

        # ====================================================
        # BƯỚC 3: THIẾU THAM SỐ SQL
        # ====================================================

        sql_missing_parameters = (
            sql_result is not None
            and not sql_result.executed
            and bool(sql_result.missing_parameters)
        )

        if sql_missing_parameters:
            missing_text = ", ".join(sql_result.missing_parameters)

            notice = (
                f"Truy vấn {sql_result.query_key} cần thêm tham số: "
                f"{missing_text}."
            )

            # SQL thuần không có nguồn khác để trả lời.
            if route == "sql":
                return self._build_early_response(
                    request=request,
                    route=route,
                    answer=notice,
                    insufficient_context=True,
                    top_documents=top_documents,
                    reranked_documents=reranked_documents,
                    sql_result=sql_result,
                )

            # Hybrid vẫn tiếp tục bằng document.
            notices.append(notice)

        # SQL không chạy vì lỗi khác, ví dụ DB lỗi hoặc query bị chặn.
        sql_failed_without_missing_parameters = (
            sql_result is not None
            and not sql_result.executed
            and not bool(sql_result.missing_parameters)
        )

        if sql_failed_without_missing_parameters:
            sql_error_text = str(
                getattr(
                    sql_result,
                    "error_message",
                    None,
                )
                or getattr(
                    sql_result,
                    "message",
                    None,
                )
                or "Truy vấn SQL không được thực thi."
            ).strip()

            if route == "sql":
                return self._build_early_response(
                    request=request,
                    route=route,
                    answer=sql_error_text,
                    insufficient_context=True,
                    top_documents=top_documents,
                    reranked_documents=reranked_documents,
                    sql_result=sql_result,
                )

            notices.append(sql_error_text)

        # ====================================================
        # BƯỚC 4: BUILD CONTEXT
        # ====================================================

        generation_sources = self.context_builder.build_sources(
            document_chunks=reranked_documents,
            sql_result=sql_result,
        )

        context_text = self.context_builder.build_context_text(
            generation_sources
        )

        # ====================================================
        # BƯỚC 5: GỌI LLAMA
        # ====================================================

        raw_answer_output = await self.answer_generator.generate_answer(
            question=request.question,
            context_text=context_text,
            sources=generation_sources,
        )

        answer_output = self._validate_answer_output(raw_answer_output)

        answer_text = answer_output["answer"]
        insufficient_context = answer_output["insufficient_context"]

        # Một nhánh hybrid thiếu/lỗi => kết quả chưa đầy đủ.
        if notices:
            insufficient_context = True
            answer_text = (
                answer_text.rstrip()
                + "\n\nLưu ý: "
                + " ".join(notices)
            )

        # ====================================================
        # BƯỚC 6: CITATION
        # ====================================================

        citations = self._build_citations(
            citation_items=answer_output["citations"],
            generation_sources=generation_sources,
        )

        # ====================================================
        # BƯỚC 7: RESPONSE
        # ====================================================

        return QuestionResponse(
            answer=answer_text,
            route=route,
            insufficient_context=insufficient_context,
            citations=citations,
            top_five_documents=(
                self._convert_reranked_chunks_to_response(
                    reranked_documents
                )
            ),
            sql_result=sql_result,
            top_twenty_documents=(
                self._convert_retrieved_chunks_to_response(top_documents)
                if request.include_debug_information
                else None
            ),
        )

    async def _run_document_pipeline(self, request: QuestionRequest) -> tuple[list[RetrievedChunk], list[RerankedChunk]]:
        """
        Dense top N -> cross-encoder top K.
        """
        # Lấy top các câu trả lời gần nhất với câu hỏi
        top_documents = await self.retriever.retrieve(
            question=request.question,
            tenant_id=request.tenant_id,
            document_id=request.document_id,
            top_k=self.settings.retrieval_top_k,
        )

        # Không có dense result thì không cần gọi model reranker.
        if not top_documents:
            return [], []
        # Từ top trên, chọn lọc lại các câu trả lời tốt nhất
        reranked_documents = await self.reranker.rerank(
            question=request.question,
            candidates=top_documents,
            top_k=self.settings.rerank_top_k,
        )

        return top_documents, reranked_documents

    async def _run_sql_pipeline(self, route_decision: RouteDecision) -> SqlExecutionResponse:
        """
        Chạy predefined query do router lựa chọn.
        """

        query_key = route_decision.query_key

        if query_key is None:
            raise RuntimeError(
                "Không thể chạy SQL khi RouteDecision.query_key=None."
            )

        parameters = (
            route_decision.parameters
            if route_decision.parameters is not None
            else {}
        )

        if not isinstance(parameters, Mapping):
            raise TypeError(
                "RouteDecision.parameters phải là Mapping hoặc None."
            )

        return await self.sql_server_service.execute_predefined_query(
            query_key=query_key,
            parameters=dict(parameters),
        )

    def _build_citations(
        self,
        citation_items: list[dict[str, str]],
        generation_sources: list[GenerationSource],
    ) -> list[AnswerCitation]:
        """
        Ghép citation label Llama trả với metadata nguồn thật.
        """

        # Lookup không phân biệt d1/D1.
        source_lookup = {
            str(source.source_label).strip().upper(): source
            for source in generation_sources
            if str(source.source_label).strip()
        }

        output: list[AnswerCitation] = []
        seen_labels: set[str] = set()

        for item in citation_items:
            if not isinstance(item, Mapping):
                continue

            label = str(item.get("source_label", "")).strip().upper()
            evidence = str(item.get("evidence", "")).strip()

            if not label or not evidence or label in seen_labels:
                continue

            source = source_lookup.get(label)

            # Nhãn bịa đặt bị bỏ.
            if source is None:
                continue

            output.append(
                AnswerCitation(
                    source_label=source.source_label,
                    source_type=source.source_type,
                    source_file=source.source_file,
                    chunk_index=source.chunk_index,
                    page_numbers=source.page_numbers,
                    headings=source.headings,
                    evidence=evidence,
                )
            )

            seen_labels.add(label)

        return output

    def _convert_retrieved_chunks_to_response(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunkResponse]:
        """
        Chuyển dense top N sang schema API.
        """

        return [
            RetrievedChunkResponse(
                point_id=chunk.point_id,
                dense_score=chunk.dense_score,
                reranker_score=None,
                final_score=None,
                document_id=chunk.document_id,
                source_file=chunk.source_file,
                chunk_index=chunk.chunk_index,
                headings=chunk.headings,
                page_numbers=chunk.page_numbers,
                text=chunk.contextualized_text,
            )
            for chunk in chunks
        ]

    def _convert_reranked_chunks_to_response(
        self,
        chunks: list[RerankedChunk],
    ) -> list[RetrievedChunkResponse]:
        """
        Chuyển top K đã rerank sang schema API.
        """

        return [
            RetrievedChunkResponse(
                point_id=chunk.point_id,
                dense_score=chunk.dense_score,
                reranker_score=chunk.reranker_score,
                final_score=chunk.final_score,
                document_id=chunk.document_id,
                source_file=chunk.source_file,
                chunk_index=chunk.chunk_index,
                headings=chunk.headings,
                page_numbers=chunk.page_numbers,
                text=chunk.contextualized_text,
            )
            for chunk in chunks
        ]

    def _build_early_response(
        self,
        request: QuestionRequest,
        route: str,
        answer: str,
        insufficient_context: bool,
        top_documents: list[RetrievedChunk],
        reranked_documents: list[RerankedChunk],
        sql_result: SqlExecutionResponse | None,
    ) -> QuestionResponse:
        """
        Response khi không cần hoặc chưa thể gọi Llama.
        """

        return QuestionResponse(
            answer=answer,
            route=route,
            insufficient_context=insufficient_context,
            citations=[],
            top_five_documents=(
                self._convert_reranked_chunks_to_response(
                    reranked_documents
                )
            ),
            sql_result=sql_result,
            top_twenty_documents=(
                self._convert_retrieved_chunks_to_response(top_documents)
                if request.include_debug_information
                else None
            ),
        )

    def _validate_route_decision(
        self,
        route_decision: RouteDecision,
    ) -> str:
        """
        Kiểm tra output intent router.
        """

        if route_decision is None:
            raise RuntimeError("Intent router trả None.")

        raw_route = getattr(route_decision, "route", None)

        # Route có thể là enum.
        if hasattr(raw_route, "value"):
            raw_route = raw_route.value

        route = str(raw_route).strip().lower()

        if route not in _ALLOWED_ROUTES:
            raise RuntimeError(
                f"Route không hợp lệ: {route!r}. "
                f"Cho phép: {sorted(_ALLOWED_ROUTES)}."
            )

        query_key = getattr(route_decision, "query_key", None)

        if (
            query_key is not None
            and (
                not isinstance(query_key, str)
                or not query_key.strip()
            )
        ):
            raise RuntimeError(
                "RouteDecision.query_key phải là string không rỗng "
                "hoặc None."
            )

        return route

    def _validate_answer_output(
        self,
        value: Any,
    ) -> dict[str, Any]:
        """
        Defense-in-depth cho output AnswerGenerator.
        """

        if not isinstance(value, Mapping):
            raise TypeError(
                "RagAnswerGenerator phải trả Mapping."
            )

        answer = value.get("answer")
        insufficient = value.get("insufficient_context")
        citations = value.get("citations")

        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError(
                "answer_output.answer phải là string không rỗng."
            )

        if not isinstance(insufficient, bool):
            raise RuntimeError(
                "answer_output.insufficient_context phải là boolean."
            )

        if not isinstance(citations, list):
            raise RuntimeError(
                "answer_output.citations phải là list."
            )

        return {
            "answer": answer.strip(),
            "insufficient_context": insufficient,
            "citations": citations,
        }

    def _validate_request(self, request: QuestionRequest) -> None:
        """
        Kiểm tra request dù caller không đi qua FastAPI/Pydantic.
        """

        if request is None:
            raise TypeError("request không được là None.")

        question = getattr(request, "question", None)

        if not isinstance(question, str) or not question.strip():
            raise ValueError("request.question phải là string không rỗng.")

        max_characters = getattr(self.settings, "question_max_characters", 4000)
        
        if (isinstance(max_characters, bool) or not isinstance(max_characters, int) or max_characters <= 0):
            raise ValueError("question_max_characters phải là int > 0.")

        if len(question.strip()) > max_characters:
            raise ValueError(f"Câu hỏi quá dài: {len(question.strip())} > {max_characters}.")

        tenant_id = getattr(request, "tenant_id", None)

        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("request.tenant_id phải là string không rỗng.")

        document_id = getattr(request, "document_id", None)

        if (
            document_id is not None
            and (not isinstance(document_id, str) or not document_id.strip())
        ):
            raise ValueError("request.document_id phải là string không rỗng hoặc None.")

        sql_parameters = getattr(request, "sql_parameters", None)

        if (sql_parameters is not None and not isinstance(sql_parameters, Mapping)):
            raise TypeError("request.sql_parameters phải là Mapping hoặc None.")

        if not isinstance(getattr(request, "include_debug_information", None), bool):
            raise TypeError(
                "include_debug_information phải là boolean."
            )

    def _validate_configuration(self) -> None:
        """
        Kiểm tra các cấu hình mà orchestrator sử dụng.
        """

        for field_name in ("retrieval_top_k", "rerank_top_k"):
            value = getattr(self.settings, field_name, None)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"{field_name} phải là int > 0."
                )

        if self.settings.rerank_top_k > self.settings.retrieval_top_k:
            raise ValueError(
                "rerank_top_k không được lớn hơn retrieval_top_k."
            )

        allow_partial = getattr(
            self.settings,
            "question_allow_partial_hybrid_results",
            True,
        )

        if not isinstance(allow_partial, bool):
            raise TypeError(
                "question_allow_partial_hybrid_results phải là boolean."
            )

    def _allow_partial_hybrid(self) -> bool:
        """
        True: một nhánh hybrid lỗi thì dùng nhánh còn lại.
        False: bất kỳ nhánh nào lỗi thì toàn request lỗi.
        """

        return bool(
            getattr(
                self.settings,
                "question_allow_partial_hybrid_results",
                True,
            )
        )


# ============================================================
# 1. CẤU HÌNH TEST ĐIỀN TRỰC TIẾP
# ============================================================

TEST_QUESTION = "Làm thế nào để gọi robot?"
TEST_TENANT_ID = "viva_factory"

# None = tìm trong tất cả tài liệu thuộc tenant.
# Có thể đổi thành document_id cụ thể đã dùng lúc ingest.
TEST_DOCUMENT_ID: str | None = None

# Lần test đầu tiên nên ép DOCUMENTS để kiểm tra riêng:
# Qdrant -> reranker -> context -> Llama.
TEST_MODE = QuestionMode.DOCUMENTS

# Chỉ dùng khi đổi TEST_MODE sang SQL/HYBRID.
TEST_SQL_QUERY_KEY: str | None = None
TEST_SQL_PARAMETERS: dict[str, Any] = {}

TEST_INCLUDE_DEBUG = False

# Không cho test treo gần 10 phút như log cũ.
TEST_TIMEOUT_SECONDS = 6000.0


async def _close_if_supported(resource: Any) -> None:
    """
    Đóng object nếu có close(), hỗ trợ cả sync và async.
    """

    if resource is None:
        return

    close_method = getattr(resource, "close", None)

    if not callable(close_method):
        return

    result = close_method()

    if inspect.isawaitable(result):
        await result


async def _close_resources(resources: list[Any]) -> None:
    """
    Đóng resource theo thứ tự ngược lúc tạo.
    """

    for resource in reversed(resources):
        try:
            await _close_if_supported(resource)
        except Exception:
            logger.exception(
                "Không đóng được resource %s.",
                type(resource).__name__,
            )


async def _build_real_service(
    settings: Settings,
) -> tuple[QuestionAnsweringService, list[Any]]:
    """
    Dựng toàn bộ pipeline thật.

    Điểm sửa quan trọng:
    - Chỉ tạo MỘT SafeSqlServerService.
    - Router và QuestionAnsweringService dùng chung instance đó.
    - Nếu dựng pipeline lỗi giữa chừng, resource đã tạo được đóng lại.
    """

    resources: list[Any] = []

    try:
        # Một OllamaClient dùng chung cho embedding/router/Llama.
        ollama_client = OllamaClient(settings)
        resources.append(ollama_client)

        # Repository Qdrant thật.
        qdrant_repository = QdrantRepository(settings)
        resources.append(qdrant_repository)

        # Warning payload index trong Qdrant local chỉ là cảnh báo,
        # không phải lỗi retrieval.
        await qdrant_repository.ensure_collection()

        # Registry predefined SQL thật.
        query_registry = PredefinedSqlQueryRegistry()

        # CHỈ TẠO MỘT SQL SERVICE.
        sql_server_service = SafeSqlServerService(
            settings,
            query_registry,
        )
        resources.append(sql_server_service)

        # Dense retrieval.
        retriever = DenseDocumentRetriever(
            settings=settings,
            ollama_client=ollama_client,
            qdrant_repository=qdrant_repository,
        )

        # Cross-encoder reranker.
        reranker = CrossEncoderDocumentReranker(settings)

        # Intent router dùng cùng SQL service phía trên.
        intent_router = QuestionIntentRouter(
            ollama_client=ollama_client,
            query_registry=query_registry,
            sql_server_service=sql_server_service,
        )

        # Context và answer generation.
        context_builder = AnswerContextBuilder(settings)
        answer_generator = RagAnswerGenerator(ollama_client)

        service = QuestionAnsweringService(
            settings=settings,
            retriever=retriever,
            reranker=reranker,
            intent_router=intent_router,
            sql_server_service=sql_server_service,
            context_builder=context_builder,
            answer_generator=answer_generator,
        )

        return service, resources

    except Exception:
        await _close_resources(resources)
        raise


def _response_to_dict(
    response: QuestionResponse,
) -> dict[str, Any]:
    """
    Hỗ trợ Pydantic v2 và v1.
    """

    model_dump = getattr(response, "model_dump", None)

    if callable(model_dump):
        return model_dump(mode="json")

    dict_method = getattr(response, "dict", None)

    if callable(dict_method):
        return dict_method()

    raise TypeError(
        "QuestionResponse không hỗ trợ model_dump() hoặc dict()."
    )


async def test_real_question() -> None:
    """
    Chạy toàn bộ pipeline thật, không dùng argparse và không tạo dữ liệu giả.
    """

    settings = get_settings()

    request = QuestionRequest(
        question=TEST_QUESTION,
        tenant_id=TEST_TENANT_ID,
        document_id=TEST_DOCUMENT_ID,
        mode=TEST_MODE,
        sql_query_key=TEST_SQL_QUERY_KEY,
        sql_parameters=TEST_SQL_PARAMETERS,
        include_debug_information=TEST_INCLUDE_DEBUG,
    )

    resources: list[Any] = []

    try:
        print()
        print("=" * 88)
        print("BẮT ĐẦU DỰNG PIPELINE THẬT")
        print("=" * 88)

        service, resources = await _build_real_service(settings)

        print("Đã dựng xong pipeline.")
        print(f"Question : {TEST_QUESTION}")
        print(f"Tenant   : {TEST_TENANT_ID}")
        print(f"Document : {TEST_DOCUMENT_ID}")
        print(f"Mode     : {TEST_MODE}")
        print(f"Timeout  : {TEST_TIMEOUT_SECONDS} giây")
        print()

        start_time = time.perf_counter()

        try:
            # Timeout ngoài giúp dừng test ngay cả khi timeout trong
            # OllamaClient đang được cấu hình quá dài.
            response = await asyncio.wait_for(
                service.answer_question(request),
                timeout=TEST_TIMEOUT_SECONDS,
            )

        except asyncio.TimeoutError as exception:
            elapsed = time.perf_counter() - start_time

            raise RuntimeError(
                "Pipeline bị timeout sau "
                f"{elapsed:.1f} giây. "
                "Theo log cũ, bước có khả năng bị treo là "
                "RagAnswerGenerator.generate_answer() khi gọi Ollama chat. "
                "Hãy kiểm tra model chat trong Settings, `ollama list`, "
                "RAM và timeout của OllamaClient."
            ) from exception

        elapsed = time.perf_counter() - start_time

        print()
        print("=" * 88)
        print("KẾT QUẢ QUESTION ANSWERING THẬT")
        print("=" * 88)
        print(f"Thời gian xử lý: {elapsed:.2f} giây")
        print()

        print(
            json.dumps(
                _response_to_dict(response),
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

    finally:
        await _close_resources(resources)


async def main() -> None:
    """
    Chạy test trực tiếp bằng các hằng số ở đầu file.
    """

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    # Giảm log download/network không cần thiết.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    await test_real_question()


if __name__ == "__main__":
    try:
        # Chỉ tạo đúng một event loop.
        asyncio.run(main())

    except KeyboardInterrupt:
        print(
            "Đã dừng bởi người dùng.",
            file=sys.stderr,
        )
        raise SystemExit(130)

    except Exception as exception:
        logger.exception(
            "Question answering test thất bại."
        )
        print(
            f"\nLỖI: {exception}",
            file=sys.stderr,
        )
        raise SystemExit(1)