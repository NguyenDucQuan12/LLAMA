from __future__ import annotations

"""
-------
Tầng retrieval nhanh bằng dense vector:

    question
        -> chuẩn hóa
        -> thêm embedding_query_prefix
        -> Ollama tạo query vector
        -> Qdrant tìm nearest points
        -> chuyển ScoredPoint thành RetrievedChunk
        -> trả cho CrossEncoder reranker

File này không đọc trực tiếp PDF/DOCX. Để test với tài liệu thật,
tài liệu phải được ingest vào Qdrant trước.

Chạy từ thư mục gốc project:

    python3 -m app.retrieval.dense_document_retriever \
        --question "Làm thế nào để gọi robot nhận hàng?" \
        --tenant-id "wms" \
        --document-id "fabric-warehouse-guide" \
        --top-k 10
"""
import argparse
import asyncio
import inspect
import json
import logging
import math
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any
from qdrant_client import models

# Mở comment 3 dòng bên dưới mỗi khi test (Chạy trực tiếp hàm if __main__)
import os,sys
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from clients.ollama_client import OllamaClient
from clients.qdrant_repository import QdrantRepository
from config import Settings, get_settings
from retrieval.models import RetrievedChunk


# Logger mang tên module hiện tại.
logger = logging.getLogger(__name__)


class DenseDocumentRetriever:
    """
    Tạo embedding cho câu hỏi và tìm top K chunk trong Qdrant.
    """

    def __init__(self, settings: Settings, ollama_client: OllamaClient, qdrant_repository: QdrantRepository) -> None:
        """
        Nhận dependency từ bên ngoài để dùng chung connection pool.
        """

        # Lưu cấu hình.
        self.settings = settings
        # Lưu client Ollama dùng cho query embedding.
        self.ollama_client = ollama_client
        # Lưu repository dùng cho vector search.
        self.qdrant_repository = qdrant_repository
        # Kiểm tra cấu hình ngay khi tạo object.
        self._validate_configuration()

    async def retrieve(self, question: str, tenant_id: str, document_id: str | None = None, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Tìm các chunk gần nhất với câu hỏi.  
        `document_id=None` nghĩa là tìm trong toàn bộ tenant.
        """

        # Chuẩn hóa câu hỏi và kiểm tra kiểu.
        normalized_question = self._normalize_question(question)

        # Câu hỏi chỉ chứa khoảng trắng trở thành chuỗi rỗng.
        if not normalized_question:
            raise ValueError("Câu hỏi không được để trống.")

        # tenant_id là khóa phân vùng bắt buộc.
        normalized_tenant_id = self._normalize_identifier(tenant_id, field_name="tenant_id")

        # document_id là filter tùy chọn.
        normalized_document_id = self._normalize_optional_identifier(document_id, field_name="document_id")

        # Dùng top_k truyền vào hoặc giá trị mặc định trong Settings.
        selected_top_k = self._resolve_top_k(top_k)

        # Prefix query phải khớp với cách model embedding được thiết kế. Thêm "search_query"
        query_embedding_text = (self.settings.embedding_query_prefix + normalized_question).strip()

        # embed_texts câu hỏi của người dùng và trả list vector.
        embedding_response = await self.ollama_client.embed_texts([query_embedding_text])

        # Vì chỉ gửi một câu hỏi, phải nhận đúng một vector.
        query_vector = self._extract_query_vector(embedding_response)

        # Repository phải filter tenant, optional document_id,
        # trả payload và không cần trả vector gốc của point.
        scored_points = await self.qdrant_repository.search_chunks(
            query_vector=query_vector,
            tenant_id=normalized_tenant_id,
            top_k=selected_top_k,
            document_id=normalized_document_id,
        )

        # Bảo vệ ranh giới giữa repository và retriever.
        if not isinstance(scored_points, Sequence):
            raise TypeError(
                "search_chunks() phải trả Sequence[ScoredPoint]."
            )

        # Danh sách output cuối cùng.
        retrieved_chunks: list[RetrievedChunk] = []

        # Ngăn cùng point xuất hiện nhiều lần trong context.
        seen_point_ids: set[str] = set()

        # Threshold là tùy chọn; mặc định None.
        minimum_dense_score = self._get_minimum_dense_score()

        # Duyệt từng kết quả Qdrant.
        for result_index, scored_point in enumerate(scored_points):
            # Cho phép ScoredPoint tương thích Pydantic, nhưng bắt buộc phải có id, score và payload.
            if not (
                hasattr(scored_point, "id")
                and hasattr(scored_point, "score")
                and hasattr(scored_point, "payload")
            ):
                raise TypeError(f"Kết quả {result_index} không phải ScoredPoint hợp lệ.")

            # Chuyển sang model nội bộ.
            retrieved_chunk = self._convert_scored_point(scored_point)

            # Point không có text không thể dùng cho RAG.
            if retrieved_chunk is None:
                logger.warning(
                    "Bỏ Qdrant point tại vị trí %s vì thiếu text.",
                    result_index,
                )
                continue

            # Chuẩn hóa ID để dùng trong set.
            point_id = str(retrieved_chunk.point_id)

            # Bỏ ID trùng.
            if point_id in seen_point_ids:
                logger.warning("Bỏ point ID trùng: %s", point_id)
                continue

            # Bỏ điểm dưới threshold nếu đã cấu hình.
            if (
                minimum_dense_score is not None
                and retrieved_chunk.dense_score < minimum_dense_score
            ):
                continue

            # Ghi nhận ID và thêm chunk.
            seen_point_ids.add(point_id)
            retrieved_chunks.append(retrieved_chunk)

        # Qdrant thường đã sort giảm dần, nhưng sort lại để bảo đảm.
        retrieved_chunks.sort(
            key=lambda item: item.dense_score,
            reverse=True,
        )

        # Trả kết quả cho reranker.
        return retrieved_chunks

    def _convert_scored_point(self, scored_point: models.ScoredPoint) -> RetrievedChunk | None:
        """
        Chuyển ScoredPoint thành RetrievedChunk.
        """

        # Chuyển point ID thành string ổn định.
        point_id = str(scored_point.id).strip()

        # ID rỗng là dữ liệu lỗi.
        if not point_id:
            raise ValueError("Qdrant ScoredPoint có ID rỗng.")

        # True/False không được dùng làm score.
        if isinstance(scored_point.score, bool):
            raise TypeError(f"Point {point_id} có dense score boolean.")

        # Chuyển score sang float.
        try:
            dense_score = float(scored_point.score)
        except (TypeError, ValueError) as exception:
            raise TypeError(
                f"Point {point_id} có score không hợp lệ."
            ) from exception

        # Chặn NaN và Infinity.
        if not math.isfinite(dense_score):
            raise ValueError(
                f"Point {point_id} có score NaN/Infinity."
            )

        # Lấy payload.
        raw_payload = scored_point.payload

        # Payload None được coi là dictionary rỗng.
        if raw_payload is None:
            payload: dict[str, Any] = {}

        # Payload bình thường là Mapping.
        elif isinstance(raw_payload, Mapping):
            payload = dict(raw_payload)

        # Kiểu khác là lỗi contract của repository/client.
        else:
            raise TypeError(f"Point {point_id} có payload không phải Mapping.")

        # Text có heading/context thường tốt hơn raw text.
        contextualized_text = self._normalize_text(payload.get("contextualized_text"))

        # Nội dung chunk gốc.
        raw_text = self._normalize_text(payload.get("text"))

        # Ưu tiên contextualized_text.
        selected_text = contextualized_text or raw_text

        # Không có text thì bỏ point.
        if not selected_text:
            return None

        # Tạo model nội bộ.
        return RetrievedChunk(
            # ID point.
            point_id=point_id,
            # Similarity score từ Qdrant.
            dense_score=dense_score,
            # Metadata tài liệu có thể thiếu.
            document_id=self._optional_string(payload.get("document_id")),
            source_file=self._optional_string(payload.get("source_file")),
            source_hash=self._optional_string(payload.get("source_hash")),
            # chunk_index phải là int >= 0.
            chunk_index=self._optional_non_negative_integer(payload.get("chunk_index")),
            # Danh sách heading đã bỏ trùng.
            headings=self._normalize_string_list(payload.get("headings")),
            # Các trang nguyên dương, tăng dần.
            page_numbers=self._normalize_integer_list(payload.get("page_numbers")),
            # Link ngược về DoclingDocument.
            doc_item_refs=self._normalize_string_list(payload.get("doc_item_refs")),
            # Bảo đảm text không rỗng.
            text=raw_text or selected_text,
            # Dùng cho cross-encoder và Llama.
            contextualized_text=selected_text,
            # Giữ payload để tầng sau dùng metadata mở rộng.
            payload=payload,
        )

    def _normalize_question(self, question: str) -> str:
        """
        Chuẩn hóa Unicode/khoảng trắng nhưng giữ nguyên mã kỹ thuật.
        """

        # Báo lỗi rõ thay vì để unicodedata.normalize tự lỗi khó hiểu.
        if not isinstance(question, str):
            raise TypeError("Câu hỏi phải là string.")

        # Hợp nhất biểu diễn Unicode tiếng Việt.
        normalized_question = unicodedata.normalize("NFC", question)

        # Gộp tab/newline/nhiều dấu cách thành một dấu cách.
        return " ".join(normalized_question.split())

    def _normalize_text(self, value: Any) -> str:
        """
        Chuyển payload text thành Unicode NFC và giữ cấu trúc newline.
        """

        # None nghĩa là không có dữ liệu.
        if value is None:
            return ""

        # Bytes thường cho thấy payload sai kiểu.
        if isinstance(value, (bytes, bytearray)):
            raise TypeError("Text payload không được là bytes.")

        # Chuyển thành string và chuẩn hóa Unicode.
        text = unicodedata.normalize("NFC", str(value))

        # Chuẩn hóa newline Windows/macOS cũ thành \n.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Bỏ whitespace đầu/cuối nhưng giữ newline bên trong.
        return text.strip()

    def _optional_string(self, value: Any) -> str | None:
        """Trả string đã chuẩn hóa hoặc None nếu rỗng."""

        # Dùng helper chung.
        normalized_value = self._normalize_text(value)

        # Chuỗi rỗng được chuyển thành None.
        return normalized_value if normalized_value else None

    def _optional_non_negative_integer(
        self,
        value: Any,
    ) -> int | None:
        """Chuyển chunk_index thành int >= 0."""

        # int(True) == 1 nên cần chặn bool.
        if isinstance(value, bool):
            return None

        # Thử chuyển sang int.
        try:
            integer_value = int(value)
        except (TypeError, ValueError):
            return None

        # Chunk index âm là không hợp lệ.
        if integer_value < 0:
            return None

        # Trả số hợp lệ.
        return integer_value

    def _normalize_string_list(self, value: Any) -> list[str]:
        """
        Chuẩn hóa string/list/tuple/set thành list[str], bỏ rỗng và trùng.
        """

        # Không có dữ liệu.
        if value is None:
            return []

        # String là một item, không duyệt từng ký tự.
        if isinstance(value, str):
            raw_items: Sequence[Any] = [value]

        # List/tuple giữ thứ tự gốc.
        elif isinstance(value, (list, tuple)):
            raw_items = value

        # Set không ổn định thứ tự nên sort để output ổn định.
        elif isinstance(value, set):
            raw_items = sorted(value, key=str)

        # Kiểu đơn lẻ khác được bọc vào list.
        else:
            raw_items = [value]

        # Danh sách output.
        output: list[str] = []

        # Set dùng loại trùng.
        seen: set[str] = set()

        # Duyệt từng item.
        for item in raw_items:
            normalized_item = self._normalize_text(item)

            # Bỏ chuỗi rỗng.
            if not normalized_item:
                continue

            # Bỏ chuỗi trùng.
            if normalized_item in seen:
                continue

            # Ghi nhận và thêm output.
            seen.add(normalized_item)
            output.append(normalized_item)

        # Trả danh sách sạch.
        return output

    def _normalize_integer_list(self, value: Any) -> list[int]:
        """
        Chuẩn hóa page_numbers thành list số nguyên dương tăng dần.
        """

        # None -> danh sách rỗng.
        if value is None:
            raw_items: Sequence[Any] = []

        # Collection phổ biến -> list.
        elif isinstance(value, (list, tuple, set)):
            raw_items = list(value)

        # Giá trị đơn -> list một phần tử.
        else:
            raw_items = [value]

        # Set tự loại trang trùng.
        output: set[int] = set()

        # Duyệt từng giá trị.
        for item in raw_items:
            # Chặn True -> 1.
            if isinstance(item, bool):
                continue

            # Thử chuyển thành int.
            try:
                page_number = int(item)
            except (TypeError, ValueError):
                continue

            # Chỉ nhận trang từ 1 trở lên.
            if page_number > 0:
                output.add(page_number)

        # Trả tăng dần.
        return sorted(output)

    def _resolve_top_k(self, top_k: int | None) -> int:
        """Chọn top_k và giới hạn request quá lớn."""

        # Dùng argument nếu có, nếu không dùng Settings.
        selected_top_k = (
            top_k
            if top_k is not None
            else self.settings.retrieval_top_k
        )

        # bool là subclass int nhưng không hợp lệ.
        if isinstance(selected_top_k, bool) or not isinstance(
            selected_top_k,
            int,
        ):
            raise TypeError("top_k phải là int.")

        # Không nhận 0 hoặc số âm.
        if selected_top_k <= 0:
            raise ValueError("top_k phải lớn hơn 0.")

        # Giới hạn tối đa; field mới có default 100.
        maximum_top_k = getattr(self.settings, "retrieval_max_top_k", 100)

        # Kiểm tra cấu hình max.
        if isinstance(maximum_top_k, bool) or not isinstance(maximum_top_k, int) or maximum_top_k <= 0:
            raise ValueError(
                "retrieval_max_top_k phải là số lớn hơn 0."
            )

        # Ngăn request lấy quá nhiều point.
        if selected_top_k > maximum_top_k:
            raise ValueError(
                f"top_k={selected_top_k} vượt "
                f"retrieval_max_top_k={maximum_top_k}."
            )

        # Trả top K hợp lệ.
        return selected_top_k

    def _extract_query_vector(self, embedding_response: Any) -> list[float]:
        """Kiểm tra response embedding và lấy query vector."""

        # Output phải là list vector.
        if not isinstance(embedding_response, list):
            raise TypeError("Embedding response phải là list.")

        # Một input phải cho đúng một vector.
        if len(embedding_response) != 1:
            raise RuntimeError(
                "Embedding một câu hỏi phải trả đúng một vector, "
                f"nhưng nhận {len(embedding_response)}."
            )

        # Lấy vector đầu tiên.
        raw_vector = embedding_response[0]

        # Vector phải là list số.
        if not isinstance(raw_vector, list):
            raise TypeError("Query vector phải là list.")

        # Dimension mong đợi.
        expected_dimensions = self.settings.embedding_vector_dimensions

        # Kiểm tra cấu hình dimension.
        if isinstance(expected_dimensions, bool) or not isinstance(
            expected_dimensions,
            int,
        ) or expected_dimensions <= 0:
            raise ValueError(
                "embedding_vector_dimensions phải là số lớn hơn 0."
            )

        # Dimension phải khớp collection.
        if len(raw_vector) != expected_dimensions:
            raise RuntimeError(
                "Query vector sai dimension: "
                f"actual={len(raw_vector)}, "
                f"expected={expected_dimensions}."
            )

        # Vector đã chuẩn hóa thành float.
        vector: list[float] = []

        # Dùng norm để chặn zero vector.
        squared_norm = 0.0

        # Duyệt từng chiều.
        for value_index, value in enumerate(raw_vector):
            # Chặn bool và kiểu không phải số.
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    f"Query vector[{value_index}] không phải số hợp lệ."
                )

            # Chuyển về float.
            numeric_value = float(value)

            # Chặn NaN/Infinity.
            if not math.isfinite(numeric_value):
                raise ValueError(
                    f"Query vector[{value_index}] là NaN/Infinity."
                )

            # Thêm vào output.
            vector.append(numeric_value)

            # Cộng norm bình phương.
            squared_norm += numeric_value * numeric_value

        # Vector toàn số 0 không có hướng ngữ nghĩa.
        if squared_norm <= 0:
            raise RuntimeError("Ollama trả zero vector cho câu hỏi.")

        # Trả vector sạch.
        return vector

    def _normalize_identifier(self, value: str, field_name: str) -> str:
        """Chuẩn hóa tenant_id/document_id nhưng không lowercase."""

        # Kiểm tra kiểu.
        if not isinstance(value, str):
            raise TypeError(f"{field_name} phải là string.")

        # Chuẩn hóa Unicode và bỏ khoảng trắng ngoài.
        normalized = unicodedata.normalize("NFC", value).strip()

        # Chặn rỗng.
        if not normalized:
            raise ValueError(f"{field_name} không được rỗng.")

        # Chặn identifier bất thường quá dài.
        if len(normalized) > 500:
            raise ValueError(f"{field_name} tối đa 500 ký tự.")

        # Trả ID đã chuẩn hóa.
        return normalized

    def _normalize_optional_identifier(
        self,
        value: str | None,
        field_name: str,
    ) -> str | None:
        """Chuẩn hóa identifier tùy chọn."""

        # None được giữ nguyên.
        if value is None:
            return None

        # Có giá trị thì dùng validation bắt buộc.
        return self._normalize_identifier(value, field_name)

    def _get_minimum_dense_score(self) -> float | None:
        """Đọc retrieval_min_dense_score nếu đã cấu hình."""

        # Mặc định không lọc theo threshold.
        value = getattr(
            self.settings,
            "retrieval_min_dense_score",
            None,
        )

        # None nghĩa là tắt threshold.
        if value is None:
            return None

        # Chặn bool.
        if isinstance(value, bool):
            raise TypeError(
                "retrieval_min_dense_score không được là boolean."
            )

        # Chuyển thành float.
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exception:
            raise TypeError(
                "retrieval_min_dense_score phải là số hoặc None."
            ) from exception

        # Chặn NaN/Infinity.
        if not math.isfinite(numeric_value):
            raise ValueError(
                "retrieval_min_dense_score phải là số hữu hạn."
            )

        # Trả threshold.
        return numeric_value

    def _validate_configuration(self) -> None:
        """Kiểm tra Settings ngay khi tạo retriever."""

        # Đọc query prefix.
        query_prefix = getattr(
            self.settings,
            "embedding_query_prefix",
            None,
        )

        # Prefix phải là string.
        if not isinstance(query_prefix, str):
            raise TypeError(
                "embedding_query_prefix phải là string."
            )

        # Prefix rỗng thường làm sai cách dùng model Nomic.
        if not query_prefix:
            raise ValueError(
                "embedding_query_prefix không được rỗng."
            )

        # Kiểm tra top K mặc định và max.
        self._resolve_top_k(None)

        # Kiểm tra threshold nếu có.
        self._get_minimum_dense_score()

        # Kiểm tra dimension.
        dimensions = getattr(
            self.settings,
            "embedding_vector_dimensions",
            None,
        )

        if isinstance(dimensions, bool) or not isinstance(
            dimensions,
            int,
        ) or dimensions <= 0:
            raise ValueError(
                "embedding_vector_dimensions phải là int > 0."
            )


# -----------------------------------------------------------------------------
# Các helper dưới đây chỉ phục vụ CLI test với dữ liệu thật.
# -----------------------------------------------------------------------------


def _chunk_to_display_dict(
    chunk: RetrievedChunk,
    preview_characters: int,
) -> dict[str, Any]:
    """Rút gọn RetrievedChunk để in terminal."""

    # Lấy text có context.
    text = chunk.contextualized_text

    # Rút gọn nếu quá dài.
    if len(text) > preview_characters:
        text = (
            text[:preview_characters].rstrip()
            + "\n...[đã rút gọn phần xem trước]..."
        )

    # Chỉ in metadata hữu ích.
    return {
        "point_id": chunk.point_id,
        "dense_score": chunk.dense_score,
        "document_id": chunk.document_id,
        "source_file": chunk.source_file,
        "source_hash": chunk.source_hash,
        "chunk_index": chunk.chunk_index,
        "headings": chunk.headings,
        "page_numbers": chunk.page_numbers,
        "doc_item_refs": chunk.doc_item_refs,
        "contextualized_text_preview": text,
    }


async def _close_if_supported(resource: Any) -> None:
    """Đóng resource nếu có close(), hỗ trợ cả sync và async."""

    # Lấy method close.
    close_method = getattr(resource, "close", None)

    # Không hỗ trợ close thì bỏ qua.
    if not callable(close_method):
        return

    # Gọi close.
    result = close_method()

    # Nếu close trả awaitable thì await.
    if inspect.isawaitable(result):
        await result


async def test_real_retrieval(
    question: str,
    tenant_id: str,
    document_id: str | None,
    top_k: int | None,
    preview_characters: int,
) -> None:
    """
    Test trực tiếp với Ollama, Qdrant và tài liệu đã ingest thật.
    """

    # Đọc Settings thật.
    settings = get_settings()

    # Tạo Ollama client thật.
    ollama_client = OllamaClient(settings)

    # Tạo Qdrant repository thật.
    qdrant_repository = QdrantRepository(settings)

    # Tạo retriever thật.
    retriever = DenseDocumentRetriever(
        settings=settings,
        ollama_client=ollama_client,
        qdrant_repository=qdrant_repository,
    )

    try:
        # Kiểm tra collection tồn tại và dimension đúng.
        await qdrant_repository.ensure_collection()

        # Chạy retrieval thật.
        chunks = await retriever.retrieve(
            question=question,
            tenant_id=tenant_id,
            document_id=document_id,
            top_k=top_k,
        )

        # Chuẩn bị output.
        output = {
            "question": question,
            "tenant_id": tenant_id,
            "document_id": document_id,
            "requested_top_k": (
                top_k
                if top_k is not None
                else settings.retrieval_top_k
            ),
            "returned_count": len(chunks),
            "results": [
                _chunk_to_display_dict(
                    chunk,
                    preview_characters,
                )
                for chunk in chunks
            ],
        }

        # In tiêu đề.
        print()
        print("=" * 88)
        print("KẾT QUẢ DENSE RETRIEVAL THẬT")
        print("=" * 88)

        # In JSON tiếng Việt.
        print(
            json.dumps(
                output,
                ensure_ascii=False,
                indent=2,
            )
        )

        # Gợi ý kiểm tra khi không có kết quả.
        if not chunks:
            print()
            print("Không tìm thấy chunk. Kiểm tra:")
            print("1. Tài liệu đã ingest chưa.")
            print("2. tenant_id có khớp payload không.")
            print("3. document_id có khớp không.")
            print("4. search_chunks() có with_payload=True không.")
            print("5. Model và vector dimension có khớp không.")

    finally:
        # Đóng trong cùng event loop đã tạo và sử dụng client.
        await _close_if_supported(ollama_client)
        await _close_if_supported(qdrant_repository)


async def main() -> None:
    """
    Main chỉ dùng dữ liệu thật, không tạo candidate/vector giả.
    """

    # Bật logging.
    logging.basicConfig(
        level=getattr(logging, "INFO"),
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    # Chạy test thật.
    await test_real_retrieval(
        question="Cách gọi robot nhận hàng?",
        tenant_id="viva_factory",
        document_id=None,
        top_k=5,
        preview_characters=800,
    )


if __name__ == "__main__":
    try:
        # Chỉ tạo một event loop cho toàn bộ client async.
        asyncio.run(main())

    except KeyboardInterrupt:
        # Ctrl+C trả exit code chuẩn 130.
        print("Đã dừng bởi người dùng.", file=sys.stderr)
        raise SystemExit(130)

    except Exception as exception:
        # Ghi traceback đầy đủ vào log.
        logger.exception("Dense retrieval test thất bại.")

        # In thông báo ngắn cho người chạy terminal.
        print(f"\nLỖI: {exception}", file=sys.stderr)

        # Trả exit code lỗi.
        raise SystemExit(1)