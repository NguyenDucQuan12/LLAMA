from __future__ import annotations

"""
1. QuestionIntentRouter
   Nhận câu hỏi và quyết định:
       - documents: chỉ tìm tài liệu;
       - sql: chỉ chạy predefined SQL query;
       - hybrid: vừa tìm tài liệu vừa chạy predefined SQL.
"""
import asyncio
import inspect
import json
import logging
import math
import sys
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

# Mở comment 3 dòng bên dưới mỗi khi test (Chạy trực tiếp hàm if __main__)
import os,sys
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from clients.ollama_client import OllamaClient
from config import Settings, get_settings
from schemas import QuestionMode
from sql.query_registry import PredefinedSqlQueryRegistry
from sql.sql_service import SafeSqlServerService


# Logger của module.
logger = logging.getLogger(__name__)


# Python chỉ cho phép ba route thực thi.
_ALLOWED_ROUTES = {
    "documents",     # Chỉ sử dụng tài liệu để tìm kiếm câu trả lời
    "sql",           # Chỉ sử dụng SQL để truy vấn dữ liệu rồi đưa ra câu trả lời
    "hybrid",        # Kết hợp giữa document và sql
}

@dataclass(frozen=True)
class RouteDecision:
    """
    Quyết định cuối cùng đã qua kiểm tra Python.

    Ví dụ:

        RouteDecision(
            route="sql",
            query_key="get_robot_status",
            parameters={"robot_id": "AGV-01"},
            reason="Câu hỏi cần trạng thái hiện tại.",
            confidence=0.94,
        )
    """
    route: str                      # Route thực tế mà QuestionAnsweringService sẽ dùng.
    query_key: str | None           # Predefined query key. None nghĩa là không chạy SQL.
    parameters: dict[str, Any]      # Parameter chuẩn bị gửi sang SafeSqlServerService.
    reason: str                     # Lý do gọi 
    confidence: float = 1.0         # Độ tin cậy 0-1.


@dataclass(frozen=True)
class RouterLlamaTrace:
    """
    Dữ liệu debug của lần gọi Llama gần nhất.

    Trace cho phép quan sát toàn bộ quá trình:

        allowed_routes
        forced_query_key
        safe_catalog
        system_message
        user_message
        raw_output
        validated_decision
        elapsed_seconds
        error

    Không chứa SQL text vì safe_catalog đã loại bỏ SQL text.
    """
    allowed_routes: list[str]                       # Những route Llama được phép chọn trong lần gọi.
    forced_query_key: str | None                    # Query key bị caller khóa, nếu có.
    safe_catalog: list[dict[str, Any]]              # Catalog metadata an toàn gửi cho model.
    system_message: str                             # System prompt thực tế.
    user_message: str                               # User prompt thực tế.
    raw_output: Any                                 # JSON thô OllamaClient.chat_json() trả về.
    validated_decision: RouteDecision | None        # Quyết định sau validation.
    elapsed_seconds: float                          # Thời gian gọi Llama.
    error: str | None                               # Lỗi nếu lời gọi thất bại.

@dataclass(frozen=True)
class RouterTestCase:
    """
    Một ca kiểm tra viết trực tiếp trong main.

    enabled=False giúp giữ lại ví dụ mà không phải chạy mọi case,
    vì mỗi case AUTO/SQL/HYBRID có thể gọi Llama và mất thời gian.
    """
    name: str
    question: str
    mode: QuestionMode
    explicit_query_key: str | None = None
    explicit_parameters: dict[str, Any] | None = None
    enabled: bool = True


# JSON Schema buộc Llama trả đúng cấu trúc.
ROUTER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",                     # Output cấp cao nhất phải là JSON object.
    "properties": {                       # Các field được phép.
        "route": {
            "type": "string",             # route phải là string.
            "enum": [                     # Chỉ được chọn một trong ba giá trị.
                "documents",
                "sql",
                "hybrid",
            ],
        },
        "query_key": {
            "type": [    
                "string",                 # documents có thể dùng null.
                "null",                   # sql/hybrid phải dùng một string hợp lệ.
            ],
        },
        "parameters": {
            "type": "object",             # Parameter luôn là JSON object.
            "additionalProperties": True, # Tên parameter phụ thuộc query nên cho phép key động.
        },
        "reason": {
            "type": "string",             # Giải thích ngắn cho quyết định.
        },
        "confidence": {                   # Mức tin cậy.
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    # Llama phải trả đủ năm field.
    "required": [
        "route",
        "query_key",
        "parameters",
        "reason",
        "confidence",
    ],
    "additionalProperties": False,        # Không cho thêm field như sql, sql_text hoặc command.
}


class QuestionIntentRouter:
    """
    Router quyết định documents, sql hoặc hybrid.

    Llama chỉ là bộ phân loại và chọn key.
    Python mới là lớp cho phép hay từ chối quyết định.
    """

    def __init__( self, ollama_client: OllamaClient, query_registry: PredefinedSqlQueryRegistry,
                    sql_server_service: SafeSqlServerService, settings: Settings | None = None) -> None:
        """
        Khởi tạo router.

        ollama_client:
            Gọi structured JSON output.

        query_registry:
            Allowlist duy nhất của predefined SQL query.

        sql_server_service:
            Router chỉ dùng để kiểm tra SQL Server có bật không.
            Router không thực thi SQL.

        settings:
            Chứa timeout, confidence threshold và giới hạn input.
        """

        self.ollama_client = ollama_client
        self.query_registry = query_registry
        self.sql_server_service = sql_server_service
        self.settings = settings

        # Lưu trace gần nhất để main/debug endpoint có thể xem.
        self.last_llama_trace: RouterLlamaTrace | None = None

        # Phát hiện cấu hình sai ngay khi tạo router.
        self._validate_configuration()

    async def decide(self, question: str, mode: QuestionMode, explicit_query_key: str | None = None, explicit_parameters: Mapping[str, Any] | None = None) -> RouteDecision:
        """
        Chọn route theo thứ tự:

        1. Chuẩn hóa input.
        2. Ưu tiên chế độ được truyền vào từ người dùng.
        3. Ưu tiên lấy câu truy vấn nếu ngừoi dùng truyền đúng tên.
        4. Chỉ gọi Llama khi thật sự cần.
        5. Xác thực lại lần nữa output từ Llama.
        6. Fallback an toàn.

        Mỗi lần decide() bắt đầu, trace cũ được xóa để tránh hiểu nhầm.
        """

        # Trace cũ không còn liên quan tới câu hỏi mới. Cho nó về None
        self.last_llama_trace = None

        # Chuẩn hóa câu hỏi, mode, key gọi sql và các tham số có thể có
        normalized_question = self._normalize_question(question)
        normalized_mode = self._normalize_mode(mode)
        normalized_query_key = self._normalize_optional_query_key(explicit_query_key)
        normalized_parameters = self._normalize_parameters(explicit_parameters)

        # Kiểm tra SQL service có đang bật không.
        sql_enabled = bool(self.sql_server_service.is_enabled())

        # ====================================================
        # MODE DOCUMENTS: KHI NGƯỜI DÙNG CHỈ ĐỊNH TÌM KIẾM THÔNG TIN TỪ TÀI LIỆU ĐÃ CÓ
        # ====================================================

        if normalized_mode == "documents":
            # Documents không dùng SQL. Vì vậy đầu vào mâu thuẫn nên bị phát hiện thay vì bỏ qua âm thầm.
            if normalized_query_key is not None:
                raise ValueError("Mode DOCUMENTS không được kèm sql_query_key.")

            if normalized_parameters:
                raise ValueError("Mode DOCUMENTS không được kèm sql_parameters.")

            # Không gọi Llama. Trả về luôn cách thức sẽ trả lời câu hỏi của người dùng
            return RouteDecision(
                route="documents",
                query_key=None,
                parameters={},
                reason=("Caller yêu cầu chỉ tìm tài liệu."),
                confidence=1.0,
            )

        # ====================================================
        # MODE SQL
        # ====================================================

        if normalized_mode == "sql":
            # Caller đã ép SQL nên SQL Server phải sẵn sàng.
            if not sql_enabled:
                raise RuntimeError("Mode SQL được yêu cầu nhưng SQL Server chưa bật.")

            # ------------------------------------------------
            # Nếu người dùng truyền vào từ khoá sql chính xác
            # ------------------------------------------------

            if normalized_query_key is not None:
                # Kiểm tra key SQL này đã được đăng ký trong hệ thống chưa.
                self._require_registered_query(normalized_query_key)

                # Không gọi Llama vì caller đã chọn query.
                return RouteDecision(
                    route="sql",
                    query_key=normalized_query_key,
                    parameters=normalized_parameters,
                    reason=("Caller ép mode SQL và chỉ định predefined query."),
                    confidence=1.0,
                )

            # ------------------------------------------------
            # Khi người dùng không truywwfn vào tên sql, cho phép llama chọn câu truy vấn dựa vào câu hỏi
            # ------------------------------------------------

            # Llama chỉ được chọn route sql.
            # Nhiệm vụ chính của Llama ở đây là:
            # 1. chọn query_key từ catalog;
            # 2. trích parameter từ câu hỏi.
            automatic = await self._decide_with_llama(question=normalized_question, allowed_routes={"sql"}, forced_query_key=None)

            # SQL mode bắt buộc phải có key.
            if automatic.query_key is None:
                raise ValueError(
                    "Mode SQL nhưng router không chọn được query_key "
                    "hợp lệ. Hãy cung cấp sql_query_key."
                )

            # Parameter caller ghi đè parameter Llama.
            #
            # Ví dụ:
            # automatic.parameters = {"robot_id": "AGV-1"}
            # normalized_parameters = {"robot_id": "AGV-01"}
            #
            # Kết quả:
            # {"robot_id": "AGV-01"}
            return RouteDecision(
                route="sql",
                query_key=automatic.query_key,
                parameters={
                    **automatic.parameters,
                    **normalized_parameters,
                },
                reason=automatic.reason,
                confidence=automatic.confidence,
            )

        # ====================================================
        # MODE HYBRID
        # ====================================================

        if normalized_mode == "hybrid":
            # HYBRID luôn có document branch.
            # query_key quyết định có chạy thêm SQL không.

            # ------------------------------------------------
            # SQL chưa bật
            # ------------------------------------------------

            if not sql_enabled:
                # Caller đã gửi key nghĩa là caller yêu cầu SQL thật.
                if normalized_query_key is not None:
                    raise RuntimeError("HYBRID có sql_query_key nhưng SQL Server chưa bật.")

                # Parameter không có key để sử dụng là input mâu thuẫn.
                if normalized_parameters:
                    raise ValueError("HYBRID có sql_parameters nhưng SQL Server chưa bật và không có sql_query_key.")

                # Vẫn giữ route=hybrid nhưng query_key=None.
                # QuestionAnsweringService sẽ chỉ chạy documents.
                return RouteDecision(
                    route="hybrid",
                    query_key=None,
                    parameters={},
                    reason=("SQL Server chưa bật; HYBRID chỉ chạy document branch."),
                    confidence=1.0,
                )

            # ------------------------------------------------
            # HYBRID có explicit query key
            # ------------------------------------------------

            if normalized_query_key is not None:
                self._require_registered_query(normalized_query_key)

                # Không gọi Llama.
                return RouteDecision(
                    route="hybrid",
                    query_key=normalized_query_key,
                    parameters=normalized_parameters,
                    reason=(
                        "Caller yêu cầu HYBRID và chỉ định "
                        "predefined SQL query."
                    ),
                    confidence=1.0,
                )

            # ------------------------------------------------
            # HYBRID không có explicit query key
            # ------------------------------------------------

            # Llama xác định có cần SQL bổ sung không.
            automatic = await self._decide_with_llama(
                question=normalized_question,
                allowed_routes={
                    "documents",
                    "sql",
                    "hybrid",
                },
                forced_query_key=None,
            )

            # Llama chọn SQL hoặc hybrid và có key:
            # giữ route bên ngoài là hybrid vì caller đã ép hybrid.
            if (
                automatic.route
                in {"sql", "hybrid"}
                and automatic.query_key is not None
            ):
                return RouteDecision(
                    route="hybrid",
                    query_key=automatic.query_key,
                    parameters={
                        **automatic.parameters,
                        **normalized_parameters,
                    },
                    reason=(
                        "Caller yêu cầu HYBRID; router chọn thêm SQL. "
                        + automatic.reason
                    ),
                    confidence=automatic.confidence,
                )

            # Llama chọn documents hoặc không tìm được key:
            # hybrid chỉ chạy document branch.
            return RouteDecision(
                route="hybrid",
                query_key=None,
                parameters={},
                reason=(
                    "Caller yêu cầu HYBRID nhưng router không chọn "
                    "được SQL query đủ an toàn; chỉ chạy tài liệu. "
                    + automatic.reason
                ),
                confidence=automatic.confidence,
            )

        # ====================================================
        # MODE AUTO
        # ====================================================

        # ----------------------------------------------------
        # AUTO có explicit query key
        # ----------------------------------------------------

        if normalized_query_key is not None:
            if not sql_enabled:
                raise RuntimeError(
                    "AUTO nhận sql_query_key nhưng SQL Server chưa bật."
                )

            # Key caller phải nằm trong allowlist.
            self._require_registered_query(
                normalized_query_key
            )

            # Key đã bị khóa.
            # Llama chỉ quyết định:
            # - sql: chỉ dữ liệu;
            # - hybrid: dữ liệu + tài liệu.
            automatic = await self._decide_with_llama(
                question=normalized_question,
                allowed_routes={
                    "sql",
                    "hybrid",
                },
                forced_query_key=normalized_query_key,
            )

            return RouteDecision(
                route=(
                    "hybrid"
                    if automatic.route == "hybrid"
                    else "sql"
                ),
                query_key=normalized_query_key,
                parameters={
                    **automatic.parameters,
                    **normalized_parameters,
                },
                reason=(
                    "AUTO có query_key do caller chỉ định; "
                    "Llama chỉ chọn SQL hoặc HYBRID. "
                    + automatic.reason
                ),
                confidence=automatic.confidence,
            )

        # ----------------------------------------------------
        # AUTO nhưng SQL chưa bật
        # ----------------------------------------------------

        if not sql_enabled:
            if normalized_parameters:
                raise ValueError(
                    "AUTO có sql_parameters nhưng SQL Server chưa bật "
                    "và không có sql_query_key."
                )

            # Không gọi Llama vì SQL không thể chạy.
            return RouteDecision(
                route="documents",
                query_key=None,
                parameters={},
                reason=(
                    "SQL Server chưa bật; AUTO dùng tài liệu."
                ),
                confidence=1.0,
            )

        # ----------------------------------------------------
        # AUTO hoàn toàn
        # ----------------------------------------------------

        # Llama được phép chọn cả ba route.
        automatic = await self._decide_with_llama(
            question=normalized_question,
            allowed_routes={
                "documents",
                "sql",
                "hybrid",
            },
            forced_query_key=None,
        )

        # SQL/hybrid trong AUTO cần vượt confidence threshold.
        if automatic.route in {
            "sql",
            "hybrid",
        }:
            minimum_confidence = (
                self._minimum_sql_confidence()
            )

            if (
                automatic.confidence
                < minimum_confidence
            ):
                logger.info(
                    "SQL confidence %.3f thấp hơn threshold %.3f; "
                    "fallback documents.",
                    automatic.confidence,
                    minimum_confidence,
                )

                return RouteDecision(
                    route="documents",
                    query_key=None,
                    parameters={},
                    reason=(
                        "Router có dấu hiệu SQL nhưng confidence thấp; "
                        "fallback tài liệu. "
                        + automatic.reason
                    ),
                    confidence=automatic.confidence,
                )

        # Caller có thể cung cấp parameters dù không cung cấp key.
        # Chỉ merge khi Llama thực sự chọn SQL/hybrid và có key.
        if (
            automatic.route
            in {"sql", "hybrid"}
            and automatic.query_key is not None
            and normalized_parameters
        ):
            automatic = RouteDecision(
                route=automatic.route,
                query_key=automatic.query_key,
                parameters={
                    **automatic.parameters,
                    **normalized_parameters,
                },
                reason=automatic.reason,
                confidence=automatic.confidence,
            )

        return automatic

    async def _decide_with_llama(self, question: str, allowed_routes: set[str], forced_query_key: str | None) -> RouteDecision:
        """
        Gọi Llama để đề xuất route, query key và parameter.

        Đây là nơi Llama thật sự lựa chọn SQL.

        Quy trình:

        1. Lấy catalog metadata từ registry.
        2. Loại SQL text khỏi catalog.
        3. Tạo system prompt giải thích documents/sql/hybrid.
        4. Tạo user prompt gồm question + safe catalog.
        5. Gọi Ollama structured output.
        6. Validate JSON Llama.
        7. Lưu trace để main in ra.
        """

        # allowed_routes không được rỗng và phải nằm trong allowlist.
        if (not allowed_routes or not allowed_routes.issubset( _ALLOWED_ROUTES)):
            raise ValueError("allowed_routes không hợp lệ.")

        # Lấy danh sách các câu truy vấn đã đăng ký từ trước.
        raw_catalog = self.query_registry.list_for_router()

        # Chỉ giữ metadata an toàn. Không gửi trực tiếp các chuỗi SQL vào modle, dễ bị lộ
        safe_catalog = self._sanitize_catalog_for_llama(raw_catalog)

        # Sau khi làm sạch dữ liệu mà không còn câu truy vấn nào có thể được sử dụng thì thông báo
        if not safe_catalog:
            # Nếu có chế độ document thì chuyển sang document và thông báo không thể sử dụng SQL
            if "documents" in allowed_routes:
                decision = RouteDecision(
                    route="documents",
                    query_key=None,
                    parameters={},
                    reason=("SQL query catalog rỗng; dùng tài liệu."),
                    confidence=1.0,
                )

                # Ghi lại thông tin lần gọi llama mới nhất
                self.last_llama_trace = RouterLlamaTrace(
                    allowed_routes=sorted(allowed_routes),
                    forced_query_key=(forced_query_key),
                    safe_catalog=[],
                    system_message="",
                    user_message="",
                    raw_output=None,
                    validated_decision=decision,
                    elapsed_seconds=0.0,
                    error=("Danh sách câu truy vấn rỗng; không gọi Llama."),
                )

                return decision

            raise RuntimeError("Cần chọn SQL nhưng danh sách câu truy vấn đang rỗng.")

        # Nếu người dùng truyền vào từ khoá câu truy vấn, tiến hành tìm kiếm trong danh sách khai báo có câu truy vấn này không
        if forced_query_key is not None:
            self._require_registered_query(forced_query_key)

        # Chuyển set chứa các loại ["document", "sql", "hybrid"] thành chuỗi để đưa vào prompt.
        allowed_routes_text = ", ".join(
            sorted(allowed_routes)
        )

        # Quy tắc riêng khi người dùng đã chỉ định trực tiếp câu lệnh sql
        forced_key_rule = (
            (
                "Người dùng đã lựa chọn các query_key là "
                f"`{forced_query_key}`. "
                "Không được thay đổi các từ khoá này."
            )
            if forced_query_key is not None
            else (
                "Nếu chọn sql/hybrid, query_key phải xuất hiện trong SQL_QUERY_CATALOG."
            )
        )

        # System prompt định nghĩa nhiệm vụ và các ràng buộc.
        system_message = f"""
        Bạn là bộ định tuyến an toàn cho hệ thống RAG và SQL Server.

        ROUTE ĐƯỢC PHÉP TRONG LẦN NÀY:
        {allowed_routes_text}

        Ý NGHĨA CỦA CÁC ROUTER:
        - documents:
        Chọn khi câu hỏi hỏi về quy trình, hướng dẫn, nguyên nhân,
        cách thao tác, giải thích hoặc kiến thức ổn định trong tài liệu.

        - sql:
        Chọn khi câu hỏi chỉ cần dữ liệu hiện tại/cụ thể trong database,
        ví dụ trạng thái robot hiện tại, vị trí pallet, nhiệm vụ đang chạy,
        dữ liệu theo QR, số lượng hoặc lỗi gần đây.

        - hybrid:
        Chọn khi câu hỏi vừa cần dữ liệu hiện tại từ SQL,
        vừa cần quy trình/hướng dẫn/giải thích từ tài liệu.
        Ví dụ: "AGV-01 đang lỗi gì và cần kiểm tra theo quy trình nào?"

        CÁCH CHỌN QUERY_KEY:
        - Đọc description của từng query trong SQL_QUERY_CATALOG.
        - So sánh ý nghĩa câu hỏi với description.
        - Chọn query_key phù hợp có trong catalog.
        - Không tạo query_key mới.

        CÁCH TRÍCH PARAMETER:
        - Đọc danh sách required_parameters/optional_parameters của query.
        - Chỉ lấy giá trị xuất hiện rõ ràng trong câu hỏi.
        - Giữ nguyên mã kỹ thuật và chữ hoa/thường.
        - Không suy đoán giá trị không xuất hiện.
        - Nếu thiếu parameter bắt buộc, vẫn có thể chọn query_key,
        nhưng không tự bịa parameter; SQL service sẽ yêu cầu bổ sung.

        QUY TẮC BẮT BUỘC:
        1. Chỉ chọn route thuộc danh sách được phép.
        2. Không tạo hoặc trả SQL text.
        3. Không tạo query_key mới.
        4. {forced_key_rule}
        5. Không suy đoán mã robot, pallet, QR, ngày hoặc trạng thái.
        6. Nếu không chắc chắn và documents được phép, chọn documents.
        7. confidence nằm trong [0,1].
        8. Trả đúng JSON schema.
        """.strip()

        # User prompt chứa câu hỏi và catalog an toàn để llama quyết định chọn câu truy vấn nào
        user_message = (
            "CÂU HỎI:\n"
            + question
            + "\n\nSQL_QUERY_CATALOG:\n"
            + json.dumps(
                safe_catalog,
                ensure_ascii=False,
                indent=2,
            )
        )

        # Đo thời gian model.
        started_at = time.perf_counter()

        try:
            # Timeout riêng cho intent router.
            async with asyncio.timeout(self._router_timeout_seconds()):
                router_output = (
                    await self.ollama_client
                    .chat_json(
                        messages=[
                            {   # Đưa ra luật và nhiệm vụ cấp cao nhất để model tuân theo
                                "role": "system",
                                "content": (system_message),
                            },
                            {  # Câu hỏi cần model lựa chọn
                                "role": "user",
                                "content": (user_message),
                            },
                        ],
                        json_schema=(
                            ROUTER_JSON_SCHEMA
                        ),
                        temperature=0.0,   # Không cho phép model sáng tạo câu trả lời
                    )
                )

            # Python kiểm tra JSON Llama.
            validated = (
                self._validate_llama_decision(
                    router_output=(router_output),
                    allowed_routes=(allowed_routes),
                    forced_query_key=(forced_query_key),
                )
            )

            elapsed = (
                time.perf_counter()
                - started_at
            )

            # Lưu trace thành công.
            self.last_llama_trace = RouterLlamaTrace(
                allowed_routes=sorted(
                    allowed_routes
                ),
                forced_query_key=(
                    forced_query_key
                ),
                safe_catalog=safe_catalog,
                system_message=system_message,
                user_message=user_message,
                raw_output=router_output,
                validated_decision=validated,
                elapsed_seconds=elapsed,
                error=None,
            )

            return validated

        except Exception as exception:
            elapsed = (
                time.perf_counter()
                - started_at
            )

            # Lưu trace lỗi.
            self.last_llama_trace = RouterLlamaTrace(
                allowed_routes=sorted(
                    allowed_routes
                ),
                forced_query_key=(
                    forced_query_key
                ),
                safe_catalog=safe_catalog,
                system_message=system_message,
                user_message=user_message,
                raw_output=None,
                validated_decision=None,
                elapsed_seconds=elapsed,
                error=(
                    f"{type(exception).__name__}: "
                    f"{exception}"
                ),
            )

            # AUTO/HYBRID có documents thì fallback an toàn.
            if "documents" in allowed_routes:
                logger.exception(
                    "Llama router lỗi; fallback documents."
                )

                decision = RouteDecision(
                    route="documents",
                    query_key=None,
                    parameters={},
                    reason=(
                        "Llama router lỗi; fallback an toàn "
                        "sang tài liệu."
                    ),
                    confidence=0.0,
                )

                # Cập nhật trace với quyết định fallback cuối.
                self.last_llama_trace = RouterLlamaTrace(
                    allowed_routes=sorted(
                        allowed_routes
                    ),
                    forced_query_key=(
                        forced_query_key
                    ),
                    safe_catalog=safe_catalog,
                    system_message=system_message,
                    user_message=user_message,
                    raw_output=None,
                    validated_decision=decision,
                    elapsed_seconds=elapsed,
                    error=(
                        f"{type(exception).__name__}: "
                        f"{exception}"
                    ),
                )

                return decision

            # SQL-only không được fallback documents.
            raise

    def _validate_llama_decision(self, router_output: Any, allowed_routes: set[str], forced_query_key: str | None) -> RouteDecision:
        """
        Kiểm tra output Llama trước khi cho SQL chạy.

        Ví dụ output thô:

            {
                "route": "sql",
                "query_key": "get_robot_status",
                "parameters": {
                    "robot_id": "AGV-01"
                },
                "reason": "...",
                "confidence": 0.92
            }
        """

        # Output phải là object/dict.
        if not isinstance(router_output, Mapping):
            return self._safe_fallback(allowed_routes, "Router output không phải JSON object.")

        # Lấy route, mặc định documents.
        route = str(
            router_output.get(
                "route",
                "documents",
            )
        ).strip().lower()

        # Route phải nằm trong allowed_routes của lần gọi.
        if route not in allowed_routes:
            return self._safe_fallback(
                allowed_routes,
                (
                    f"Router trả route {route!r} "
                    "ngoài allowed_routes."
                ),
            )

        # Chuẩn hóa reason.
        reason = str(
            router_output.get(
                "reason",
                "",
            )
        ).strip()

        if not reason:
            reason = (
                "Router không cung cấp lý do."
            )

        # Chuẩn hóa confidence.
        confidence = (
            self._normalize_confidence(
                router_output.get(
                    "confidence",
                    0.0,
                )
            )
        )

        # Chuẩn hóa parameter.
        parameters = (
            self._normalize_parameters(
                router_output.get(
                    "parameters"
                )
            )
        )

        # Documents không được mang query key hoặc parameters.
        if route == "documents":
            return RouteDecision(
                route="documents",
                query_key=None,
                parameters={},
                reason=reason,
                confidence=confidence,
            )

        # SQL/hybrid phải có query_key string.
        raw_query_key = (
            router_output.get(
                "query_key"
            )
        )

        if not isinstance(
            raw_query_key,
            str,
        ):
            return self._safe_fallback(
                allowed_routes,
                (
                    "Router chọn SQL/HYBRID nhưng "
                    "query_key không phải string."
                ),
                confidence=confidence,
            )

        # Bỏ khoảng trắng đầu/cuối.
        query_key = raw_query_key.strip()

        if not query_key:
            return self._safe_fallback(
                allowed_routes,
                (
                    "Router chọn SQL/HYBRID nhưng "
                    "query_key đang rỗng."
                ),
                confidence=confidence,
            )

        # Caller đã khóa key thì Llama không được đổi.
        if (
            forced_query_key is not None
            and query_key
            != forced_query_key
        ):
            logger.warning(
                "Llama đổi forced key %s thành %s; "
                "Python giữ key caller.",
                forced_query_key,
                query_key,
            )

            query_key = forced_query_key

        # Query key phải tồn tại trong registry.
        try:
            self._require_registered_query(
                query_key
            )
        except KeyError:
            logger.warning(
                "Llama tạo query_key ngoài allowlist: %s",
                query_key,
            )

            return self._safe_fallback(
                allowed_routes,
                (
                    "Router trả query_key ngoài allowlist."
                ),
                confidence=confidence,
            )

        # Chỉ sau tất cả validation mới tạo RouteDecision SQL/hybrid.
        return RouteDecision(
            route=route,
            query_key=query_key,
            parameters=parameters,
            reason=reason,
            confidence=confidence,
        )

    def _safe_fallback(self, allowed_routes: set[str], reason: str, confidence: float = 0.0) -> RouteDecision:
        """
        Fallback khi câu trả lời của Llama không an toàn.
        Nếu router:  
        AUTO/HYBRID thường cho phép documents:
            => fallback documents.

        SQL-only chỉ có allowed_routes={"sql"}:
            => không được fallback documents;
            => trả về ValueError.
        """

        if "documents" in allowed_routes:
            return RouteDecision(
                route="documents",
                query_key=None,
                parameters={},
                reason=(
                    reason
                    + " Fallback tài liệu."
                ),
                confidence=confidence,
            )

        raise ValueError(
            reason
            + " Không thể fallback documents; hãy cung cấp chính xác từ khoá mẫu câu truy vấn cơ sở dữ liệu."
        )

    def _sanitize_catalog_for_llama( self, raw_catalog: Any, ) -> list[dict[str, Any]]:
        """
        Tạo danh sách mẫu truy vấn đã được đăng ký một cách an toàn gửi Llama.

        Chỉ giữ:
        - query_key;
        - description;
        - parameters;
        - required_parameters;
        - optional_parameters.

        Không gửi các trường dưới đây vì có thể lộ cơ sở dữ liệu:
        - sql;
        - sql_text;
        - statement;
        - query_text;
        - connection_string.
        """

        if raw_catalog is None:
            return []

        # Catalog phải là sequence nhưng không phải string/bytes.
        if (
            not isinstance(raw_catalog, Sequence)
            or isinstance(raw_catalog, (str, bytes, bytearray))
        ):
            raise TypeError(
                "query_registry.list_for_router() phải trả Sequence."
            )

        output: list[dict[str, Any]] = []

        # Duyệt danh sách mẫu truy vấn csdl đã lọc lần 1
        for index, item in enumerate(raw_catalog):
            # Mỗi item phải là mapping.
            if not isinstance(item, Mapping):
                raise TypeError(
                    f"Catalog item {index} không phải Mapping."
                )

            # Lấy từ khoá của mỗi câu truy vấn dựa vào từ khoá là key hoặc query_key
            query_key = (item.get("query_key") or item.get("key"))

            if (
                not isinstance(query_key, str)
                or not query_key.strip()
            ):
                raise ValueError(
                    f"Mẫu truy vấn CSDL {index} thiếu từ khoá để xác định."
                )
            # Chuẩn hoá key
            normalized_key = query_key.strip()

            # Chỉ lấy mẫu câu truy vấn DUY NHẤT TẠI DANH SÁCH ĐÃ ĐĂNG KÝ TRƯỚC ĐÓ TẠI TỆP QUERY_REGISTRY
            self._require_registered_query(normalized_key)

            # Tạo danh sách mới các mẫu truy vấn an toàn, tránh để lộ thông tin nhạy cảm
            safe_item: dict[str, Any] = {
                "query_key": normalized_key,
                "description": str(
                    item.get(
                        "description",
                        "",
                    )
                ).strip(),
            }

            # Copy parameter metadata nếu có.
            for field_name in (
                "parameters",
                "required_parameters",
                "optional_parameters",
            ):
                if field_name in item:
                    safe_item[field_name] = self._make_json_safe(item[field_name])

            # Thêm mẫu truy vấn đã làm sạch vào danh sách trả về
            output.append(safe_item)

        return output

    def _make_json_safe(self, value: Any, depth: int = 0) -> Any:
        """
        Chuyển metadata thành kiểu JSON đơn giản.

        depth > 5:
            chuyển thành string để tránh object lồng quá sâu.
        """

        if depth > 5:
            return str(value)

        if (
            value is None
            or isinstance(value, (str, int, float, bool))
        ):
            return value

        if isinstance(value, Mapping):
            return {
                str(key): self._make_json_safe(
                    child,
                    depth + 1,
                )
                for key, child
                in value.items()
            }

        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
        ):
            return [
                self._make_json_safe(child, depth + 1)
                for child in value
            ]

        return str(value)

    def _require_registered_query(self, query_key: str) -> Any:
        """
        Tìm kiếm trong danh sách khai báo các mẫu truy vấn, có câu truy vấn nào có từ khoá là: query_key
        """

        try:
            # Lấy câu truy vấn được khai báo trước với từ khoá là: query_key
            return self.query_registry.get(query_key)
        except KeyError:
            # Giữ KeyError để caller biết key không tồn tại.
            raise
        except Exception as exception:
            # Lỗi registry khác được mô tả rõ.
            raise RuntimeError(
                f"Xảy ra lỗi khi tìm kiếm lệnh truy vấn với từ khoá: {query_key!r}."
            ) from exception

    def _normalize_question(self, question: str) -> str:
        """
        Chuẩn hóa câu hỏi.

        Ví dụ:
            "  Robot   AGV-01\\nđang ở đâu? "
        thành:
            "Robot AGV-01 đang ở đâu?"

        Không lowercase để giữ nguyên:
            AGV-01
            F3-29
            F260406000151
        """

        if not isinstance(question, str):
            raise TypeError("Câu hỏi phải là chuỗi văn bản.")

        # Chuẩn hóa Unicode tiếng Việt.
        normalized = (
            unicodedata.normalize("NFC", question)
        )

        # Gộp mọi khoảng trắng.
        normalized = " ".join(normalized.split())

        if not normalized:
            raise ValueError("Câu hỏi sau khi chuẩn hoá không được rỗng.")

        maximum = self._positive_setting("router_max_question_characters", 4000)

        if len(normalized) > maximum:
            raise ValueError(
                "question quá dài: "
                f"{len(normalized)} > "
                f"{maximum}."
            )

        return normalized

    def _normalize_mode(self, mode: QuestionMode) -> str:
        """
        Chuyển enum chế độ câu hỏi thành:
            auto
            documents
            sql
            hybrid
        """

        if mode is None:
            raise TypeError("Chế độ trả lời câu hỏi không được là None.")

        # Enum có .value.
        raw_value = (
            mode.value
            if hasattr(mode, "value")
            else mode
        )

        normalized = str(raw_value).strip().lower()

        # Xử lý "QuestionMode.AUTO".
        if "." in normalized:
            normalized = (normalized.rsplit(".", 1)[-1])

        if normalized not in {"auto", "documents", "sql", "hybrid"}:
            raise ValueError(f"QuestionMode không hợp lệ:{normalized!r}.")

        return normalized

    def _normalize_optional_query_key(self, value: str | None) -> str | None:
        """
        Chuẩn hóa key của câu truy vấn
        """

        if value is None:
            return None

        if not isinstance(value, str):
            raise TypeError("Tên khoá truy vấn phải là string hoặc None.")

        normalized = value.strip()

        if not normalized:
            raise ValueError("Tên khoá câu truy vấn không được rỗng.")

        maximum = self._positive_setting("router_max_query_key_characters", 200)

        if len(normalized) > maximum:
            raise ValueError("Tên khoá câu truy vấn quá dài.")

        return normalized

    def _normalize_parameters(self, value: Any) -> dict[str, Any]:
        """
        Chuẩn hóa tham số trong SQL.

        Chỉ hỗ trợ:
        - None;
        - str;
        - int;
        - float hữu hạn;
        - bool;
        - list/tuple chứa các kiểu trên.

        Không hỗ trợ dict lồng nhau.
        """

        if value is None:
            return {}

        if not isinstance(value, Mapping):
            raise TypeError("Tham số phải là Mapping hoặc None.")
        # Giới hạn số lượng tham số cho một câu truy vấn 
        maximum_count = self._positive_setting("router_max_parameter_count",20,)

        if len(value) > maximum_count:
            raise ValueError(f"Quá nhiều tham số cho một câu truy vấn: {len(value)} > {maximum_count}.")

        output: dict[str, Any] = {}

        for raw_key, raw_value in (value.items()):
            if not isinstance(raw_key, str):
                raise TypeError("Tên tham só truyền vào phải là string.")

            key = raw_key.strip()

            if not key:
                raise ValueError("Tên của tham số truy vấn không được rỗng.")

            output[key] = (
                self._normalize_parameter_value(raw_value, key))

        return output

    def _normalize_parameter_value(self, value: Any, parameter_name: str) -> Any:
        """
        Chuẩn hóa một giá trị tham số truyền vào mẫu truy vấn CSDL.
        """

        if value is None:
            return None

        if isinstance(
            value,
            str,
        ):
            normalized = value.strip()

            maximum = self._positive_setting(
                "router_max_parameter_string_characters",
                500,
            )

            if len(normalized) > maximum:
                raise ValueError(
                    f"Parameter {parameter_name!r} "
                    "quá dài."
                )

            return normalized

        # Kiểm tra bool trước int vì bool là subclass của int.
        if isinstance(
            value,
            bool,
        ):
            return value

        if isinstance(
            value,
            int,
        ):
            return value

        if isinstance(
            value,
            float,
        ):
            if not math.isfinite(value):
                raise ValueError(
                    f"Parameter {parameter_name!r} "
                    "là NaN/Infinity."
                )

            return value

        if (
            isinstance(
                value,
                Sequence,
            )
            and not isinstance(
                value,
                (str, bytes, bytearray),
            )
        ):
            maximum_items = (
                self._positive_setting(
                    "router_max_parameter_list_items",
                    100,
                )
            )

            if len(value) > maximum_items:
                raise ValueError(
                    f"List parameter "
                    f"{parameter_name!r} quá dài."
                )

            return [
                self._normalize_parameter_value(
                    child,
                    parameter_name,
                )
                for child in value
            ]

        raise TypeError(
            f"Parameter {parameter_name!r} "
            "có kiểu không hỗ trợ: "
            f"{type(value).__name__}."
        )

    def _normalize_confidence(self, value: Any) -> float:
        """
        Chuyển giá trị ngưỡng thành kiểu float và nằm trong [0,1].

        Ví dụ:
            "0.8" -> 0.8
            2.0 -> 1.0
            -1 -> 0.0
            NaN -> 0.0
            True -> 0.0
        """

        if isinstance(value, bool):
            return 0.0

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0

        if not math.isfinite(numeric):
            return 0.0

        return min(1.0, max(0.0, numeric))

    def _minimum_sql_confidence(self) -> float:
        """
        Ngưỡng AUTO được phép chọn SQL/hybrid.

        Default:
            0.65
        """
        # Giá trị ngưỡng được quy định trong cấu hình nếu có
        raw_value = self._setting("router_min_sql_confidence", 0.65)

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exception:
            raise TypeError("router_min_sql_confidence phải là số.") from exception

        if (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("router_min_sql_confidence phải trong khoảng [0,1].")

        return value

    def _router_timeout_seconds(self,) -> float:
        """
        Timeout cho việc lựa chọn cách truy vấn dữ liệu: document, sql hay hybrid
        """

        raw_value = self._setting("router_timeout_seconds",600.0,)

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exception:
            raise TypeError(f"router_timeout_seconds phải là số. Không được sử dụng: {raw_value}") from exception

        if (not math.isfinite(value) or value <= 0):
            raise ValueError("router_timeout_seconds phải là số lớn hơn 0.")

        return value

    def _positive_setting(self, name: str, default: int) -> int:
        """
        Đọc các giá trị cấu hình trong setting với yêu cầu là số nguyên lớn hơn 0
        """
        # Lấy giá trị trong setting
        value = self._setting(name, default)

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"{name} phải là số nguyên lớn hơn 0.")

        return value

    def _setting(self, name: str, default: Any) -> Any:
        """
        Lấy giá trị được cấu hình trong setting với giá trị default nếu không lấy được
        """

        if self.settings is None:
            return default

        return getattr(self.settings, name, default)

    def _validate_configuration(self) -> None:
        """
        Validate cấu hình lúc khởi tạo.
        """

        self._positive_setting("router_max_question_characters", 4000)
        self._positive_setting("router_max_query_key_characters", 200)
        self._positive_setting("router_max_parameter_count", 20)
        self._positive_setting("router_max_parameter_string_characters", 500)
        self._positive_setting("router_max_parameter_list_items", 100)

        self._minimum_sql_confidence()
        self._router_timeout_seconds()


# ============================================================
# TEST
# ============================================================

T = TypeVar("T")


def _construct_component( component_class: type[T], dependencies: Mapping[str, Any]) -> T:
    """
    Dựng component bằng tên tham số constructor.

    Ví dụ:
        SafeSqlServerService(
            settings=settings,
            query_registry=query_registry,
        )
    """
    # Lấy thông tin hàm __init__ của class (constructor)
    signature = inspect.signature(component_class)


    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    # signature.parameters là danh sách các tham số của constructor.
    for name, parameter in (signature.parameters.items()):
        # Bỏ qua *args và **kwargs vì những tham số này không thể truy vấn trực tiếp bằng tên
        if parameter.kind in {inspect.Parameter.VAR_POSITIONAL,inspect.Parameter.VAR_KEYWORD,}:
            continue

        # Có dependency cùng tên.
        # Ví dụ nếu name == "settings" và dependencies["settings"] tồn tại, thì kwargs["settings"] = dependencies["settings"]
        if name in dependencies:
            kwargs[name] = (dependencies[name])

        # Không có dependency và không có default.
        elif (parameter.default is inspect.Parameter.empty):
            missing.append(name)

    if missing:
        raise RuntimeError(
            "Không dựng được "
            f"{component_class.__name__}; "
            "thiếu dependency: "
            + ", ".join(missing)
            + ". Hãy bổ sung trong "
            "_build_real_router()."
        )

    # Khởi tạo class với các tham số đã thu thập được
    return component_class(
        **kwargs
    )


async def _close_if_supported(resource: Any) -> None:
    """
    Đóng resource hỗ trợ aclose() hoặc close().
    """

    if resource is None:
        return

    for method_name in ("aclose", "close"):
        close_method = getattr(resource, method_name, None)

        if not callable(close_method):
            continue

        result = close_method()

        if inspect.isawaitable(result):
            await result

        return

async def _build_real_router(settings: Settings) -> tuple[QuestionIntentRouter, list[Any]]:
    """
    Dựng router thật.
    """

    resources: list[Any] = []

    try:
        # Ollama client thật.
        ollama_client = OllamaClient(settings)
        resources.append(ollama_client)

        dependencies: dict[str, Any] = {
            "settings": settings,
            "ollama_client": ollama_client,
        }

        # Registry thật.
        query_registry = _construct_component(PredefinedSqlQueryRegistry, dependencies)

        dependencies["query_registry"] = query_registry

        # SQL service thật.
        sql_server_service = _construct_component( SafeSqlServerService, dependencies)
        resources.append(sql_server_service)

        # Router thật.
        router = QuestionIntentRouter(
            ollama_client=ollama_client,
            query_registry=query_registry,
            sql_server_service=sql_server_service,
            settings=settings,
        )

        return (
            router,
            resources,
        )

    except Exception:
        # Đóng những resource đã tạo nếu startup giữa chừng lỗi.
        for resource in reversed(resources):
            try:
                await _close_if_supported(resource)
            except Exception:
                logger.exception("Không đóng được resource %s.", type(resource).__name__)

        raise


def _print_json(title: str, value: Any) -> None:
    """
    In object dưới dạng JSON UTF-8.
    """

    print()
    print("=" * 88)
    print(title)
    print("=" * 88)
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def _print_trace(trace: RouterLlamaTrace | None) -> None:
    """
    In dữ liệu Llama đã nhận và trả về.

    Trace=None nghĩa là nhánh được code quyết định trực tiếp,
    ví dụ mode=documents hoặc SQL có explicit query key.
    """

    if trace is None:
        print()
        print(
            "LLAMA TRACE: Không gọi Llama. "
            "Python/caller đã quyết định trực tiếp."
        )
        return

    _print_json(
        "1. ALLOWED ROUTES + FORCED KEY",
        {
            "allowed_routes": (
                trace.allowed_routes
            ),
            "forced_query_key": (
                trace.forced_query_key
            ),
            "elapsed_seconds": (
                trace.elapsed_seconds
            ),
            "error": trace.error,
        },
    )

    _print_json(
        "2. SAFE SQL CATALOG GỬI CHO LLAMA",
        trace.safe_catalog,
    )

    print()
    print("=" * 88)
    print("3. SYSTEM MESSAGE GỬI CHO LLAMA")
    print("=" * 88)
    print(trace.system_message)

    print()
    print("=" * 88)
    print("4. USER MESSAGE GỬI CHO LLAMA")
    print("=" * 88)
    print(trace.user_message)

    _print_json(
        "5. JSON THÔ LLAMA TRẢ VỀ",
        trace.raw_output,
    )

    _print_json(
        "6. QUYẾT ĐỊNH SAU PYTHON VALIDATION",
        (
            asdict(
                trace.validated_decision
            )
            if trace.validated_decision
            is not None
            else None
        ),
    )


async def _run_test_case(router: QuestionIntentRouter, test_case: RouterTestCase) -> None:
    """
    Chạy một test case và tiếp tục nếu case đó lỗi.
    """

    print()
    print("#" * 100)
    print(
        f"TEST: {test_case.name}"
    )
    print("#" * 100)

    _print_json(
        "INPUT",
        {
            "question": (
                test_case.question
            ),
            "mode": str(
                test_case.mode.value
                if hasattr(
                    test_case.mode,
                    "value",
                )
                else test_case.mode
            ),
            "explicit_query_key": (
                test_case
                .explicit_query_key
            ),
            "explicit_parameters": (
                test_case
                .explicit_parameters
                or {}
            ),
        },
    )

    started_at = time.perf_counter()

    try:
        decision = await router.decide(
            question=test_case.question,
            mode=test_case.mode,
            explicit_query_key=(
                test_case
                .explicit_query_key
            ),
            explicit_parameters=(
                test_case
                .explicit_parameters
            ),
        )

        elapsed = (
            time.perf_counter()
            - started_at
        )

        _print_trace(
            router.last_llama_trace
        )

        _print_json(
            "7. ROUTE DECISION CUỐI CÙNG",
            {
                **asdict(decision),
                "total_elapsed_seconds": (
                    elapsed
                ),
            },
        )

        print()
        print("PIPELINE SẼ THỰC HIỆN:")

        if decision.route == "documents":
            print(
                "- Dense retrieval + reranker."
            )
            print(
                "- Không chạy SQL."
            )

        elif decision.route == "sql":
            print(
                "- Không chạy document retrieval."
            )
            print(
                "- Chạy predefined SQL key: "
                f"{decision.query_key}"
            )
            print(
                "- Parameters: "
                + json.dumps(
                    decision.parameters,
                    ensure_ascii=False,
                )
            )

        elif decision.route == "hybrid":
            print(
                "- Chạy dense retrieval + reranker."
            )

            if decision.query_key is None:
                print(
                    "- Không chạy SQL vì không có "
                    "query_key đủ an toàn."
                )
            else:
                print(
                    "- Chạy song song predefined SQL: "
                    f"{decision.query_key}"
                )
                print(
                    "- Parameters: "
                    + json.dumps(
                        decision.parameters,
                        ensure_ascii=False,
                    )
                )

    except Exception as exception:
        elapsed = (
            time.perf_counter()
            - started_at
        )

        _print_trace(
            router.last_llama_trace
        )

        _print_json(
            "TEST CASE BỊ LỖI",
            {
                "exception_type": (
                    type(exception).__name__
                ),
                "message": str(exception),
                "elapsed_seconds": elapsed,
            },
        )


async def main() -> None:
    """
    MAIN TEST THẬT - KHÔNG DÙNG ARGPARSE.

    Bạn sửa trực tiếp danh sách TEST_CASES bên dưới.

    Khuyến nghị:
    - Chạy 3-5 case mỗi lần.
    - Mỗi case AUTO/SQL/HYBRID có thể gọi Llama.
    - Đặt enabled=False cho case chưa cần chạy.
    """

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    # Giảm log mạng không quan trọng.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = get_settings()

    router, resources = await _build_real_router(settings)

    try:
        # In catalog thật một lần trước khi test.
        safe_catalog = router._sanitize_catalog_for_llama(router.query_registry.list_for_router())

        _print_json(
            "CATALOG THẬT - LLAMA ĐƯỢC PHÉP CHỌN",
            safe_catalog,
        )

        # ====================================================
        # DỮ LIỆU TEST VIẾT TRỰC TIẾP
        # ====================================================

        test_cases: list[RouterTestCase] = [
            # 1. Câu hỏi quy trình:
            # Llama nên chọn documents.
            RouterTestCase(
                name=("AUTO - câu hỏi quy trình tài liệu"),
                question=("Quy trình gọi robot nhận hàng được thực hiện như thế nào?"),
                mode=QuestionMode.AUTO,
            ),

            # 2. Câu hỏi dữ liệu hiện tại:
            # Llama nên tìm query liên quan trạng thái robot
            # và trích robot_id="AGV-01" nếu catalog có query phù hợp.
            RouterTestCase(
                name=("AUTO - trạng thái robot hiện tại"),
                question=("Robot AGV-01 hiện đang ở trạng thái nào?"),
                mode=QuestionMode.AUTO,
            ),

            # 3. Câu hỏi vị trí pallet:
            # Llama nên tìm query vị trí pallet và trích A1-2.
            RouterTestCase(
                name=("AUTO - vị trí pallet"),
                question=("Pallet A1-2 hiện đang nằm ở vị trí nào?"),
                mode=QuestionMode.AUTO,
            ),

            # 4. Câu hỏi QR:
            # Llama nên trích nguyên mã QR.
            RouterTestCase(
                name=("AUTO - tra cứu cuộn vải theo QR"),
                question=("Cuộn vải có mã QR F260406000151 hiện đang ở đâu?"),
                mode=QuestionMode.AUTO,
            ),

            # 5. Câu hỏi cần cả dữ liệu hiện tại và quy trình:
            # Llama nên chọn hybrid nếu catalog có query trạng thái/lỗi.
            RouterTestCase(
                name=("AUTO - câu hỏi hybrid"),
                question=("Robot AGV-01 hiện đang lỗi gì và cần kiểm tra theo quy trình nào?"),
                mode=QuestionMode.AUTO,
            ),

            # 6. Câu hỏi mơ hồ:
            # "AGV thứ nhất" không phải mã rõ ràng.
            # Llama không nên tự bịa AGV-01.
            RouterTestCase(
                name=("AUTO - kiểm tra chống suy đoán parameter"),
                question=("AGV thứ nhất đang ở đâu?"),
                mode=QuestionMode.AUTO,
            ),

            # 7. SQL mode không explicit key:
            # Llama bắt buộc chọn một key SQL.
            # Nếu không chọn được, test sẽ báo lỗi.
            RouterTestCase(
                name=("SQL - Llama tự chọn query và parameter"),
                question=("Robot AGV-01 hiện đang ở trạng thái nào?"),
                mode=QuestionMode.SQL,
            ),

            # 8. HYBRID mode không explicit key:
            # Documents luôn chạy;
            # Llama quyết định có thêm SQL không.
            RouterTestCase(
                name=("HYBRID - Llama quyết định SQL bổ sung"),
                question=("Robot AGV-01 hiện đang lỗi gì và cách xử lý theo tài liệu là gì?"),
                mode=QuestionMode.HYBRID,
            ),
        ]

        # Test explicit key bằng query đầu tiên trong registry.
        #
        # Mục tiêu của case này không phải kiểm tra parameter nghiệp vụ,
        # mà kiểm tra nhánh:
        # caller khóa key -> không gọi Llama.
        if safe_catalog:
            first_query_key = str(safe_catalog[0]["query_key"])

            test_cases.append(
                RouterTestCase(
                    name=("SQL explicit - caller khóa query key"),
                    question=(f"Kiểm tra predefined query {first_query_key}"),
                    mode=QuestionMode.SQL,
                    explicit_query_key=(first_query_key),
                    explicit_parameters={},
                )
            )

        # Chạy lần lượt để log không bị trộn.
        for test_case in test_cases:
            if not test_case.enabled:
                continue

            await _run_test_case(router, test_case)

    finally:
        # Đóng resource theo thứ tự ngược.
        for resource in reversed(resources):
            try:
                await _close_if_supported(resource)
            except Exception:
                logger.exception(
                    "Không đóng được resource %s.",
                    type(resource).__name__,
                )


if __name__ == "__main__":
    try:
        # Chỉ một event loop cho mọi AsyncClient.
        asyncio.run(
            main()
        )

    except KeyboardInterrupt:
        print(
            "Đã dừng bởi người dùng.",
            file=sys.stderr,
        )
        raise SystemExit(130)

    except Exception as exception:
        logger.exception(
            "Intent router test thất bại."
        )

        print(
            f"\nLỖI: {exception}",
            file=sys.stderr,
        )

        raise SystemExit(1)