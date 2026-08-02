from __future__ import annotations

"""
========================

Router an toàn cho ba hướng xử lý:

    documents
        Chỉ tìm tài liệu.

    sql
        Chỉ chạy một predefined SELECT đã đăng ký trong registry.

    hybrid
        Chạy document retrieval và predefined SQL song song.

Nguyên tắc:
- Mode do caller chỉ định được ưu tiên cao nhất.
- Llama chỉ được xem catalog metadata, không được xem SQL text.
- Llama chỉ đề xuất route/query_key/parameters.
- Python kiểm tra lại route, allowlist, confidence và parameter.
- SafeSqlServerService vẫn kiểm tra parameter lần cuối trước khi bind.
- Khi AUTO không chắc chắn, fallback sang documents.

Ví dụ:

    python3 -m app.sql.intent_router \
        --question "Quy trình gọi robot như thế nào?" \
        --mode auto

    python3 -m app.sql.intent_router \
        --question "Robot AGV-01 hiện ở trạng thái nào?" \
        --mode auto

    python3 -m app.sql.intent_router \
        --question "AGV-01 đang lỗi; cần kiểm tra theo quy trình nào?" \
        --mode auto

    python3 -m app.sql.intent_router \
        --question "Kiểm tra AGV-01" \
        --mode sql \
        --sql-query-key "get_robot_status" \
        --sql-parameters '{"robot_id":"AGV-01"}'
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
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

from clients.ollama_client import OllamaClient
from config import Settings, get_settings
from schemas import QuestionMode
from sql.query_registry import PredefinedSqlQueryRegistry
from sql.sql_service import SafeSqlServerService


logger = logging.getLogger(__name__)

_ALLOWED_ROUTES = {"documents", "sql", "hybrid"}


@dataclass(frozen=True)
class RouteDecision:
    """
    Kết quả cuối cùng sau khi Python đã xác minh output Llama.

    route:
        documents, sql hoặc hybrid.

    query_key:
        None nếu không chạy SQL.
        Một key trong PredefinedSqlQueryRegistry nếu chạy SQL.

    parameters:
        Parameter được caller hoặc Llama trích xuất.
        SQL service vẫn phải validate trước khi bind.

    reason:
        Lý do phục vụ debug.

    confidence:
        Độ tin cậy 0-1.
        Quyết định explicit của caller có confidence=1.
    """

    route: str
    query_key: str | None
    parameters: dict[str, Any]
    reason: str
    confidence: float = 1.0


# Mã cũ chỉ cho phép ["documents", "sql"], vì vậy AUTO không thể sinh hybrid.
# Schema mới cho phép cả ba route.
ROUTER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "route": {
            "type": "string",
            "enum": ["documents", "sql", "hybrid"],
        },
        "query_key": {
            "type": ["string", "null"],
        },
        "parameters": {
            "type": "object",
            "additionalProperties": True,
        },
        "reason": {
            "type": "string",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "required": [
        "route",
        "query_key",
        "parameters",
        "reason",
        "confidence",
    ],
    "additionalProperties": False,
}


class QuestionIntentRouter:
    """
    Router quyết định documents, sql hoặc hybrid.

    Llama không có quyền quyết định cuối cùng.
    Tất cả output đều được validate bằng Python.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        query_registry: PredefinedSqlQueryRegistry,
        sql_server_service: SafeSqlServerService,
        settings: Settings | None = None,
    ) -> None:
        # Client gọi structured output.
        self.ollama_client = ollama_client

        # Registry là allowlist duy nhất của SQL query.
        self.query_registry = query_registry

        # Router chỉ dùng service này để kiểm tra SQL đã bật hay chưa.
        self.sql_server_service = sql_server_service

        # Settings là optional để tương thích constructor cũ.
        self.settings = settings

        self._validate_configuration()

    async def decide(
        self,
        question: str,
        mode: QuestionMode,
        explicit_query_key: str | None = None,
        explicit_parameters: Mapping[str, Any] | None = None,
    ) -> RouteDecision:
        """
        Chọn route theo thứ tự ưu tiên:

        1. Caller mode.
        2. Explicit predefined query key.
        3. Llama classifier.
        4. Python validation/fallback.
        """

        normalized_question = self._normalize_question(question)
        normalized_mode = self._normalize_mode(mode)
        normalized_query_key = self._normalize_optional_query_key(
            explicit_query_key
        )
        normalized_parameters = self._normalize_parameters(
            explicit_parameters
        )

        sql_enabled = bool(self.sql_server_service.is_enabled())

        # ----------------------------------------------------
        # MODE DOCUMENTS
        # ----------------------------------------------------
        if normalized_mode == "documents":
            # Input mâu thuẫn phải được phát hiện thay vì âm thầm bỏ qua.
            if normalized_query_key is not None:
                raise ValueError(
                    "Mode DOCUMENTS không được kèm sql_query_key."
                )

            if normalized_parameters:
                raise ValueError(
                    "Mode DOCUMENTS không được kèm sql_parameters."
                )

            return RouteDecision(
                route="documents",
                query_key=None,
                parameters={},
                reason="Caller yêu cầu chỉ tìm tài liệu.",
                confidence=1.0,
            )

        # ----------------------------------------------------
        # MODE SQL
        # ----------------------------------------------------
        if normalized_mode == "sql":
            if not sql_enabled:
                raise RuntimeError(
                    "Mode SQL được yêu cầu nhưng SQL Server chưa bật."
                )

            # Có key explicit: không cần Llama chọn key.
            if normalized_query_key is not None:
                self._require_registered_query(normalized_query_key)

                return RouteDecision(
                    route="sql",
                    query_key=normalized_query_key,
                    parameters=normalized_parameters,
                    reason=(
                        "Caller ép mode SQL và chỉ định predefined query."
                    ),
                    confidence=1.0,
                )

            # Không có key: Llama chỉ được phép chọn SQL.
            automatic = await self._decide_with_llama(
                question=normalized_question,
                allowed_routes={"sql"},
                forced_query_key=None,
            )

            if automatic.query_key is None:
                raise ValueError(
                    "Mode SQL nhưng router không chọn được query_key hợp lệ. "
                    "Hãy cung cấp sql_query_key."
                )

            return RouteDecision(
                route="sql",
                query_key=automatic.query_key,
                # Parameter caller có ưu tiên cao hơn parameter Llama.
                parameters={
                    **automatic.parameters,
                    **normalized_parameters,
                },
                reason=automatic.reason,
                confidence=automatic.confidence,
            )

        # ----------------------------------------------------
        # MODE HYBRID
        # ----------------------------------------------------
        if normalized_mode == "hybrid":
            # HYBRID luôn có document branch.
            #
            # SQL chưa bật:
            # - có explicit key => lỗi vì caller yêu cầu SQL.
            # - không có key => hybrid docs-only.
            if not sql_enabled:
                if normalized_query_key is not None:
                    raise RuntimeError(
                        "HYBRID có sql_query_key nhưng SQL Server chưa bật."
                    )

                if normalized_parameters:
                    raise ValueError(
                        "HYBRID có sql_parameters nhưng SQL Server chưa bật "
                        "và không có sql_query_key để sử dụng."
                    )

                return RouteDecision(
                    route="hybrid",
                    query_key=None,
                    parameters={},
                    reason=(
                        "SQL Server chưa bật; HYBRID chỉ chạy tài liệu."
                    ),
                    confidence=1.0,
                )

            # Caller chỉ định key => documents + key đó.
            if normalized_query_key is not None:
                self._require_registered_query(normalized_query_key)

                return RouteDecision(
                    route="hybrid",
                    query_key=normalized_query_key,
                    parameters=normalized_parameters,
                    reason=(
                        "Caller yêu cầu HYBRID và chỉ định SQL query."
                    ),
                    confidence=1.0,
                )

            # Không có key:
            # Llama quyết định có cần thêm SQL hay không.
            automatic = await self._decide_with_llama(
                question=normalized_question,
                allowed_routes={"documents", "sql", "hybrid"},
                forced_query_key=None,
            )

            if (
                automatic.route in {"sql", "hybrid"}
                and automatic.query_key is not None
            ):
                # Route bên ngoài vẫn là hybrid vì caller đã yêu cầu hybrid.
                return RouteDecision(
                    route="hybrid",
                    query_key=automatic.query_key,
                    # Parameter caller có ưu tiên cao hơn parameter Llama.
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

            # Không chọn được SQL đủ an toàn => chỉ tài liệu.
            return RouteDecision(
                route="hybrid",
                query_key=None,
                parameters={},
                reason=(
                    "Caller yêu cầu HYBRID nhưng router không chọn được "
                    "SQL query đủ tin cậy; chỉ chạy tài liệu. "
                    + automatic.reason
                ),
                confidence=automatic.confidence,
            )

        # ----------------------------------------------------
        # MODE AUTO
        # ----------------------------------------------------

        # AUTO có explicit query key:
        # key bị khóa; Llama chỉ chọn SQL hoặc HYBRID.
        if normalized_query_key is not None:
            if not sql_enabled:
                raise RuntimeError(
                    "AUTO nhận sql_query_key nhưng SQL Server chưa bật."
                )

            self._require_registered_query(normalized_query_key)

            automatic = await self._decide_with_llama(
                question=normalized_question,
                allowed_routes={"sql", "hybrid"},
                forced_query_key=normalized_query_key,
            )

            return RouteDecision(
                route=(
                    "hybrid"
                    if automatic.route == "hybrid"
                    else "sql"
                ),
                query_key=normalized_query_key,
                # Parameter explicit của caller ghi đè parameter Llama.
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

        # SQL chưa bật thì AUTO chắc chắn dùng documents.
        if not sql_enabled:
            if normalized_parameters:
                raise ValueError(
                    "AUTO có sql_parameters nhưng SQL Server chưa bật "
                    "và không có sql_query_key."
                )

            return RouteDecision(
                route="documents",
                query_key=None,
                parameters={},
                reason="SQL Server chưa bật; AUTO dùng tài liệu.",
                confidence=1.0,
            )

        automatic = await self._decide_with_llama(
            question=normalized_question,
            allowed_routes={"documents", "sql", "hybrid"},
            forced_query_key=None,
        )

        # Với AUTO, chỉ cho SQL/HYBRID khi confidence đủ cao.
        if automatic.route in {"sql", "hybrid"}:
            minimum_confidence = self._minimum_sql_confidence()

            if automatic.confidence < minimum_confidence:
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

        if (
            automatic.route in {"sql", "hybrid"}
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

    async def _decide_with_llama(
        self,
        question: str,
        allowed_routes: set[str],
        forced_query_key: str | None,
    ) -> RouteDecision:
        """
        Gọi Llama để đề xuất route/query_key.

        Hàm này không tin output Llama; kết quả phải qua
        _validate_llama_decision().
        """

        if (
            not allowed_routes
            or not allowed_routes.issubset(_ALLOWED_ROUTES)
        ):
            raise ValueError("allowed_routes không hợp lệ.")

        raw_catalog = self.query_registry.list_for_router()
        safe_catalog = self._sanitize_catalog_for_llama(raw_catalog)

        if not safe_catalog:
            if "documents" in allowed_routes:
                return RouteDecision(
                    route="documents",
                    query_key=None,
                    parameters={},
                    reason="SQL query catalog rỗng; dùng tài liệu.",
                    confidence=1.0,
                )

            raise RuntimeError(
                "Cần chọn SQL nhưng query catalog đang rỗng."
            )

        if forced_query_key is not None:
            self._require_registered_query(forced_query_key)

        allowed_routes_text = ", ".join(sorted(allowed_routes))

        forced_key_rule = (
            (
                f"Caller đã khóa query_key là `{forced_query_key}`. "
                "Không được thay đổi key này."
            )
            if forced_query_key is not None
            else (
                "Nếu chọn sql/hybrid, query_key phải có trong catalog."
            )
        )

        system_message = f"""
Bạn là bộ định tuyến an toàn cho hệ thống RAG và SQL Server.

ROUTE ĐƯỢC PHÉP:
{allowed_routes_text}

Ý NGHĨA:
- documents:
  Câu hỏi hỏi quy trình, hướng dẫn, nguyên nhân, cách thao tác,
  giải thích hoặc kiến thức ổn định trong tài liệu.

- sql:
  Câu hỏi chỉ cần dữ liệu hiện tại/cụ thể trong cơ sở dữ liệu,
  ví dụ trạng thái, vị trí pallet, nhiệm vụ đang chạy,
  dữ liệu theo QR, số lượng hoặc lỗi gần đây.

- hybrid:
  Câu hỏi vừa cần dữ liệu hiện tại từ SQL,
  vừa cần quy trình/hướng dẫn/giải thích từ tài liệu.
  Ví dụ: "AGV-01 đang lỗi; cần kiểm tra theo quy trình nào?"

QUY TẮC:
1. Chỉ chọn route được phép.
2. Không tạo hoặc trả SQL text.
3. Không tạo query_key mới.
4. {forced_key_rule}
5. Chỉ lấy parameter nếu giá trị xuất hiện rõ trong câu hỏi.
6. Không suy đoán mã robot, pallet, QR, thời gian hoặc trạng thái.
7. Khi không chắc chắn và documents được phép, chọn documents.
8. confidence nằm trong [0,1].
9. Trả đúng JSON schema.
""".strip()

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

        try:
            router_output = await self.ollama_client.chat_json(
                messages=[
                    {
                        "role": "system",
                        "content": system_message,
                    },
                    {
                        "role": "user",
                        "content": user_message,
                    },
                ],
                json_schema=ROUTER_JSON_SCHEMA,
                temperature=0.0,
            )
        except Exception:
            # AUTO/HYBRID có thể fallback tài liệu.
            if "documents" in allowed_routes:
                logger.exception(
                    "Llama router lỗi; fallback documents."
                )

                return RouteDecision(
                    route="documents",
                    query_key=None,
                    parameters={},
                    reason=(
                        "Llama router lỗi; fallback an toàn sang tài liệu."
                    ),
                    confidence=0.0,
                )

            # SQL explicit mode phải lộ lỗi để caller sửa.
            raise

        return self._validate_llama_decision(
            router_output=router_output,
            allowed_routes=allowed_routes,
            forced_query_key=forced_query_key,
        )

    def _validate_llama_decision(
        self,
        router_output: Any,
        allowed_routes: set[str],
        forced_query_key: str | None,
    ) -> RouteDecision:
        """
        Kiểm tra output Llama trước khi cho phép SQL chạy.
        """

        if not isinstance(router_output, Mapping):
            return self._safe_fallback(
                allowed_routes,
                "Router output không phải JSON object.",
            )

        route = str(
            router_output.get("route", "documents")
        ).strip().lower()

        if route not in allowed_routes:
            return self._safe_fallback(
                allowed_routes,
                f"Router trả route {route!r} ngoài allowed_routes.",
            )

        reason = str(
            router_output.get("reason", "")
        ).strip() or "Router không cung cấp lý do."

        confidence = self._normalize_confidence(
            router_output.get("confidence", 0.0)
        )

        parameters = self._normalize_parameters(
            router_output.get("parameters")
        )

        # documents không được giữ query key/parameters.
        if route == "documents":
            return RouteDecision(
                route="documents",
                query_key=None,
                parameters={},
                reason=reason,
                confidence=confidence,
            )

        raw_query_key = router_output.get("query_key")

        if not isinstance(raw_query_key, str):
            return self._safe_fallback(
                allowed_routes,
                "Router chọn SQL/HYBRID nhưng query_key không phải string.",
                confidence=confidence,
            )

        query_key = raw_query_key.strip()

        if not query_key:
            return self._safe_fallback(
                allowed_routes,
                "Router chọn SQL/HYBRID nhưng query_key rỗng.",
                confidence=confidence,
            )

        # Caller khóa key => model không được đổi.
        if (
            forced_query_key is not None
            and query_key != forced_query_key
        ):
            logger.warning(
                "Llama đổi forced key %s thành %s; giữ key caller.",
                forced_query_key,
                query_key,
            )
            query_key = forced_query_key

        try:
            self._require_registered_query(query_key)
        except KeyError:
            logger.warning(
                "Llama tạo query_key ngoài allowlist: %s",
                query_key,
            )

            return self._safe_fallback(
                allowed_routes,
                "Router trả query_key ngoài allowlist.",
                confidence=confidence,
            )

        return RouteDecision(
            route=route,
            query_key=query_key,
            parameters=parameters,
            reason=reason,
            confidence=confidence,
        )

    def _safe_fallback(
        self,
        allowed_routes: set[str],
        reason: str,
        confidence: float = 0.0,
    ) -> RouteDecision:
        """
        Nếu documents được phép thì fallback documents.
        Nếu chỉ SQL được phép thì báo lỗi để caller cung cấp key.
        """

        if "documents" in allowed_routes:
            return RouteDecision(
                route="documents",
                query_key=None,
                parameters={},
                reason=reason + " Fallback tài liệu.",
                confidence=confidence,
            )

        raise ValueError(
            reason
            + " Không thể fallback documents; "
            "hãy cung cấp sql_query_key."
        )

    def _sanitize_catalog_for_llama(
        self,
        raw_catalog: Any,
    ) -> list[dict[str, Any]]:
        """
        Chỉ gửi metadata an toàn cho Llama.

        Không gửi các field như:
        sql, sql_text, statement, query_text, connection_string.
        """

        if raw_catalog is None:
            return []

        if (
            not isinstance(raw_catalog, Sequence)
            or isinstance(raw_catalog, (str, bytes, bytearray))
        ):
            raise TypeError(
                "query_registry.list_for_router() phải trả Sequence."
            )

        output: list[dict[str, Any]] = []

        for index, item in enumerate(raw_catalog):
            if not isinstance(item, Mapping):
                raise TypeError(
                    f"Catalog item {index} không phải Mapping."
                )

            query_key = item.get("query_key") or item.get("key")

            if (
                not isinstance(query_key, str)
                or not query_key.strip()
            ):
                raise ValueError(
                    f"Catalog item {index} thiếu query_key."
                )

            normalized_key = query_key.strip()

            # Registry là source of truth.
            self._require_registered_query(normalized_key)

            safe_item: dict[str, Any] = {
                "query_key": normalized_key,
                "description": str(
                    item.get("description", "")
                ).strip(),
            }

            # Chỉ copy metadata parameter.
            for field_name in (
                "parameters",
                "required_parameters",
                "optional_parameters",
            ):
                if field_name in item:
                    safe_item[field_name] = self._make_json_safe(
                        item[field_name]
                    )

            output.append(safe_item)

        return output

    def _make_json_safe(
        self,
        value: Any,
        depth: int = 0,
    ) -> Any:
        """
        Chuyển catalog metadata thành kiểu JSON đơn giản.
        """

        if depth > 5:
            return str(value)

        if value is None or isinstance(
            value, (str, int, float, bool)
        ):
            return value

        if isinstance(value, Mapping):
            return {
                str(key): self._make_json_safe(child, depth + 1)
                for key, child in value.items()
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

    def _require_registered_query(
        self,
        query_key: str,
    ) -> Any:
        """
        query_key phải tồn tại trong allowlist registry.
        """

        try:
            return self.query_registry.get(query_key)
        except KeyError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Không kiểm tra được registry cho key {query_key!r}."
            ) from exc

    def _normalize_question(
        self,
        question: str,
    ) -> str:
        """
        Chuẩn hóa Unicode/space nhưng giữ nguyên mã kỹ thuật.
        """

        if not isinstance(question, str):
            raise TypeError("question phải là string.")

        normalized = unicodedata.normalize("NFC", question)
        normalized = " ".join(normalized.split())

        if not normalized:
            raise ValueError("question không được rỗng.")

        maximum = self._positive_setting(
            "router_max_question_characters",
            4000,
        )

        if len(normalized) > maximum:
            raise ValueError(
                f"question quá dài: {len(normalized)} > {maximum}."
            )

        return normalized

    def _normalize_mode(
        self,
        mode: QuestionMode,
    ) -> str:
        """
        Chuyển enum QuestionMode thành auto/documents/sql/hybrid.
        """

        if mode is None:
            raise TypeError("mode không được là None.")

        raw_value = mode.value if hasattr(mode, "value") else mode
        normalized = str(raw_value).strip().lower()

        # Xử lý chuỗi dạng "QuestionMode.AUTO".
        if "." in normalized:
            normalized = normalized.rsplit(".", 1)[-1]

        if normalized not in {
            "auto",
            "documents",
            "sql",
            "hybrid",
        }:
            raise ValueError(
                f"QuestionMode không hợp lệ: {normalized!r}."
            )

        return normalized

    def _normalize_optional_query_key(
        self,
        value: str | None,
    ) -> str | None:
        """
        Chuẩn hóa explicit query key.
        """

        if value is None:
            return None

        if not isinstance(value, str):
            raise TypeError(
                "explicit_query_key phải là string hoặc None."
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "explicit_query_key không được rỗng."
            )

        maximum = self._positive_setting(
            "router_max_query_key_characters",
            200,
        )

        if len(normalized) > maximum:
            raise ValueError(
                "explicit_query_key quá dài."
            )

        return normalized

    def _normalize_parameters(
        self,
        value: Any,
    ) -> dict[str, Any]:
        """
        Chỉ nhận object phẳng với scalar hoặc list scalar.

        SQL service vẫn là lớp validate nghiệp vụ cuối cùng.
        """

        if value is None:
            return {}

        if not isinstance(value, Mapping):
            raise TypeError(
                "parameters phải là Mapping hoặc None."
            )

        maximum_count = self._positive_setting(
            "router_max_parameter_count",
            20,
        )

        if len(value) > maximum_count:
            raise ValueError(
                f"Quá nhiều parameter: {len(value)} > {maximum_count}."
            )

        output: dict[str, Any] = {}

        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(
                    "Tên parameter phải là string."
                )

            key = raw_key.strip()

            if not key:
                raise ValueError(
                    "Tên parameter không được rỗng."
                )

            output[key] = self._normalize_parameter_value(
                raw_value,
                key,
            )

        return output

    def _normalize_parameter_value(
        self,
        value: Any,
        parameter_name: str,
    ) -> Any:
        """
        Hỗ trợ:
        - None
        - str
        - int
        - float hữu hạn
        - bool
        - list/tuple chứa các kiểu trên

        Không cho dict lồng nhau.
        """

        if value is None:
            return None

        if isinstance(value, str):
            normalized = value.strip()

            maximum = self._positive_setting(
                "router_max_parameter_string_characters",
                500,
            )

            if len(normalized) > maximum:
                raise ValueError(
                    f"Parameter {parameter_name!r} quá dài."
                )

            return normalized

        if isinstance(value, bool):
            return value

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(
                    f"Parameter {parameter_name!r} là NaN/Infinity."
                )
            return value

        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
        ):
            maximum_items = self._positive_setting(
                "router_max_parameter_list_items",
                100,
            )

            if len(value) > maximum_items:
                raise ValueError(
                    f"List parameter {parameter_name!r} quá dài."
                )

            return [
                self._normalize_parameter_value(child, parameter_name)
                for child in value
            ]

        raise TypeError(
            f"Parameter {parameter_name!r} có kiểu không hỗ trợ: "
            f"{type(value).__name__}."
        )

    def _normalize_confidence(
        self,
        value: Any,
    ) -> float:
        """
        Chuyển confidence thành float trong [0,1].
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
        Default 0.65 cho AUTO SQL/HYBRID.
        """

        raw_value = self._setting(
            "router_min_sql_confidence",
            0.65,
        )

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "router_min_sql_confidence phải là số."
            ) from exc

        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(
                "router_min_sql_confidence phải trong [0,1]."
            )

        return value

    def _positive_setting(
        self,
        name: str,
        default: int,
    ) -> int:
        """
        Đọc một Settings int > 0.
        """

        value = self._setting(name, default)

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"{name} phải là int > 0.")

        return value

    def _setting(
        self,
        name: str,
        default: Any,
    ) -> Any:
        """
        Tương thích Settings cũ: field chưa có thì dùng default.
        """

        if self.settings is None:
            return default

        return getattr(self.settings, name, default)

    def _validate_configuration(self) -> None:
        """
        Validate giới hạn khi khởi tạo.
        """

        self._positive_setting(
            "router_max_question_characters",
            4000,
        )
        self._positive_setting(
            "router_max_query_key_characters",
            200,
        )
        self._positive_setting(
            "router_max_parameter_count",
            20,
        )
        self._positive_setting(
            "router_max_parameter_string_characters",
            500,
        )
        self._positive_setting(
            "router_max_parameter_list_items",
            100,
        )
        self._minimum_sql_confidence()


# ============================================================
# DỰNG COMPONENT THẬT CHO __main__
# ============================================================

T = TypeVar("T")


def _construct_component(
    component_class: type[T],
    dependencies: Mapping[str, Any],
) -> T:
    """
    Dựng component theo tên tham số constructor.

    Nếu SafeSqlServerService cần engine/session_factory,
    lỗi sẽ nêu đúng dependency còn thiếu.
    """

    signature = inspect.signature(component_class)
    kwargs: dict[str, Any] = {}
    missing: list[str] = []

    for name, parameter in signature.parameters.items():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        if name in dependencies:
            kwargs[name] = dependencies[name]
        elif parameter.default is inspect.Parameter.empty:
            missing.append(name)

    if missing:
        raise RuntimeError(
            f"Không dựng được {component_class.__name__}; "
            f"thiếu dependency: {', '.join(missing)}. "
            "Hãy bổ sung trong _build_real_router()."
        )

    return component_class(**kwargs)


def _parse_question_mode(
    raw_mode: str,
) -> QuestionMode:
    """
    Parse enum name hoặc enum value.
    """

    normalized = raw_mode.strip().lower()

    for member in QuestionMode:
        if normalized in {
            member.name.lower(),
            str(member.value).strip().lower(),
        }:
            return member

    valid = sorted(
        {member.name.lower() for member in QuestionMode}
        | {
            str(member.value).strip().lower()
            for member in QuestionMode
        }
    )

    raise ValueError(
        f"Mode {raw_mode!r} không hợp lệ. Các mode: {valid}."
    )


def _parse_json_parameters(
    raw_json: str | None,
) -> dict[str, Any]:
    """
    Parse --sql-parameters.
    """

    if raw_json is None:
        return {}

    try:
        value = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"--sql-parameters không phải JSON hợp lệ: {exc}"
        ) from exc

    if not isinstance(value, dict):
        raise TypeError(
            "--sql-parameters phải là JSON object."
        )

    return value


async def _close_if_supported(
    resource: Any,
) -> None:
    """
    Đóng resource hỗ trợ close().
    """

    close_method = getattr(resource, "close", None)

    if not callable(close_method):
        return

    result = close_method()

    if inspect.isawaitable(result):
        await result


async def _build_real_router(
    settings: Settings,
) -> tuple[QuestionIntentRouter, list[Any]]:
    """
    Dựng router bằng Ollama, registry và SQL service thật.
    """

    ollama_client = OllamaClient(settings)

    dependencies: dict[str, Any] = {
        "settings": settings,
        "ollama_client": ollama_client,
    }

    query_registry = _construct_component(
        PredefinedSqlQueryRegistry,
        dependencies,
    )
    dependencies["query_registry"] = query_registry

    sql_server_service = _construct_component(
        SafeSqlServerService,
        dependencies,
    )
    dependencies["sql_server_service"] = sql_server_service

    router = QuestionIntentRouter(
        ollama_client=ollama_client,
        query_registry=query_registry,
        sql_server_service=sql_server_service,
        settings=settings,
    )

    return router, [
        sql_server_service,
        ollama_client,
    ]


async def test_real_router(
    *,
    question: str,
    mode_text: str,
    sql_query_key: str | None,
    sql_parameters_json: str | None,
    show_catalog: bool,
) -> None:
    """
    Test bằng registry, SQL enabled-state và Ollama thật.
    Không tạo fake data.
    """

    settings = get_settings()
    router, resources = await _build_real_router(settings)

    try:
        if show_catalog:
            safe_catalog = router._sanitize_catalog_for_llama(
                router.query_registry.list_for_router()
            )

            print()
            print("=" * 80)
            print("CATALOG AN TOÀN GỬI CHO LLAMA")
            print("=" * 80)
            print(
                json.dumps(
                    safe_catalog,
                    ensure_ascii=False,
                    indent=2,
                )
            )

        decision = await router.decide(
            question=question,
            mode=_parse_question_mode(mode_text),
            explicit_query_key=sql_query_key,
            explicit_parameters=_parse_json_parameters(
                sql_parameters_json
            ),
        )

        print()
        print("=" * 80)
        print("KẾT QUẢ INTENT ROUTER")
        print("=" * 80)
        print(
            json.dumps(
                asdict(decision),
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

        print()
        print("QUESTION ANSWERING SERVICE SẼ CHẠY:")

        if decision.route == "documents":
            print("- Dense retrieval + reranker.")
            print("- Không chạy SQL.")

        elif decision.route == "sql":
            print("- Không chạy document retrieval.")
            print(
                f"- Chạy predefined SQL: {decision.query_key}"
            )

        elif decision.route == "hybrid":
            print("- Dense retrieval + reranker.")

            if decision.query_key is None:
                print(
                    "- Không có SQL key đủ tin cậy; chỉ dùng tài liệu."
                )
            else:
                print(
                    f"- Chạy song song SQL: {decision.query_key}"
                )

    finally:
        for resource in resources:
            try:
                await _close_if_supported(resource)
            except Exception:
                logger.exception(
                    "Không đóng được resource %s.",
                    type(resource).__name__,
                )


def build_argument_parser() -> argparse.ArgumentParser:
    """
    CLI dùng dữ liệu/config thật.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Test QuestionIntentRouter với Ollama và SQL registry thật."
        )
    )

    parser.add_argument(
        "--question",
        required=True,
        help="Câu hỏi cần route.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "documents", "sql", "hybrid"],
        default="auto",
    )
    parser.add_argument(
        "--sql-query-key",
        default=None,
    )
    parser.add_argument(
        "--sql-parameters",
        default=None,
        help='Ví dụ: {"robot_id":"AGV-01"}',
    )
    parser.add_argument(
        "--show-catalog",
        action="store_true",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    return parser


async def main() -> None:
    """
    Main test thật, không tạo fake catalog/model.
    """

    args = build_argument_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    await test_real_router(
        question=args.question,
        mode_text=args.mode,
        sql_query_key=args.sql_query_key,
        sql_parameters_json=args.sql_parameters,
        show_catalog=args.show_catalog,
    )


if __name__ == "__main__":
    try:
        # Chỉ một event loop cho toàn bộ AsyncClient.
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Đã dừng bởi người dùng.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        logger.exception("Intent router test thất bại.")
        print(f"\nLỖI: {exc}", file=sys.stderr)
        raise SystemExit(1)