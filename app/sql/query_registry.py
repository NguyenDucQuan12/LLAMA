from __future__ import annotations

"""
Registry các SQL query được cho phép.

Llama không được phép sinh SQL tự do. Model chỉ có thể chọn một `query_key`
đã tồn tại trong registry này và cung cấp giá trị cho các bind parameter.

Các tên view bên dưới là ví dụ. Bạn phải ánh xạ chúng tới view hoặc bảng
thật trong WMS của mình trước khi bật SQL_SERVER_ENABLED=true.
"""

import datetime as datetime_module
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SqlParameterType(str, Enum):
    """
    Các kiểu parameter mà registry hỗ trợ.
    """

    STRING = "string"
    INTEGER = "integer"
    DATE = "date"
    DATETIME = "datetime"


@dataclass(frozen=True)
class SqlParameterDefinition:
    """
    Mô tả và quy tắc kiểm tra một bind parameter.  
    Ví dụ:  
    ```python
    SqlQueryDefinition(
        key="pallet_by_location",
        description="Tra cứu pallet ...",
        sql_text=\"""
            SELECT TOP (50) ...
            WHERE location_code = :location_code
        \""".strip(),
        parameters=(
            SqlParameterDefinition(
                name="location_code",
                parameter_type=SqlParameterType.STRING,
                description="Mã vị trí kho, ví dụ A1-2 hoặc F3-29.",
                required=True,
                maximum_length=50,
                pattern=r"[A-Za-z0-9_-]+",
            ),
        ),
    )
    ```
    Thì parameter tuân theo như trên
    """

    name: str
    parameter_type: SqlParameterType
    description: str
    required: bool = True
    maximum_length: int | None = None
    pattern: str | None = None

    def validate(self, raw_value: Any) -> Any:
        """
        Chuyển và kiểm tra giá trị parameter.
        """

        if raw_value is None:
            if self.required:
                raise ValueError(
                    f"Thiếu parameter bắt buộc: {self.name}."
                )

            return None

        if self.parameter_type == SqlParameterType.STRING:
            normalized_value = str(raw_value).strip()

            if not normalized_value:
                if self.required:
                    raise ValueError(
                        f"Parameter {self.name} không được rỗng."
                    )

                return None

            if (
                self.maximum_length is not None
                and len(normalized_value) > self.maximum_length
            ):
                raise ValueError(
                    f"Parameter {self.name} vượt quá "
                    f"{self.maximum_length} ký tự."
                )

            if self.pattern is not None:
                if re.fullmatch(self.pattern, normalized_value) is None:
                    raise ValueError(
                        f"Parameter {self.name} không đúng định dạng."
                    )

            return normalized_value

        if self.parameter_type == SqlParameterType.INTEGER:
            try:
                integer_value = int(raw_value)
            except (TypeError, ValueError) as exception:
                raise ValueError(
                    f"Parameter {self.name} phải là số nguyên."
                ) from exception

            return integer_value

        if self.parameter_type == SqlParameterType.DATE:
            if isinstance(raw_value, datetime_module.date):
                return raw_value

            try:
                return datetime_module.date.fromisoformat(
                    str(raw_value).strip()
                )
            except ValueError as exception:
                raise ValueError(
                    f"Parameter {self.name} phải có dạng YYYY-MM-DD."
                ) from exception

        if self.parameter_type == SqlParameterType.DATETIME:
            if isinstance(raw_value, datetime_module.datetime):
                return raw_value

            try:
                return datetime_module.datetime.fromisoformat(
                    str(raw_value).strip()
                )
            except ValueError as exception:
                raise ValueError(
                    f"Parameter {self.name} phải là ISO datetime."
                ) from exception

        raise ValueError(
            f"Kiểu parameter chưa được hỗ trợ: {self.parameter_type}."
        )


@dataclass(frozen=True)
class SqlQueryDefinition:
    """
    Một SQL query cố định được phép thực thi.
    """

    key: str
    description: str
    sql_text: str
    parameters: tuple[SqlParameterDefinition, ...] = field(default_factory=tuple)
    keyword_hints: tuple[str, ...] = field(default_factory=tuple)
    maximum_rows: int = 100

    def validate_parameters(self, provided_parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Chỉ giữ và kiểm tra các tham số được câu truy vấn định nghĩa.

        Các tham số lạ lạ bị từ chối để tránh người gọi nghĩ rằng nó có ảnh hưởng
        tới SQL trong khi thực tế không được bind.
        """
        # Lấy danh sách tên các tham số mà sql yêu cầu
        allowed_parameter_names = {parameter.name for parameter in self.parameters}
        # Lấy danh sách các tham số mà không được đăng ký
        unexpected_parameter_names = set(provided_parameters) - allowed_parameter_names

        if unexpected_parameter_names:
            raise ValueError(
                "Câu truy vấn nhận được các tham số không được đăng ký: "
                f"{sorted(unexpected_parameter_names)}."
            )

        validated_parameters: dict[str, Any] = {}
        # Duyệt qua danh sách các tham số trong câu truy vấn
        for parameter_definition in self.parameters:
            # Lấy các giá trị tham số
            raw_value = provided_parameters.get(parameter_definition.name)
            validated_value = parameter_definition.validate(raw_value)

            # Nếu có giá trị thì thêm nó vào danh sách tham số
            if validated_value is not None:
                validated_parameters[parameter_definition.name] = validated_value

        return validated_parameters

    def missing_required_parameters(self, provided_parameters: dict[str, Any]) -> list[str]:
        """
        Liệt kê danh sách tham số mà câu truy vấn bắt buộc còn thiếu
        """
        # Tạo list chứa các tham số còn thiếu để thông báo tới người gọi
        missing_parameters: list[str] = []
        # Duyệt qua danh sách tham số mà câu truy vấn này chấp nhận
        for parameter_definition in self.parameters:
            # Nếu không phải trường bắt buộc (required = True) thì có thể bỏ qua
            if not parameter_definition.required:
                continue
            
            # Lấy giá trị cảu tham số này
            raw_value = provided_parameters.get(parameter_definition.name)

            # Nếu tham số này là bắt buộc, nhưng nó không có giá trị thì thêm nó vào danh sách thiếu
            if raw_value is None or not str(raw_value).strip():
                missing_parameters.append(parameter_definition.name)

        return missing_parameters


class PredefinedSqlQueryRegistry:
    """
    Kho query duy nhất được SQL service sử dụng.
    """

    forbidden_sql_keywords: tuple[str, ...] = (
        "INSERT",
        "UPDATE",
        "DELETE",
        "MERGE",
        "DROP",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "EXEC",
        "EXECUTE",
        "GRANT",
        "REVOKE",
        "DENY",
    )

    def __init__(self) -> None:
        self.query_definitions = self._build_default_queries()

        for query_definition in self.query_definitions.values():
            self._validate_query_definition(query_definition)

    def get(self, query_key: str) -> SqlQueryDefinition:
        """
        Lấy mẫu truy vấn theo key (là từ khoá của mẫu truy vấn) hoặc báo lỗi rõ ràng.  
        Ví dụ:  
        ```python
        {
            "agv_tasks_today": SqlQueryDefinition
        }
        """
        # Chuẩn hoá key
        normalized_key = query_key.strip()
        query_definition = self.query_definitions.get(normalized_key)

        if query_definition is None:
            raise KeyError(f"Không tồn tại mẫu truy vấn có tên: {normalized_key}.")

        return query_definition

    def list_for_router(self) -> list[dict[str, Any]]:
        """
        Trả catalog không chứa SQL text để Llama chọn query an toàn.
        """

        catalog: list[dict[str, Any]] = []

        for query_definition in self.query_definitions.values():
            parameter_catalog: list[dict[str, Any]] = []

            for parameter in query_definition.parameters:
                parameter_catalog.append(
                    {
                        "name": parameter.name,
                        "type": parameter.parameter_type.value,
                        "description": parameter.description,
                        "required": parameter.required,
                    }
                )

            catalog.append(
                {
                    "key": query_definition.key,
                    "description": query_definition.description,
                    "parameters": parameter_catalog,
                    "keyword_hints": list(query_definition.keyword_hints),
                }
            )

        return catalog

    def keys(self) -> list[str]:
        """
        Liệt kê query key để API hoặc CLI hiển thị.
        """
        return sorted(self.query_definitions)

    def _build_default_queries(self) -> dict[str, SqlQueryDefinition]:
        """
        Khai báo các mẫu truy vấn cho hệ thống WMS/AGV.  
        Nên tạo một view riêng cho RAG, tách biệt với nghiệp vụ sản xuất
        """

        query_definitions = [
            SqlQueryDefinition(
                key="agv_tasks_today",
                description=(
                    "Lấy tối đa 100 nhiệm vụ AGV được tạo trong ngày hôm nay, "
                    "gồm mã nhiệm vụ, robot, trạng thái, vị trí nguồn và đích."
                ),
                sql_text="""
                SELECT TOP (100)
                    task_id,
                    task_code,
                    robot_code,
                    task_status,
                    source_location,
                    target_location,
                    created_at,
                    completed_at
                FROM dbo.vw_rag_agv_tasks
                WHERE created_at >= CONVERT(date, GETDATE())
                AND created_at < DATEADD(day, 1, CONVERT(date, GETDATE()))
                ORDER BY created_at DESC
                """.strip(),
                keyword_hints=(
                    "hôm nay robot nhận lệnh gì",
                    "nhiệm vụ robot hôm nay",
                    "agv task today",
                ),
                maximum_rows=100,
            ),
            SqlQueryDefinition(
                key="pallet_by_location",
                description=(
                    "Tra cứu pallet hiện đang được ghi nhận tại một mã vị trí "
                    "cụ thể, ví dụ A1-2 hoặc F3-29."
                ),
                sql_text="""
                SELECT TOP (50)
                    location_code,
                    pallet_code,
                    material_code,
                    quantity,
                    stock_status,
                    updated_at
                FROM dbo.vw_rag_pallet_locations
                WHERE location_code = :location_code
                ORDER BY updated_at DESC
                """.strip(),
                parameters=(
                    SqlParameterDefinition(
                        name="location_code",
                        parameter_type=SqlParameterType.STRING,
                        description="Mã vị trí kho, ví dụ A1-2 hoặc F3-29.",
                        required=True,
                        maximum_length=50,
                        pattern=r"[A-Za-z0-9_-]+",
                    ),
                ),
                keyword_hints=(
                    "pallet tại vị trí",
                    "vị trí pallet",
                    "location code",
                ),
                maximum_rows=50,
            ),
            SqlQueryDefinition(
                key="fabric_roll_by_qr",
                description=(
                    "Tra cứu cuộn vải theo mã QR, gồm mã vật liệu, pallet, "
                    "vị trí và trạng thái tồn kho."
                ),
                sql_text="""
                SELECT TOP (50)
                    qr_code,
                    material_code,
                    fabric_name,
                    pallet_code,
                    location_code,
                    stock_status,
                    updated_at
                FROM dbo.vw_rag_fabric_rolls
                WHERE qr_code = :qr_code
                ORDER BY updated_at DESC
                """.strip(),
                parameters=(
                    SqlParameterDefinition(
                        name="qr_code",
                        parameter_type=SqlParameterType.STRING,
                        description=(
                            "Mã QR của cuộn vải, ví dụ F260406000151."
                        ),
                        required=True,
                        maximum_length=100,
                        pattern=r"[A-Za-z0-9_-]+",
                    ),
                ),
                keyword_hints=(
                    "tra cứu cuộn vải",
                    "mã qr cuộn vải",
                    "fabric qr",
                ),
                maximum_rows=50,
            ),
            SqlQueryDefinition(
                key="agv_task_by_code",
                description=(
                    "Tra cứu chi tiết và trạng thái của một nhiệm vụ AGV theo "
                    "mã nhiệm vụ cụ thể."
                ),
                sql_text="""
                SELECT TOP (20)
                    task_id,
                    task_code,
                    robot_code,
                    task_status,
                    source_location,
                    target_location,
                    error_code,
                    error_message,
                    created_at,
                    completed_at
                FROM dbo.vw_rag_agv_tasks
                WHERE task_code = :task_code
                ORDER BY created_at DESC
                """.strip(),
                parameters=(
                    SqlParameterDefinition(
                        name="task_code",
                        parameter_type=SqlParameterType.STRING,
                        description="Mã nhiệm vụ AGV cần tra cứu.",
                        required=True,
                        maximum_length=100,
                        pattern=r"[A-Za-z0-9_.-]+",
                    ),
                ),
                keyword_hints=(
                    "trạng thái nhiệm vụ",
                    "task code",
                    "mã nhiệm vụ agv",
                ),
                maximum_rows=20,
            ),
            SqlQueryDefinition(
                key="recent_agv_errors",
                description=(
                    "Lấy các lỗi AGV gần nhất trong 24 giờ để kiểm tra lỗi "
                    "robot, vị trí và thông báo lỗi."
                ),
                sql_text="""
                SELECT TOP (100)
                    task_code,
                    robot_code,
                    error_code,
                    error_message,
                    source_location,
                    target_location,
                    created_at
                FROM dbo.vw_rag_agv_tasks
                WHERE error_code IS NOT NULL
                AND created_at >= DATEADD(hour, -24, GETDATE())
                ORDER BY created_at DESC
                """.strip(),
                keyword_hints=(
                    "lỗi robot gần đây",
                    "agv error",
                    "robot đang lỗi",
                ),
                maximum_rows=100,
            ),
        ]

        # Tạo và trả về một dict từ query_definitions bằng cách dùng query_definition.key làm key và query_definition làm value.
        return {
            query_definition.key: query_definition
            for query_definition in query_definitions
        }

    def _validate_query_definition(
        self,
        query_definition: SqlQueryDefinition,
    ) -> None:
        """
        Chặn query không phải read-only ngay khi ứng dụng khởi động.

        Đây là lớp bảo vệ bổ sung. Lớp bảo vệ quan trọng nhất vẫn là dùng
        tài khoản SQL Server chỉ có quyền SELECT trên các view dành cho RAG.
        """

        normalized_sql = re.sub(
            r"\s+",
            " ",
            query_definition.sql_text,
        ).strip()

        uppercase_sql = normalized_sql.upper()

        if not (
            uppercase_sql.startswith("SELECT ")
            or uppercase_sql.startswith("WITH ")
        ):
            raise ValueError(
                f"Query {query_definition.key} không bắt đầu bằng SELECT/WITH."
            )

        # Không cho phép nhiều statement phân tách bằng dấu chấm phẩy.
        if ";" in normalized_sql:
            raise ValueError(
                f"Query {query_definition.key} không được chứa dấu ';'."
            )

        for forbidden_keyword in self.forbidden_sql_keywords:
            keyword_pattern = rf"\b{re.escape(forbidden_keyword)}\b"

            if re.search(keyword_pattern, uppercase_sql):
                raise ValueError(
                    f"Query {query_definition.key} chứa từ khóa bị cấm: "
                    f"{forbidden_keyword}."
                )
