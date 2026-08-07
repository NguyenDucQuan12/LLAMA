from __future__ import annotations

"""
- Chỉ chạy SELECT hoặc stored procedure đã đăng ký trong registry.
- Tạo đúng một AsyncEngine cho mỗi process và tái sử dụng connection pool.
- Giới hạn số truy vấn đồng thời để không dồn tải vào SQL Server.
- Có timeout khi chờ hàng đợi, chờ pool và chạy command.
- Không tự retry procedure để tránh nguy cơ thực thi lặp.

Đăng ký SELECT trong registry, ví dụ:

    key="get_robot_status"
    sql_text='''
        SELECT TOP (1)
            RobotCode,
            RobotStatus,
            UpdatedAt
        FROM dbo.RobotStatus
        WHERE RobotCode = :robot_id
        ORDER BY UpdatedAt DESC
    '''
    required_parameters=["robot_id"]
    maximum_rows=1

Đăng ký stored procedure, ví dụ:

    key="get_robot_status_procedure"
    sql_text='''
        SET NOCOUNT ON;
        EXEC dbo.usp_get_robot_status
            @robot_id = :robot_id
    '''
    required_parameters=["robot_id"]
    maximum_rows=10

Llama/người dùng chỉ được chọn query key. Không được truyền SQL text hoặc tên
procedure trực tiếp vào service.
"""

import asyncio
import datetime as datetime_module
import decimal
import inspect
import json
import logging
import math
import sys
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import URL, text
from sqlalchemy.exc import DBAPIError, SQLAlchemyError, TimeoutError as SqlAlchemyPoolTimeoutError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

# Mở comment 3 dòng bên dưới mỗi khi test (Chạy trực tiếp hàm if __main__)
import os,sys
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from config import Settings, get_settings
from schemas import SqlExecutionResponse
from sql.query_registry import PredefinedSqlQueryRegistry


logger = logging.getLogger(__name__)


class SqlServerQueueTimeoutError(RuntimeError):
    """Không lấy được slot ứng dụng hoặc connection đúng hạn."""


class SqlServerCommandTimeoutError(RuntimeError):
    """Query hoặc procedure vượt command timeout."""


class SqlServerServiceClosedError(RuntimeError):
    """Service đã shutdown nhưng vẫn bị gọi."""


@dataclass(frozen=True)
class SqlServiceStatistics:
    """Thống kê nhẹ trong một process Uvicorn."""

    active_queries: int
    completed_queries: int
    failed_queries: int
    queue_timeouts: int
    command_timeouts: int
    pool_status: str


class SafeSqlServerService:
    """
    Chỉ chạy statement lấy từ PredefinedSqlQueryRegistry.

    Vòng đời đúng:

        FastAPI startup:
            service = SafeSqlServerService(...)
            await service.startup()

        Mỗi request:
            await service.execute_predefined_query(...)

        FastAPI shutdown:
            await service.close()

    Không tạo service hoặc engine mới cho từng câu hỏi.
    """

    def __init__(self, settings: Settings, query_registry: PredefinedSqlQueryRegistry) -> None:
        # Lưu cấu hình ứng dụng.
        self.settings = settings
        # Registry là allowlist duy nhất của query/procedure.
        self.query_registry = query_registry
        # Engine được tạo lazy và dùng chung trong suốt vòng đời process.
        self._engine: AsyncEngine | None = None
        # Nếu nhiều request đầu cùng đến, lock đảm bảo chỉ tạo một engine.
        self._engine_creation_lock = asyncio.Lock()
        # Bảo vệ close() khỏi bị gọi đồng thời nhiều lần.
        self._close_lock = asyncio.Lock()
        # Sau khi close(), service không được sử dụng lại.
        self._closed = False

        # pool_size là số connection giữ thường trực trong pool.
        self._pool_size = self._positive_int_setting("sql_server_pool_size", 5)

        # max_overflow là connection tạm thời vượt pool_size.
        # Khuyến nghị production bắt đầu bằng 0 để kiểm soát tổng connection.
        self._max_overflow = self._non_negative_int_setting("sql_server_max_overflow", 0)

        # Tổng connection tối đa mà một worker có thể mở.
        self._pool_capacity = self._pool_size + self._max_overflow

        # Giới hạn query đồng thời ở tầng ứng dụng.
        configured_concurrency = getattr(self.settings, "sql_server_max_concurrent_queries", self._pool_capacity)
        self._max_concurrent_queries = self._positive_int(configured_concurrency, "sql_server_max_concurrent_queries")

        # Nếu semaphore lớn hơn pool, request chỉ chuyển sang chờ ở pool.
        if self._max_concurrent_queries > self._pool_capacity:
            raise ValueError(
                "sql_server_max_concurrent_queries không được lớn hơn "
                "sql_server_pool_size + sql_server_max_overflow. "
                f"Actual: {self._max_concurrent_queries} > "
                f"{self._pool_capacity}."
            )

        # Semaphore ngăn quá nhiều coroutine cùng chạm SQL Server.
        self._query_semaphore = asyncio.Semaphore(self._max_concurrent_queries)

        # Thống kê process-local phục vụ debug/monitoring cơ bản.
        self._active_queries = 0
        self._completed_queries = 0
        self._failed_queries = 0
        self._queue_timeouts = 0
        self._command_timeouts = 0

    @property
    def engine(self) -> AsyncEngine | None:
        """Property chỉ đọc để tương thích code cũ."""

        return self._engine

    def is_enabled(self) -> bool:
        """
        SQL được xem là bật khi flag=True và connection string không rỗng.
        """

        enabled = bool(getattr(self.settings, "sql_server_enabled", False))
        connection_string = getattr(self.settings, "sql_server_odbc_connection_string", "")

        return bool(enabled and isinstance(connection_string, str) and connection_string.strip())

    async def startup(self) -> None:
        """
        Tạo engine và chạy SELECT 1 để phát hiện lỗi ngay lúc startup.
        """

        if not self.is_enabled():
            logger.info("SQL Server đang tắt; bỏ qua warm-up.")
            return

        await self._get_or_create_engine()
        await self.health_check()

    async def close(self) -> None:
        """
        Dispose pool khi FastAPI shutdown; không gọi sau mỗi query.
        """

        async with self._close_lock:
            if self._closed:
                return

            self._closed = True
            engine = self._engine
            self._engine = None

            if engine is not None:
                await engine.dispose()
                logger.info("Đã dispose SQL Server AsyncEngine.")

    async def health_check(self) -> bool:
        """Lấy connection từ pool và chạy SELECT 1."""

        if not self.is_enabled():
            return False
        # Tạo engine kết nối tới DB
        engine = await self._get_or_create_engine()

        try:
            async with engine.connect() as connection:
                async with asyncio.timeout(self._command_timeout_seconds()):
                    result = await connection.execute(
                        text("SELECT 1 AS health_value")
                    )
                    return result.scalar_one() == 1

        except TimeoutError as exception:
            raise SqlServerCommandTimeoutError("SQL Server health check bị timeout.") from exception
        except SQLAlchemyError as exception:
            raise RuntimeError("Không thể kết nối SQL Server.") from exception

    async def execute_predefined_query(self, query_key: str, parameters: Mapping[str, Any] | None) -> SqlExecutionResponse:
        """
        Chạy SELECT hoặc EXEC statement đã đăng ký.

        Các bước:
        1. Kiểm tra service và cấu hình.
        2. Chuẩn hóa query key/parameter.
        3. Lấy definition từ registry.
        4. Kiểm tra parameter bắt buộc.
        5. Validate parameter theo definition.
        6. Chờ semaphore slot.
        7. Checkout connection từ pool.
        8. Bind parameter và chạy statement.
        9. Fetch tối đa maximum_rows.
        10. Serialize kết quả thành JSON-safe.
        """
        # Kiểm tra service đang đóng hay mở
        self._raise_if_closed()
        # Kiểm tra SQL Server có được bật không
        if not self.is_enabled():
            raise RuntimeError("SQL Server chưa được bật hoặc chưa có connection string.")

        # Xác thực và tối ưu hoá key truy vấn và các tham số
        normalized_query_key = self._normalize_query_key(query_key)
        normalized_parameters = self._normalize_parameters(parameters)

        # Lấy đối tượng câu truy vấn đã được đăng ký, nếu không có câu truy vấn hợp lệ thì phải báo lỗi
        query_definition = self.query_registry.get(normalized_query_key)
        # Lấy chuỗi truy vấn từ cau truy vấn đã lấy
        registered_sql_text = self._get_registered_sql_text(query_definition)
        # maximum_rows được kiểm tra cả ở query và mức global.
        maximum_rows = self._get_maximum_rows(query_definition)

        # Kiểm tra các tham số đã đủ cho câu truy vấn chưa, nếu thiếu thì không được phép chạy
        missing_parameters = query_definition.missing_required_parameters(normalized_parameters)

        if missing_parameters:
            return SqlExecutionResponse(
                executed=False,
                query_key=query_definition.key,
                query_description=query_definition.description,
                parameters=normalized_parameters,
                rows=[],
                row_count=0,
                missing_parameters=list(missing_parameters),
            )

        # Definition kiểm tra tham số truyền vào lạ, kiểu và format nghiệp vụ.
        validated_parameters = query_definition.validate_parameters(normalized_parameters)
        if not isinstance(validated_parameters, Mapping):
            raise TypeError("validate_parameters() phải trả Mapping.")
        validated_parameters_dict = dict(validated_parameters)

        # text() sử dụng bind parameter, ví dụ :robot_id.
        statement = text(registered_sql_text)
        execution_kind = self._infer_execution_kind(registered_sql_text)

        # Lấy slot để cho phép thực hiện truy vấn dữ liệu
        async with self._query_slot(normalized_query_key):
            # Bắt đầu tính thời gian chạy
            started_at = time.perf_counter()

            try:
                # Thực hiện câu truy vấn nếu có thể dành được slot thưucj thi
                raw_rows, was_truncated = await self._execute_statement(
                    sql_statement=statement,
                    validated_parameters=validated_parameters_dict,
                    maximum_rows=maximum_rows,
                    query_key=normalized_query_key,
                )

                # Serialize sau khi connection đã được trả về pool.
                serialized_rows = self._serialize_rows(raw_rows)
                # Kết thúc tính toán thừoi gian chạy cho 1 lệnh truy vấn
                elapsed = time.perf_counter() - started_at
                self._completed_queries += 1

                # Log WARNING nếu query chậm hơn threshold.
                log_method = (
                    logger.warning
                    if elapsed >= self._slow_query_threshold_seconds()
                    else logger.info
                )
                log_method(
                    "Predefined %s %s hoàn thành: rows=%s, "
                    "truncated=%s, elapsed=%.3fs, pool=%s",
                    execution_kind,
                    query_definition.key,
                    len(serialized_rows),
                    was_truncated,
                    elapsed,
                    self.pool_status(),
                )

                return SqlExecutionResponse(
                    executed=True,
                    query_key=query_definition.key,
                    query_description=query_definition.description,
                    parameters={
                        key: self._serialize_sql_value(value)
                        for key, value in validated_parameters_dict.items()
                    },
                    rows=serialized_rows,
                    row_count=len(serialized_rows),
                    missing_parameters=[],
                )

            except Exception:
                self._failed_queries += 1
                raise

    async def execute_predefined_procedure(self, procedure_key: str, parameters: Mapping[str, Any] | None) -> SqlExecutionResponse:
        """
        Thực hiện chạy 1 procedure đã đăng ký trước
        """

        return await self.execute_predefined_query(query_key=procedure_key, parameters=parameters)

    def pool_status(self) -> str:
        """Trả chuỗi trạng thái pool để debug."""

        if self._engine is None:
            return "engine-not-created"

        try:
            return self._engine.sync_engine.pool.status()
        except Exception:
            return "pool-status-unavailable"

    def get_statistics(self) -> SqlServiceStatistics:
        """Snapshot thống kê trong worker hiện tại."""

        return SqlServiceStatistics(
            active_queries=self._active_queries,
            completed_queries=self._completed_queries,
            failed_queries=self._failed_queries,
            queue_timeouts=self._queue_timeouts,
            command_timeouts=self._command_timeouts,
            pool_status=self.pool_status(),
        )

    async def _get_or_create_engine(self) -> AsyncEngine:
        """
        Trả engine hiện tại hoặc tạo đúng một engine bằng asyncio.Lock.
        """

        self._raise_if_closed()

        # Fast path: engine đã tồn tại.
        if self._engine is not None:
            return self._engine

        # Slow path: chỉ một coroutine được tạo engine.
        async with self._engine_creation_lock:
            # Request khác có thể đã tạo engine trong lúc ta chờ lock.
            if self._engine is not None:
                return self._engine

            self._raise_if_closed()

            connection_string = getattr(self.settings, "sql_server_odbc_connection_string", "")
            if ( not isinstance(connection_string, str) or not connection_string.strip() ):
                raise RuntimeError(
                    "sql_server_odbc_connection_string đang rỗng."
                )

            # URL.create cho phép chuyển nguyên ODBC string qua odbc_connect.
            connection_url = URL.create(
                drivername="mssql+aioodbc",
                query={"odbc_connect": connection_string.strip()},
            )

            # AsyncEngine tự dùng AsyncAdaptedQueuePool.
            self._engine = create_async_engine(
                connection_url,

                # Kiểm tra connection còn sống mỗi lần checkout.
                pool_pre_ping=True,

                # Số connection giữ trong pool của worker này.
                pool_size=self._pool_size,

                # Số connection tạm vượt pool_size.
                max_overflow=self._max_overflow,

                # Chờ connection trong pool tối đa bao lâu.
                pool_timeout=self._pool_timeout_seconds(),

                # Connection quá tuổi sẽ được thay ở checkout tiếp theo.
                pool_recycle=self._pool_recycle_seconds(),

                # Tái sử dụng connection vừa trả gần nhất.
                pool_use_lifo=True,

                # Rollback trạng thái transaction khi trả về pool.
                pool_reset_on_return="rollback",

                # Cache statement đã compile trong SQLAlchemy.
                query_cache_size=self._query_cache_size(),

                # Timeout cấp driver; bên ngoài vẫn có asyncio.timeout.
                connect_args={
                    "timeout": int(
                        math.ceil(self._command_timeout_seconds())
                    )
                },

                # Không log SQL/parameter nhạy cảm mặc định.
                echo=False,

                # Chỉ bật khi cần debug checkout/checkin pool.
                echo_pool=bool(
                    getattr(
                        self.settings,
                        "sql_server_echo_pool",
                        False,
                    )
                ),
            )

            logger.info(
                "Đã tạo SQL Server AsyncEngine: pool_size=%s, "
                "max_overflow=%s, max_concurrent_queries=%s.",
                self._pool_size,
                self._max_overflow,
                self._max_concurrent_queries,
            )

            return self._engine

    async def _execute_statement(self, *, sql_statement: Any, validated_parameters: dict[str, Any], maximum_rows: int, query_key: str) -> tuple[list[Mapping[str, Any]], bool]:
        """
        Kiểm tra kết nối, chạy các câu truy vấn

        Trả:
            (rows, was_truncated)
        """
        # Lấy kết nối tới CSDL
        engine = await self._get_or_create_engine()
        # Thời hian tối đa cho 1 lệnh truy vấn dữ liệu
        command_timeout = self._command_timeout_seconds()

        try:
            # Khi thoát context, connection được trả về pool.
            async with engine.connect() as connection:
                try:
                    async with asyncio.timeout(command_timeout):
                        result = await connection.execute(sql_statement, validated_parameters)

                        # Một procedure có thể chỉ thực hiện công việc và không trả result set.
                        if not result.returns_rows:
                            return [], False

                        # MappingResult cho row theo tên cột.
                        mapping_result = result.mappings()

                        # Lấy thêm một row để biết kết quả bị cắt hay không.
                        fetched_rows = mapping_result.fetchmany(maximum_rows + 1)
                        was_truncated = len(fetched_rows) > maximum_rows
                        rows = list(fetched_rows[:maximum_rows])

                        return rows, was_truncated

                except TimeoutError as exception:
                    self._command_timeouts += 1

                    # Sau cancellation, trạng thái connection có thể không rõ.
                    # Invalidate để pool không tái sử dụng connection đó.
                    try:
                        await connection.invalidate(exception)
                    except Exception:
                        logger.warning("Không invalidate được connection sau timeout.", exc_info=True)

                    raise SqlServerCommandTimeoutError(
                        f"Câu truy vấn SQL {query_key!r} vượt timeout {command_timeout:.1f} giây."
                    ) from exception

        except SqlAlchemyPoolTimeoutError as exception:
            raise SqlServerQueueTimeoutError(
                f"Connection pool không cấp được connection cho {query_key!r} đúng hạn. Pool: {self.pool_status()}."
            ) from exception

        except SqlServerCommandTimeoutError:
            raise

        except DBAPIError as exception:
            if exception.connection_invalidated:
                logger.warning("Connection bị invalidated khi chạy %s.", query_key)

            # Không tự retry procedure. Nếu mạng đứt sau khi server đã chạy, retry có thể khiến procedure thực thi hai lần.
            raise RuntimeError(
                f"SQL Server/ODBC lỗi khi chạy predefined query {query_key!r}."
            ) from exception

        except SQLAlchemyError as exception:
            raise RuntimeError(
                f"SQLAlchemy lỗi khi chạy câu truy vấn đã đăng ký {query_key!r}."
            ) from exception

    @asynccontextmanager
    async def _query_slot(self, query_key: str) -> AsyncIterator[None]:
        """
        Chặn quá nhiều truy vấn chạy đồng thời trong một process.

        Request vượt giới hạn sẽ chờ tối đa queue_timeout rồi bị từ chối.
        """
        # Lấy thời gian tối đa được phép chờ đến lượt và bắt đầu tính thời gian đợi
        queue_timeout = self._queue_timeout_seconds()
        queue_started = time.perf_counter()

        try:
            async with asyncio.timeout(queue_timeout):
                # Xin một slot để chạy truy vấn
                await self._query_semaphore.acquire()
        # Tăng bộ đếm nếu đã quá thời gian chờ queue_timeout mà vẫn chưa xin được slot
        except TimeoutError as exception:
            self._queue_timeouts += 1
            raise SqlServerQueueTimeoutError(
                "Hệ thống SQL đang bận; câu truy vấn "
                f"{query_key!r} không thể thực thi trong "
                f"{queue_timeout:.1f} giây."
            ) from exception
        # Tính thời gian thực tế đã chờ và tăng số lượng truy vấn đang hoạt động trên process này
        queue_elapsed = time.perf_counter() - queue_started
        self._active_queries += 1

        logger.debug( "Câu truy vấn %s lấy slot sau %.3fs; active=%s/%s.", query_key, queue_elapsed, self._active_queries, self._max_concurrent_queries)

        # Trả quyền điều khiển cho khối async with gọi tới hàm này để thực thi SQL
        try:
            yield
        finally:
            # Sau đó giảm số lượng truy vấn và giải phóng slot cho các truy vấn khác
            self._active_queries -= 1
            self._query_semaphore.release()

    def _serialize_rows(self, raw_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Chuyển các RowMapping thành dictionary JSON-safe."""

        output: list[dict[str, Any]] = []

        for raw_row in raw_rows:
            serialized_row: dict[str, Any] = {}

            for column_name, column_value in raw_row.items():
                serialized_row[str(column_name)] = (
                    self._serialize_sql_value(column_value)
                )

            output.append(serialized_row)

        return output

    def _serialize_sql_value(self, value: Any) -> Any:
        """
        Chuyển kiểu SQL thành kiểu JSON serialize.

        Decimal được chuyển thành string để không mất precision.
        """

        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(
            value,
            (
                datetime_module.datetime,
                datetime_module.date,
                datetime_module.time,
            ),
        ):
            return value.isoformat()

        if isinstance(value, decimal.Decimal):
            return str(value)

        if isinstance(value, uuid.UUID):
            return str(value)

        if isinstance(value, (bytes, bytearray, memoryview)):
            raw_bytes = bytes(value)
            maximum_bytes = self._positive_int_setting(
                "sql_server_max_binary_bytes",
                4096,
            )

            if len(raw_bytes) > maximum_bytes:
                return {
                    "encoding": "hex",
                    "truncated": True,
                    "original_bytes": len(raw_bytes),
                    "value": raw_bytes[:maximum_bytes].hex(),
                }

            return raw_bytes.hex()

        return str(value)

    def _normalize_query_key(self, value: str) -> str:
        """Chuẩn hóa query key nhưng không lowercase."""

        if not isinstance(value, str):
            raise TypeError("Từ khoá câu truy vấn phải là string.")

        normalized = value.strip()

        if not normalized:
            raise ValueError("Từ khoá câu truy vấn không được rỗng.")

        if len(normalized) > 200:
            raise ValueError("Từ khoá câu truy vấn cho phép tối đa 200 ký tự.")

        return normalized

    def _normalize_parameters(self, value: Mapping[str, Any] | None) -> dict[str, Any]:
        """
        Sao chép các tham số của câu truy vấn; validation nghiệp vụ do query definition xử lý.
        """
        if value is None:
            return {}

        if not isinstance(value, Mapping):
            raise TypeError("Tham số truy vấn phải là Mapping hoặc None.")

        output: dict[str, Any] = {}
        for key, parameter_value in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Tên tham số truy vấn phải là string không rỗng.")

            output[key.strip()] = parameter_value

        return output

    def _get_registered_sql_text(self, query_definition: Any) -> str:
        """Chỉ lấy chuỗi truy vấn (SQL Text) từ mẫu truy vấn đã được đăng ký)."""
        # Lấy chuỗi truy vấn
        value = getattr(query_definition, "sql_text", None)

        if not isinstance(value, str) or not value.strip():
            raise ValueError("Mẫu truy vấn được đăng ký trước có sql_text rỗng hoặc sai kiểu.")
        # Làm sạch khoảng trắng trước và sau
        return value.strip()

    def _get_maximum_rows(self, query_definition: Any) -> int:
        """Giới hạn số dòng dữ liệu lấy được từ CSDL"""

        value = getattr(query_definition, "maximum_rows", None)

        if (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError(
                "Số dòng dữ liệu được lấy phải được khai báo lúc đăng ký câu truy vấn phải  là số nguyên lớn hơn 0."
            )
        # Lấy giá trị cấu hình: Số dòng tối đa cho phép mà mỗi câu truy vấn được lấy toàn bộ hệ thống
        # Còn value là số dòng tối đa được lấy khi cấu hình riêng cho mỗi câu truy vấn
        global_maximum = self._positive_int_setting("sql_server_global_maximum_rows", 1000,)

        if value > global_maximum:
            raise ValueError(f"Số lượng dòng truy vấn cho phép vượt giới hạn toàn cục: {value} > {global_maximum}.")

        return value

    def _infer_execution_kind(self, sql_text_value: str) -> str:
        """
        Phân loại câu truy vấn là thuần SQL hay chạy procedure
        """

        normalized = sql_text_value.lstrip().upper()

        if (
            normalized.startswith("EXEC ")
            or normalized.startswith("EXECUTE ")
            or (normalized.startswith("SET NOCOUNT ON") and "EXEC " in normalized)
        ):
            return "procedure"

        return "query"

    def _raise_if_closed(self) -> None:
        """Không cho sử dụng service sau shutdown."""

        if self._closed:
            raise SqlServerServiceClosedError(
                "Hệ thống truy vấn CSDL đã được đóng. Hãy mở lại để sử dụng"
            )

    def _positive_int_setting(self, name: str, default: int) -> int:
        value = getattr(self.settings, name, default)
        return self._positive_int(value, name)

    def _non_negative_int_setting(self, name: str, default: int) -> int:
        value = getattr(self.settings, name, default)

        if (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"{name} phải là int >= 0.")

        return value

    def _positive_int(self, value: Any, field_name: str) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"{field_name} phải là int > 0.")

        return value

    def _positive_float_setting(
        self,
        name: str,
        default: float,
    ) -> float:
        raw_value = getattr(self.settings, name, default)

        if isinstance(raw_value, bool):
            raise ValueError(f"{name} phải là số > 0.")

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exception:
            raise TypeError(f"{name} phải là số.") from exception

        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} phải là số hữu hạn > 0.")

        return value

    def _pool_timeout_seconds(self) -> float:
        return self._positive_float_setting(
            "sql_server_pool_timeout_seconds",
            5.0,
        )

    def _pool_recycle_seconds(self) -> int:
        return self._positive_int_setting(
            "sql_server_pool_recycle_seconds",
            1800,
        )

    def _queue_timeout_seconds(self) -> float:
        return self._positive_float_setting("sql_server_queue_timeout_seconds", 2.0)

    def _command_timeout_seconds(self) -> float:
        """
        Thời gian tối đa thực hiện một lệnh truy vấn
        """
        return self._positive_float_setting("sql_server_command_timeout_seconds", 30.0)

    def _slow_query_threshold_seconds(self) -> float:
        return self._positive_float_setting(
            "sql_server_slow_query_seconds",
            2.0,
        )

    def _query_cache_size(self) -> int:
        return self._positive_int_setting(
            "sql_server_query_cache_size",
            500,
        )

# ============================================================
# TEST THẬT TRONG __main__ - KHÔNG DÙNG ARGPARSE
# ============================================================

# Key phải tồn tại trong PredefinedSqlQueryRegistry thật.
# Ví dụ SELECT:
#     TEST_QUERY_KEY = "get_robot_status"
# Ví dụ procedure:
#     TEST_QUERY_KEY = "get_robot_status_procedure"
TEST_QUERY_KEY = "pallet_by_location"

# Parameter phải khớp query definition thật.
TEST_PARAMETERS: dict[str, Any] = {"location_code": "K1-10",}     #{}

# "query" hoặc "procedure".
# Hai loại đều dùng registry; biến này chỉ chọn method dễ đọc trong test.
TEST_EXECUTION_TYPE = "query"

# False: chỉ chạy một lần.
# True: chạy nhiều request đồng thời bằng cùng engine/pool.
TEST_RUN_CONCURRENCY_TEST = True

# Chỉ có hiệu lực khi TEST_RUN_CONCURRENCY_TEST=True.
TEST_CONCURRENT_CALLS = 7


async def _close_if_supported(resource: Any) -> None:
    """Đóng resource hỗ trợ aclose() hoặc close()."""

    if resource is None:
        return

    for method_name in ("aclose", "close"):
        method = getattr(resource, method_name, None)

        if not callable(method):
            continue

        result = method()

        if inspect.isawaitable(result):
            await result

        return


def _response_to_dict(response: SqlExecutionResponse) -> dict[str, Any]:
    """Hỗ trợ cả Pydantic v2 và v1."""

    model_dump = getattr(response, "model_dump", None)

    if callable(model_dump):
        return model_dump(mode="json")

    dict_method = getattr(response, "dict", None)

    if callable(dict_method):
        return dict_method()

    raise TypeError(
        "SqlExecutionResponse không hỗ trợ model_dump() hoặc dict()."
    )


def _print_json(title: str, value: Any) -> None:
    """In object dưới dạng JSON UTF-8 dễ đọc."""

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


async def _execute_test_call(
    service: SafeSqlServerService,
    call_number: int,
) -> dict[str, Any]:
    """Chạy một query/procedure thật và trả kết quả test."""

    started_at = time.perf_counter()

    try:
        if TEST_EXECUTION_TYPE == "procedure":
            response = await service.execute_predefined_procedure(
                procedure_key=TEST_QUERY_KEY,
                parameters=TEST_PARAMETERS,
            )
        elif TEST_EXECUTION_TYPE == "query":
            response = await service.execute_predefined_query(
                query_key=TEST_QUERY_KEY,
                parameters=TEST_PARAMETERS,
            )
        else:
            raise ValueError(
                "TEST_EXECUTION_TYPE phải là 'query' hoặc 'procedure'."
            )

        return {
            "call_number": call_number,
            "success": True,
            "elapsed_seconds": time.perf_counter() - started_at,
            "response": _response_to_dict(response),
        }

    except Exception as exception:
        return {
            "call_number": call_number,
            "success": False,
            "elapsed_seconds": time.perf_counter() - started_at,
            "exception_type": type(exception).__name__,
            "message": str(exception),
        }


async def main() -> None:
    """
    Test SQL Server và registry thật.
    """

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    # Settings thật của project.
    settings = get_settings()

    # Registry thật của project.
    query_registry = PredefinedSqlQueryRegistry()

    # Chỉ tạo một service/engine trong lần test.
    service = SafeSqlServerService(
        settings=settings,
        query_registry=query_registry,
    )

    try:
        # In catalog để biết các key đang có.
        list_for_router = getattr(query_registry, "list_for_router", None)

        if callable(list_for_router):
            _print_json("PREDEFINED SQL CATALOG", list_for_router())

        # Kiểm tra test key trước khi kết nối database.
        try:
            query_definition = query_registry.get(TEST_QUERY_KEY)
        except KeyError as exception:
            raise KeyError(
                f"TEST_QUERY_KEY={TEST_QUERY_KEY!r} không tồn tại. "
                "Hãy xem catalog phía trên và sửa key."
            ) from exception

        _print_json(
            "TEST CONFIGURATION",
            {
                "query_key": TEST_QUERY_KEY,
                "parameters": TEST_PARAMETERS,
                "execution_type": TEST_EXECUTION_TYPE,
                "description": getattr(
                    query_definition,
                    "description",
                    "",
                ),
                "maximum_rows": getattr(
                    query_definition,
                    "maximum_rows",
                    None,
                ),
                "run_concurrency_test": TEST_RUN_CONCURRENCY_TEST,
                "concurrent_calls": TEST_CONCURRENT_CALLS,
            },
        )

        # Warm-up engine và kiểm tra SELECT 1.
        await service.startup()

        _print_json(
            "POOL SAU STARTUP",
            {
                "pool_status": service.pool_status(),
                "statistics": asdict(service.get_statistics()),
            },
        )

        # Chạy một lời gọi thật trước.
        single_result = await _execute_test_call(
            service,
            call_number=1,
        )

        _print_json(
            "KẾT QUẢ GỌI ĐƠN",
            single_result,
        )

        # Stress test nhẹ, chỉ chạy khi bật rõ ràng.
        if TEST_RUN_CONCURRENCY_TEST:
            if (
                isinstance(TEST_CONCURRENT_CALLS, bool)
                or not isinstance(TEST_CONCURRENT_CALLS, int)
                or TEST_CONCURRENT_CALLS <= 0
            ):
                raise ValueError(
                    "TEST_CONCURRENT_CALLS phải là int > 0."
                )

            concurrency_started = time.perf_counter()

            concurrent_results = await asyncio.gather(
                *[
                    _execute_test_call(
                        service,
                        call_number=index,
                    )
                    for index in range(
                        1,
                        TEST_CONCURRENT_CALLS + 1,
                    )
                ]
            )

            _print_json(
                "KẾT QUẢ CONCURRENCY TEST",
                {
                    "total_elapsed_seconds": (
                        time.perf_counter() - concurrency_started
                    ),
                    "calls": concurrent_results,
                    "statistics": asdict(service.get_statistics()),
                },
            )

        _print_json(
            "POOL TRƯỚC SHUTDOWN",
            {
                "pool_status": service.pool_status(),
                "statistics": asdict(service.get_statistics()),
            },
        )

    finally:
        # Dispose connection pool rõ ràng.
        await _close_if_supported(service)


if __name__ == "__main__":
    try:
        # Chỉ một event loop cho toàn bộ AsyncEngine/connection pool.
        asyncio.run(main())

    except KeyboardInterrupt:
        print("Đã dừng bởi người dùng.", file=sys.stderr)
        raise SystemExit(130)

    except Exception as exception:
        logger.exception("SQL Server service test thất bại.")
        print(f"\nLỖI: {exception}", file=sys.stderr)
        raise SystemExit(1)
