from __future__ import annotations

"""
Thực thi predefined SQL query bằng SQLAlchemy AsyncIO và aioodbc.
"""

import datetime as datetime_module
import decimal
import logging
import uuid
from typing import Any

from sqlalchemy import URL, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from config import Settings
from schemas import SqlExecutionResponse
from sql.query_registry import PredefinedSqlQueryRegistry


logger = logging.getLogger(__name__)


class SafeSqlServerService:
    """
    Chỉ thực thi query lấy từ PredefinedSqlQueryRegistry.

    Không có method nhận SQL text từ người dùng hoặc từ Llama.
    """

    def __init__(
        self,
        settings: Settings,
        query_registry: PredefinedSqlQueryRegistry,
    ) -> None:
        self.settings = settings
        self.query_registry = query_registry
        self.engine: AsyncEngine | None = None

    async def close(self) -> None:
        """
        Dispose connection pool nếu SQL Server đã được khởi tạo.
        """

        if self.engine is not None:
            await self.engine.dispose()
            self.engine = None

    def is_enabled(self) -> bool:
        """
        SQL chỉ được xem là bật khi flag và connection string đều có.
        """

        return bool(
            self.settings.sql_server_enabled
            and self.settings.sql_server_odbc_connection_string.strip()
        )

    async def execute_predefined_query(
        self,
        query_key: str,
        parameters: dict[str, Any],
    ) -> SqlExecutionResponse:
        """
        Validate query key, validate bind parameter và thực thi SELECT.
        """

        if not self.is_enabled():
            raise RuntimeError(
                "SQL Server chưa được bật hoặc chưa có connection string."
            )

        query_definition = self.query_registry.get(query_key)

        missing_parameters = (
            query_definition.missing_required_parameters(parameters)
        )

        if missing_parameters:
            return SqlExecutionResponse(
                executed=False,
                query_key=query_definition.key,
                query_description=query_definition.description,
                parameters=parameters,
                rows=[],
                row_count=0,
                missing_parameters=missing_parameters,
            )

        validated_parameters = query_definition.validate_parameters(
            parameters
        )

        engine = self._get_or_create_engine()
        sql_statement = text(query_definition.sql_text)

        async with engine.connect() as connection:
            execution_result = await connection.execute(
                sql_statement,
                validated_parameters,
            )

            mapping_result = execution_result.mappings()
            raw_rows = mapping_result.fetchmany(
                query_definition.maximum_rows
            )

        serialized_rows: list[dict[str, Any]] = []

        for raw_row in raw_rows:
            serialized_row: dict[str, Any] = {}

            for column_name, column_value in raw_row.items():
                serialized_row[str(column_name)] = (
                    self._serialize_sql_value(column_value)
                )

            serialized_rows.append(serialized_row)

        logger.info(
            "Đã chạy predefined query %s và nhận %s dòng.",
            query_definition.key,
            len(serialized_rows),
        )

        return SqlExecutionResponse(
            executed=True,
            query_key=query_definition.key,
            query_description=query_definition.description,
            parameters={
                key: self._serialize_sql_value(value)
                for key, value in validated_parameters.items()
            },
            rows=serialized_rows,
            row_count=len(serialized_rows),
            missing_parameters=[],
        )

    def _get_or_create_engine(self) -> AsyncEngine:
        """
        Tạo SQLAlchemy async engine bằng ODBC connection string.
        """

        if self.engine is not None:
            return self.engine

        connection_url = URL.create(
            drivername="mssql+aioodbc",
            query={
                "odbc_connect": (
                    self.settings.sql_server_odbc_connection_string
                )
            },
        )

        self.engine = create_async_engine(
            connection_url,
            pool_pre_ping=True,
            pool_size=self.settings.sql_server_pool_size,
            max_overflow=self.settings.sql_server_max_overflow,
            connect_args={
                "timeout": self.settings.sql_server_command_timeout_seconds
            },
        )

        return self.engine

    def _serialize_sql_value(self, value: Any) -> Any:
        """
        Chuyển kiểu dữ liệu SQL thành kiểu có thể JSON serialize.
        """

        if value is None:
            return None

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):
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
            return float(value)

        if isinstance(value, uuid.UUID):
            return str(value)

        if isinstance(value, bytes):
            return value.hex()

        return str(value)
