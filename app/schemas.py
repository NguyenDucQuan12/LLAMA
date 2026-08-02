from __future__ import annotations

"""
Các schema Pydantic dùng chung cho API và các tầng dịch vụ.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QuestionMode(str, Enum):
    """
    Chế độ xử lý câu hỏi.

    AUTO:
        Hệ thống tự quyết định dùng tài liệu hay predefined SQL query.

    DOCUMENTS:
        Chỉ tìm kiếm tài liệu trong Qdrant.

    SQL:
        Chỉ chạy một predefined SQL query.

    HYBRID:
        Kết hợp tài liệu Qdrant và dữ liệu SQL trong cùng context.
    """

    AUTO = "auto"
    DOCUMENTS = "documents"
    SQL = "sql"
    HYBRID = "hybrid"


class QuestionRequest(BaseModel):
    """
    Dữ liệu đầu vào của endpoint hỏi đáp.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    mode: QuestionMode = QuestionMode.AUTO
    tenant_id: str = Field(default="wms", min_length=1)

    # Có thể giới hạn retrieval trong một tài liệu cụ thể.
    document_id: str | None = None

    # Dùng khi caller muốn chỉ định trực tiếp predefined query.
    sql_query_key: str | None = None
    sql_parameters: dict[str, Any] = Field(default_factory=dict)

    # Khi true, response trả thêm top 20 để debug retrieval.
    include_debug_information: bool = False


class IngestedDocumentSummary(BaseModel):
    """
    Kết quả sau khi chuyển đổi, chunk, embedding và upsert Qdrant.
    """

    document_id: str
    source_file: str
    source_hash: str
    chunk_count: int
    collection_name: str
    chunks_jsonl_path: str
    document_json_path: str


class RetrievedChunkResponse(BaseModel):
    """
    Dạng chunk được phép trả về API.
    """

    point_id: str
    dense_score: float
    reranker_score: float | None = None
    final_score: float | None = None
    document_id: str | None = None
    source_file: str | None = None
    chunk_index: int | None = None
    headings: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    text: str


class SqlExecutionResponse(BaseModel):
    """
    Thông tin kết quả SQL đã được thực thi từ registry an toàn.
    """

    executed: bool
    query_key: str | None = None
    query_description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    missing_parameters: list[str] = Field(default_factory=list)


class AnswerCitation(BaseModel):
    """
    Một nguồn được Llama sử dụng để tạo câu trả lời.
    """

    source_label: str
    source_type: str
    source_file: str | None = None
    chunk_index: int | None = None
    page_numbers: list[int] = Field(default_factory=list)
    headings: list[str] = Field(default_factory=list)
    evidence: str


class QuestionResponse(BaseModel):
    """
    Response hoàn chỉnh của dịch vụ hỏi đáp.
    """

    answer: str
    route: str
    insufficient_context: bool
    citations: list[AnswerCitation] = Field(default_factory=list)

    top_five_documents: list[RetrievedChunkResponse] = Field(
        default_factory=list
    )

    sql_result: SqlExecutionResponse | None = None

    # Chỉ có khi request bật include_debug_information.
    top_twenty_documents: list[RetrievedChunkResponse] | None = None


class HealthResponse(BaseModel):
    """
    Dữ liệu trả về endpoint health.
    """

    status: str
    application: str
    qdrant_configured: bool
    sql_server_enabled: bool
    reranker_enabled: bool
