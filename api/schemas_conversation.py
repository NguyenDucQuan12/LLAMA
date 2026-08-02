from __future__ import annotations

"""
Schema dành riêng cho API hội thoại.

Không sửa trực tiếp QuestionRequest/QuestionResponse hiện tại để tránh làm
ảnh hưởng các CLI hoặc endpoint cũ đang sử dụng chúng.
"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas import (
    QuestionMode,
    QuestionRequest,
    QuestionResponse,
)


class ConversationQuestionRequest(BaseModel):
    """
    Request gửi tới endpoint /v1/ask.

    conversation_id:
        - None: server tạo cuộc trò chuyện mới.
        - Có giá trị: tiếp tục cuộc trò chuyện cũ.

    Các field còn lại tương ứng QuestionRequest hiện tại.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=4000,
    )

    tenant_id: str = Field(
        min_length=1,
        max_length=200,
    )

    conversation_id: UUID | None = None

    document_id: str | None = Field(
        default=None,
        max_length=500,
    )

    mode: QuestionMode = QuestionMode.AUTO

    sql_query_key: str | None = Field(
        default=None,
        max_length=200,
    )

    sql_parameters: dict[str, Any] = Field(
        default_factory=dict
    )

    include_debug_information: bool = False

    def to_question_request(
        self,
        standalone_question: str,
    ) -> QuestionRequest:
        """
        Chuyển request hội thoại thành QuestionRequest của pipeline RAG.

        Pipeline dùng standalone_question thay cho câu hỏi phụ thuộc ngữ cảnh.
        Ví dụ:
            "Còn cách xử lý thì sao?"
        được viết lại thành:
            "Cách xử lý khi robot AGV-01 bị treo nhiệm vụ là gì?"
        """

        return QuestionRequest(
            question=standalone_question,
            tenant_id=self.tenant_id,
            document_id=self.document_id,
            mode=self.mode,
            sql_query_key=self.sql_query_key,
            sql_parameters=dict(
                self.sql_parameters
            ),
            include_debug_information=(
                self.include_debug_information
            ),
        )


class ConversationQuestionResponse(BaseModel):
    """
    Response hội thoại.

    result giữ nguyên QuestionResponse hiện tại nên frontend vẫn đọc được
    answer, citations, SQL result và top documents.
    """

    model_config = ConfigDict(extra="forbid")

    conversation_id: UUID
    original_question: str
    standalone_question: str
    used_chat_history: bool
    result: QuestionResponse


class CreateConversationRequest(BaseModel):
    """
    Tạo trước một conversation ID.

    Endpoint /v1/ask cũng có thể tự tạo nếu conversation_id=None.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(
        min_length=1,
        max_length=200,
    )


class CreateConversationResponse(BaseModel):
    conversation_id: UUID
    tenant_id: str


class ConversationMessageResponse(BaseModel):
    role: str
    content: str
    created_at: str


class ConversationHistoryResponse(BaseModel):
    conversation_id: UUID
    tenant_id: str
    summary: str
    messages: list[
        ConversationMessageResponse
    ]
