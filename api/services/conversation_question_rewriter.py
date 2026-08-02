from __future__ import annotations

"""
Dùng lịch sử hội thoại để viết lại câu hỏi phụ thuộc ngữ cảnh thành một câu hỏi
độc lập trước khi đưa vào intent router, embedding và retrieval.

Không gửi toàn bộ lịch sử thô vào DenseDocumentRetriever vì điều đó dễ làm
vector query bị nhiễu và tăng token không giới hạn.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from app.clients.ollama_client import OllamaClient
from app.services.conversation_history_service import (
    ConversationContext,
)


logger = logging.getLogger(__name__)


_REWRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "standalone_question": {
            "type": "string",
        },
        "used_history": {
            "type": "boolean",
        },
    },
    "required": [
        "standalone_question",
        "used_history",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class RewrittenQuestion:
    standalone_question: str
    used_history: bool


class ConversationQuestionRewriter:
    """
    Rewriter an toàn.

    Nếu Llama lỗi hoặc timeout, fallback về câu hỏi nguyên bản để API vẫn chạy.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        *,
        timeout_seconds: float = 30.0,
        maximum_context_characters: int = 12000,
        maximum_output_characters: int = 4000,
    ) -> None:
        self.ollama_client = (
            ollama_client
        )

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds phải > 0."
            )

        self.timeout_seconds = float(
            timeout_seconds
        )

        self.maximum_context_characters = (
            int(maximum_context_characters)
        )
        self.maximum_output_characters = (
            int(maximum_output_characters)
        )

        if (
            self.maximum_context_characters
            <= 0
            or self.maximum_output_characters
            <= 0
        ):
            raise ValueError(
                "Giới hạn ký tự phải > 0."
            )

    async def rewrite(
        self,
        *,
        question: str,
        context: ConversationContext,
    ) -> RewrittenQuestion:
        """
        Viết lại câu hỏi.

        Không có history:
            trả nguyên câu hỏi, không gọi Llama.

        Có history:
            Llama chỉ được làm rõ đại từ/tham chiếu, không tự trả lời.
        """

        normalized_question = (
            question.strip()
        )

        if not normalized_question:
            raise ValueError(
                "question không được rỗng."
            )

        if (
            not context.summary
            and not context.recent_messages
        ):
            return RewrittenQuestion(
                standalone_question=(
                    normalized_question
                ),
                used_history=False,
            )

        history_text = (
            self._build_history_text(
                context
            )
        )

        system_message = """
Bạn là bộ viết lại câu hỏi cho hệ thống RAG.

NHIỆM VỤ:
- Dùng lịch sử để biến câu hỏi mới thành một câu hỏi độc lập, đầy đủ ngữ cảnh.
- Chỉ bổ sung thông tin đã xuất hiện rõ trong lịch sử.
- Giữ nguyên mã kỹ thuật, tên robot, pallet, QR, stored procedure và con số.
- Giữ các yêu cầu hội thoại còn áp dụng như ngôn ngữ, độ dài, định dạng hoặc
  phạm vi trả lời; đưa chúng vào câu hỏi độc lập khi cần.
- Không trả lời câu hỏi.
- Không thêm kiến thức bên ngoài.
- Nếu câu hỏi đã độc lập, giữ nguyên.
- Trả đúng JSON schema.
""".strip()

        user_message = (
            "LỊCH SỬ HỘI THOẠI:\n"
            + history_text
            + "\n\nCÂU HỎI MỚI:\n"
            + normalized_question
        )

        try:
            async with asyncio.timeout(
                self.timeout_seconds
            ):
                output = (
                    await self.ollama_client
                    .chat_json(
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    system_message
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    user_message
                                ),
                            },
                        ],
                        json_schema=(
                            _REWRITE_SCHEMA
                        ),
                        temperature=0.0,
                    )
                )
        except Exception:
            logger.exception(
                "Không thể viết lại câu hỏi; "
                "dùng câu hỏi nguyên bản."
            )

            return RewrittenQuestion(
                standalone_question=(
                    normalized_question
                ),
                used_history=False,
            )

        if not isinstance(output, dict):
            return RewrittenQuestion(
                standalone_question=(
                    normalized_question
                ),
                used_history=False,
            )

        standalone = str(
            output.get(
                "standalone_question",
                "",
            )
        ).strip()

        used_history = output.get(
            "used_history",
            False,
        )

        if (
            not standalone
            or len(standalone)
            > self.maximum_output_characters
            or not isinstance(
                used_history,
                bool,
            )
        ):
            return RewrittenQuestion(
                standalone_question=(
                    normalized_question
                ),
                used_history=False,
            )

        return RewrittenQuestion(
            standalone_question=standalone,
            used_history=used_history,
        )

    def _build_history_text(
        self,
        context: ConversationContext,
    ) -> str:
        """
        Tạo lịch sử rút gọn.

        Summary được đặt trước, sau đó là các message gần nhất.
        """

        sections: list[str] = []

        if context.summary:
            sections.append(
                "TÓM TẮT CÁC LƯỢT CŨ:\n"
                + context.summary
            )

        if context.recent_messages:
            lines: list[str] = []

            for message in (
                context.recent_messages
            ):
                role_label = {
                    "user": "Người dùng",
                    "assistant": "Trợ lý",
                    "system": "Hệ thống",
                }.get(
                    message.role,
                    message.role,
                )

                lines.append(
                    f"{role_label}: "
                    f"{message.content}"
                )

            sections.append(
                "CÁC LƯỢT GẦN NHẤT:\n"
                + "\n".join(lines)
            )

        text = "\n\n".join(
            sections
        )

        if (
            len(text)
            > self.maximum_context_characters
        ):
            # Ưu tiên phần gần nhất ở cuối.
            text = text[
                -self.maximum_context_characters:
            ]

        return text
