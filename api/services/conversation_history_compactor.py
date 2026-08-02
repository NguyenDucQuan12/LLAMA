from __future__ import annotations

"""
Rolling summary cho cuộc trò chuyện dài.

Khi số message vượt compact_trigger_messages:
1. Giữ nguyên keep_recent_messages gần nhất.
2. Gửi summary cũ + các message cũ cho Llama.
3. Lưu summary mới vào Redis.
4. Thay Redis list bằng các message gần nhất.

Nhờ vậy conversation có thể kéo dài mà prompt không tăng vô hạn.
"""

import asyncio
import logging
from typing import Any
from uuid import UUID

from app.clients.ollama_client import OllamaClient
from app.services.conversation_history_service import (
    ChatMessage,
    ConversationHistoryService,
)


logger = logging.getLogger(__name__)


_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
        },
    },
    "required": ["summary"],
    "additionalProperties": False,
}


class ConversationHistoryCompactor:
    """Tóm tắt lịch sử cũ theo ngưỡng."""

    def __init__(
        self,
        history_service: ConversationHistoryService,
        ollama_client: OllamaClient,
        *,
        compact_trigger_messages: int = 24,
        keep_recent_messages: int = 10,
        timeout_seconds: float = 45.0,
        maximum_summary_characters: int = 6000,
    ) -> None:
        self.history_service = history_service
        self.ollama_client = ollama_client

        self.compact_trigger_messages = self._positive_int(
            compact_trigger_messages,
            "compact_trigger_messages",
        )
        self.keep_recent_messages = self._positive_int(
            keep_recent_messages,
            "keep_recent_messages",
        )

        if self.keep_recent_messages >= self.compact_trigger_messages:
            raise ValueError(
                "keep_recent_messages phải nhỏ hơn compact_trigger_messages."
            )

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds phải > 0.")

        self.timeout_seconds = float(timeout_seconds)
        self.maximum_summary_characters = self._positive_int(
            maximum_summary_characters,
            "maximum_summary_characters",
        )

    async def compact_if_needed(
        self,
        *,
        tenant_id: str,
        conversation_id: UUID,
    ) -> bool:
        """
        Trả True nếu đã compact, False nếu chưa đạt ngưỡng hoặc Llama lỗi.

        Caller nên giữ conversation_lock trong khi gọi hàm này.
        """

        message_count = await self.history_service.get_message_count(
            tenant_id,
            conversation_id,
        )

        if message_count <= self.compact_trigger_messages:
            return False

        full_context = await self.history_service.get_full_history(
            tenant_id,
            conversation_id,
        )

        messages = full_context.recent_messages

        if len(messages) <= self.keep_recent_messages:
            return False

        old_messages = messages[:-self.keep_recent_messages]
        recent_messages = messages[-self.keep_recent_messages:]

        history_text = self._format_messages(old_messages)

        system_message = """
Bạn tóm tắt lịch sử hội thoại cho hệ thống RAG nội bộ.

QUY TẮC:
- Giữ lại các thực thể quan trọng: tên người, robot, pallet, QR, vị trí, query key,
  stored procedure, mốc thời gian và con số.
- Giữ các quyết định, yêu cầu của người dùng, vấn đề chưa giải quyết và kết quả
  đã xác nhận.
- Không thêm kiến thức mới.
- Không suy đoán.
- Không viết dài dòng.
- Summary phải giúp hiểu các câu hỏi follow-up trong tương lai.
- Trả đúng JSON schema.
""".strip()

        user_message = (
            "SUMMARY CŨ:\n"
            + (full_context.summary or "(chưa có)")
            + "\n\nCÁC LƯỢT CẦN GỘP:\n"
            + history_text
        )

        try:
            async with asyncio.timeout(self.timeout_seconds):
                output = await self.ollama_client.chat_json(
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
                    json_schema=_SUMMARY_SCHEMA,
                    temperature=0.0,
                )
        except Exception:
            logger.exception(
                "Không thể compact conversation %s.",
                conversation_id,
            )
            return False

        if not isinstance(output, dict):
            return False

        summary = str(output.get("summary", "")).strip()

        if not summary:
            return False

        if len(summary) > self.maximum_summary_characters:
            summary = summary[: self.maximum_summary_characters].rstrip()

        await self.history_service.replace_messages_and_summary(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            summary=summary,
            messages=recent_messages,
        )

        logger.info(
            "Đã compact conversation %s: %s -> %s messages.",
            conversation_id,
            len(messages),
            len(recent_messages),
        )

        return True

    def _format_messages(
        self,
        messages: list[ChatMessage],
    ) -> str:
        lines: list[str] = []

        for message in messages:
            role = {
                "user": "Người dùng",
                "assistant": "Trợ lý",
                "system": "Hệ thống",
            }.get(message.role, message.role)

            lines.append(f"{role}: {message.content}")

        return "\n".join(lines)

    def _positive_int(
        self,
        value: int,
        field_name: str,
    ) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"{field_name} phải là int > 0.")

        return value
