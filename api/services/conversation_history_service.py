from __future__ import annotations

"""
Lưu lịch sử hội thoại trong Redis.

Mỗi conversation có:
- metadata: tenant_id, created_at, updated_at, rolling summary;
- Redis list chứa các message gần đây;
- TTL để dữ liệu tự hết hạn;
- Redis lock để hai request không cùng sửa một conversation.

Thiết kế này hỗ trợ nhiều Uvicorn worker/process vì lịch sử không nằm trong RAM
của một process cụ thể.
"""

import json
import logging
import math
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

from redis.asyncio import Redis


logger = logging.getLogger(__name__)


class ConversationNotFoundError(
    LookupError
):
    """Không tìm thấy conversation của tenant."""


class ConversationBusyError(
    RuntimeError
):
    """Conversation đang được một request khác xử lý."""


@dataclass(frozen=True)
class ChatMessage:
    """
    Một message đã được chuẩn hóa.

    metadata không được gửi thẳng cho Llama; nó dùng cho debug/audit.
    """

    role: str
    content: str
    created_at: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ConversationContext:
    """
    Context dùng cho bước viết lại câu hỏi.

    summary:
        Tóm tắt các lượt cũ đã được compact.

    recent_messages:
        Các message gần nhất còn giữ nguyên văn.
    """

    conversation_id: UUID
    tenant_id: str
    summary: str
    recent_messages: list[ChatMessage]


class ConversationHistoryService:
    """
    Repository/service quản lý conversation trong Redis.
    """

    _ALLOWED_ROLES = {
        "user",
        "assistant",
        "system",
    }

    def __init__(
        self,
        redis_client: Redis,
        *,
        key_prefix: str = "rag:conversation",
        ttl_seconds: int = 604800,
        max_stored_messages: int = 100,
        context_message_count: int = 12,
        max_message_characters: int = 12000,
        lock_timeout_seconds: int = 180,
        lock_wait_seconds: float = 2.0,
    ) -> None:
        """
        ttl_seconds:
            604800 = 7 ngày.

        max_stored_messages:
            Giới hạn an toàn trên Redis list.

        context_message_count:
            Số message nguyên văn đưa vào rewriter.

        max_message_characters:
            Chặn một message quá lớn.

        lock_timeout_seconds:
            Redis tự giải phóng lock nếu process chết giữa request.
        """

        self.redis_client = redis_client
        self.key_prefix = (
            key_prefix.strip().rstrip(":")
        )

        self.ttl_seconds = self._positive_int(
            ttl_seconds,
            "ttl_seconds",
        )

        self.max_stored_messages = (
            self._positive_int(
                max_stored_messages,
                "max_stored_messages",
            )
        )

        self.context_message_count = (
            self._positive_int(
                context_message_count,
                "context_message_count",
            )
        )

        if (
            self.context_message_count
            > self.max_stored_messages
        ):
            raise ValueError(
                "context_message_count không được lớn hơn "
                "max_stored_messages."
            )

        self.max_message_characters = (
            self._positive_int(
                max_message_characters,
                "max_message_characters",
            )
        )

        self.lock_timeout_seconds = (
            self._positive_int(
                lock_timeout_seconds,
                "lock_timeout_seconds",
            )
        )

        if (
            isinstance(lock_wait_seconds, bool)
            or not isinstance(
                lock_wait_seconds,
                (int, float),
            )
            or not math.isfinite(
                float(lock_wait_seconds)
            )
            or lock_wait_seconds < 0
        ):
            raise ValueError(
                "lock_wait_seconds phải là số hữu hạn >= 0."
            )

        self.lock_wait_seconds = float(
            lock_wait_seconds
        )

    async def create_conversation(
        self,
        tenant_id: str,
        conversation_id: UUID | None = None,
    ) -> UUID:
        """
        Tạo metadata conversation.

        UUID do server tạo giúp tránh người dùng chọn key Redis tùy ý.
        """

        normalized_tenant = (
            self._normalize_identifier(
                tenant_id,
                "tenant_id",
            )
        )

        selected_id = (
            conversation_id
            if conversation_id is not None
            else uuid4()
        )

        metadata_key = self._metadata_key(
            normalized_tenant,
            selected_id,
        )

        now = self._utc_now()

        # HSETNX giúp không ghi đè created_at nếu ID đã tồn tại.
        async with self.redis_client.pipeline(
            transaction=True
        ) as pipeline:
            pipeline.hsetnx(
                metadata_key,
                "tenant_id",
                normalized_tenant,
            )
            pipeline.hsetnx(
                metadata_key,
                "created_at",
                now,
            )
            pipeline.hset(
                metadata_key,
                mapping={
                    "updated_at": now,
                    "summary": "",
                },
            )
            pipeline.expire(
                metadata_key,
                self.ttl_seconds,
            )
            await pipeline.execute()

        return selected_id

    async def ensure_conversation(
        self,
        tenant_id: str,
        conversation_id: UUID,
    ) -> None:
        """
        Xác minh conversation tồn tại đúng tenant.
        """

        normalized_tenant = (
            self._normalize_identifier(
                tenant_id,
                "tenant_id",
            )
        )

        metadata_key = self._metadata_key(
            normalized_tenant,
            conversation_id,
        )

        tenant_value = await self.redis_client.hget(
            metadata_key,
            "tenant_id",
        )

        if tenant_value is None:
            raise ConversationNotFoundError(
                "Không tìm thấy conversation."
            )

        if tenant_value != normalized_tenant:
            # Trường hợp này gần như không xảy ra vì tenant nằm trong key,
            # nhưng vẫn giữ kiểm tra defense-in-depth.
            raise ConversationNotFoundError(
                "Conversation không thuộc tenant."
            )

    async def load_context(
        self,
        tenant_id: str,
        conversation_id: UUID,
    ) -> ConversationContext:
        """
        Lấy summary + số message gần nhất theo context_message_count.
        """

        return await self._load_context_with_count(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            message_count=self.context_message_count,
        )

    async def _load_context_with_count(
        self,
        *,
        tenant_id: str,
        conversation_id: UUID,
        message_count: int,
    ) -> ConversationContext:
        """
        Hàm đọc dùng chung, không thay đổi state của service.

        Điều này tránh race condition của cách tạm sửa self.context_message_count
        khi nhiều request đồng thời gọi load_context/get_full_history.
        """

        selected_count = self._positive_int(
            message_count,
            "message_count",
        )

        normalized_tenant = self._normalize_identifier(
            tenant_id,
            "tenant_id",
        )

        await self.ensure_conversation(
            normalized_tenant,
            conversation_id,
        )

        metadata_key = self._metadata_key(
            normalized_tenant,
            conversation_id,
        )
        messages_key = self._messages_key(
            normalized_tenant,
            conversation_id,
        )

        async with self.redis_client.pipeline(
            transaction=False
        ) as pipeline:
            pipeline.hget(
                metadata_key,
                "summary",
            )
            pipeline.lrange(
                messages_key,
                -selected_count,
                -1,
            )
            pipeline.expire(
                metadata_key,
                self.ttl_seconds,
            )
            pipeline.expire(
                messages_key,
                self.ttl_seconds,
            )

            (
                summary_value,
                serialized_messages,
                _,
                _,
            ) = await pipeline.execute()

        messages = self._decode_messages(
            serialized_messages
        )

        return ConversationContext(
            conversation_id=conversation_id,
            tenant_id=normalized_tenant,
            summary=str(
                summary_value or ""
            ).strip(),
            recent_messages=messages,
        )

    def _decode_messages(
        self,
        serialized_messages: list[str],
    ) -> list[ChatMessage]:
        """
        Chuyển JSON trong Redis thành ChatMessage, bỏ record hỏng.
        """

        messages: list[ChatMessage] = []

        for raw_message in serialized_messages:
            try:
                decoded = json.loads(raw_message)
            except json.JSONDecodeError:
                logger.warning(
                    "Bỏ message Redis không phải JSON hợp lệ."
                )
                continue

            if not isinstance(decoded, dict):
                continue

            role = str(decoded.get("role", "")).strip()
            content = str(decoded.get("content", "")).strip()
            created_at = str(
                decoded.get("created_at", "")
            ).strip()
            metadata = decoded.get("metadata", {})

            if role not in self._ALLOWED_ROLES or not content:
                continue

            if not isinstance(metadata, dict):
                metadata = {}

            messages.append(
                ChatMessage(
                    role=role,
                    content=content,
                    created_at=created_at,
                    metadata=dict(metadata),
                )
            )

        return messages

    async def append_turn(
        self,
        *,
        tenant_id: str,
        conversation_id: UUID,
        user_message: str,
        assistant_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Ghi atomically một lượt gồm user + assistant.

        RPUSH + LTRIM + EXPIRE nằm trong Redis transaction pipeline.
        """

        normalized_tenant = (
            self._normalize_identifier(
                tenant_id,
                "tenant_id",
            )
        )

        await self.ensure_conversation(
            normalized_tenant,
            conversation_id,
        )

        normalized_user = (
            self._normalize_message_content(
                user_message,
                "user_message",
            )
        )

        normalized_assistant = (
            self._normalize_message_content(
                assistant_message,
                "assistant_message",
            )
        )

        safe_metadata = (
            dict(metadata)
            if isinstance(metadata, dict)
            else {}
        )

        timestamp = self._utc_now()

        user_payload = json.dumps(
            {
                "role": "user",
                "content": normalized_user,
                "created_at": timestamp,
                "metadata": safe_metadata,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

        assistant_payload = json.dumps(
            {
                "role": "assistant",
                "content": normalized_assistant,
                "created_at": timestamp,
                "metadata": safe_metadata,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

        messages_key = self._messages_key(
            normalized_tenant,
            conversation_id,
        )
        metadata_key = self._metadata_key(
            normalized_tenant,
            conversation_id,
        )

        async with self.redis_client.pipeline(
            transaction=True
        ) as pipeline:
            pipeline.rpush(
                messages_key,
                user_payload,
                assistant_payload,
            )

            # Giữ tối đa N message gần nhất.
            pipeline.ltrim(
                messages_key,
                -self.max_stored_messages,
                -1,
            )

            pipeline.hset(
                metadata_key,
                "updated_at",
                timestamp,
            )

            pipeline.expire(
                messages_key,
                self.ttl_seconds,
            )
            pipeline.expire(
                metadata_key,
                self.ttl_seconds,
            )

            await pipeline.execute()

    async def set_summary(
        self,
        *,
        tenant_id: str,
        conversation_id: UUID,
        summary: str,
    ) -> None:
        """
        Cập nhật rolling summary.

        Service tóm tắt có thể gọi hàm này khi muốn compact lịch sử cũ.
        """

        normalized_tenant = (
            self._normalize_identifier(
                tenant_id,
                "tenant_id",
            )
        )

        await self.ensure_conversation(
            normalized_tenant,
            conversation_id,
        )

        normalized_summary = (
            self._normalize_message_content(
                summary,
                "summary",
            )
        )

        metadata_key = self._metadata_key(
            normalized_tenant,
            conversation_id,
        )

        await self.redis_client.hset(
            metadata_key,
            mapping={
                "summary": normalized_summary,
                "updated_at": self._utc_now(),
            },
        )

        await self.redis_client.expire(
            metadata_key,
            self.ttl_seconds,
        )

    async def get_full_history(
        self,
        tenant_id: str,
        conversation_id: UUID,
    ) -> ConversationContext:
        """
        Trả tối đa max_stored_messages mà không sửa shared state.
        """

        return await self._load_context_with_count(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            message_count=self.max_stored_messages,
        )

    async def get_message_count(
        self,
        tenant_id: str,
        conversation_id: UUID,
    ) -> int:
        """Đếm số message hiện còn trong Redis list."""

        normalized_tenant = self._normalize_identifier(
            tenant_id,
            "tenant_id",
        )

        await self.ensure_conversation(
            normalized_tenant,
            conversation_id,
        )

        return int(
            await self.redis_client.llen(
                self._messages_key(
                    normalized_tenant,
                    conversation_id,
                )
            )
        )

    async def replace_messages_and_summary(
        self,
        *,
        tenant_id: str,
        conversation_id: UUID,
        summary: str,
        messages: list[ChatMessage],
    ) -> None:
        """
        Atomically thay history cũ bằng rolling summary + recent messages.

        Hàm được gọi khi conversation đã giữ Redis lock.
        """

        normalized_tenant = self._normalize_identifier(
            tenant_id,
            "tenant_id",
        )

        await self.ensure_conversation(
            normalized_tenant,
            conversation_id,
        )

        normalized_summary = self._normalize_message_content(
            summary,
            "summary",
        )

        if len(messages) > self.max_stored_messages:
            raise ValueError(
                "messages vượt max_stored_messages."
            )

        serialized: list[str] = []

        for message in messages:
            if message.role not in self._ALLOWED_ROLES:
                raise ValueError(
                    f"Role không hợp lệ: {message.role!r}."
                )

            content = self._normalize_message_content(
                message.content,
                "message.content",
            )

            serialized.append(
                json.dumps(
                    {
                        "role": message.role,
                        "content": content,
                        "created_at": message.created_at,
                        "metadata": dict(message.metadata),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            )

        messages_key = self._messages_key(
            normalized_tenant,
            conversation_id,
        )
        metadata_key = self._metadata_key(
            normalized_tenant,
            conversation_id,
        )

        async with self.redis_client.pipeline(
            transaction=True
        ) as pipeline:
            pipeline.delete(messages_key)

            if serialized:
                pipeline.rpush(
                    messages_key,
                    *serialized,
                )

            pipeline.hset(
                metadata_key,
                mapping={
                    "summary": normalized_summary,
                    "updated_at": self._utc_now(),
                },
            )
            pipeline.expire(
                messages_key,
                self.ttl_seconds,
            )
            pipeline.expire(
                metadata_key,
                self.ttl_seconds,
            )
            await pipeline.execute()

    async def delete_conversation(
        self,
        tenant_id: str,
        conversation_id: UUID,
    ) -> int:
        """
        Xóa metadata, messages và lock key.

        Trả số key Redis đã xóa.
        """

        normalized_tenant = (
            self._normalize_identifier(
                tenant_id,
                "tenant_id",
            )
        )

        return int(
            await self.redis_client.delete(
                self._metadata_key(
                    normalized_tenant,
                    conversation_id,
                ),
                self._messages_key(
                    normalized_tenant,
                    conversation_id,
                ),
                self._lock_key(
                    normalized_tenant,
                    conversation_id,
                ),
            )
        )

    @asynccontextmanager
    async def conversation_lock(
        self,
        tenant_id: str,
        conversation_id: UUID,
    ) -> AsyncIterator[None]:
        """
        Khóa theo conversation.

        Hai request đồng thời cho cùng conversation sẽ không được phép
        đọc cùng một lịch sử rồi ghi đè thứ tự hội thoại.
        """

        normalized_tenant = (
            self._normalize_identifier(
                tenant_id,
                "tenant_id",
            )
        )

        lock = self.redis_client.lock(
            self._lock_key(
                normalized_tenant,
                conversation_id,
            ),
            timeout=self.lock_timeout_seconds,
            blocking_timeout=self.lock_wait_seconds,
        )

        acquired = await lock.acquire()

        if not acquired:
            raise ConversationBusyError(
                "Conversation đang xử lý một yêu cầu khác."
            )

        try:
            yield
        finally:
            try:
                await lock.release()
            except Exception:
                # Lock có thể hết hạn nếu request quá lâu.
                logger.warning(
                    "Không thể release conversation lock; "
                    "lock có thể đã hết hạn.",
                    exc_info=True,
                )

    def _metadata_key(
        self,
        tenant_id: str,
        conversation_id: UUID,
    ) -> str:
        return (
            f"{self.key_prefix}:"
            f"{tenant_id}:"
            f"{conversation_id}:meta"
        )

    def _messages_key(
        self,
        tenant_id: str,
        conversation_id: UUID,
    ) -> str:
        return (
            f"{self.key_prefix}:"
            f"{tenant_id}:"
            f"{conversation_id}:messages"
        )

    def _lock_key(
        self,
        tenant_id: str,
        conversation_id: UUID,
    ) -> str:
        return (
            f"{self.key_prefix}:"
            f"{tenant_id}:"
            f"{conversation_id}:lock"
        )

    def _normalize_identifier(
        self,
        value: str,
        field_name: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} phải là string."
            )

        normalized = unicodedata.normalize(
            "NFC",
            value,
        ).strip()

        if not normalized:
            raise ValueError(
                f"{field_name} không được rỗng."
            )

        if len(normalized) > 200:
            raise ValueError(
                f"{field_name} tối đa 200 ký tự."
            )

        return normalized

    def _normalize_message_content(
        self,
        value: str,
        field_name: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} phải là string."
            )

        normalized = unicodedata.normalize(
            "NFC",
            value,
        ).strip()

        if not normalized:
            raise ValueError(
                f"{field_name} không được rỗng."
            )

        if (
            len(normalized)
            > self.max_message_characters
        ):
            raise ValueError(
                f"{field_name} quá dài: "
                f"{len(normalized)} > "
                f"{self.max_message_characters}."
            )

        return normalized

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
            raise ValueError(
                f"{field_name} phải là int > 0."
            )

        return value

    def _utc_now(self) -> str:
        return datetime.now(
            UTC
        ).isoformat()
