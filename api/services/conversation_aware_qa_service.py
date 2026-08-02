from __future__ import annotations

"""
Wrapper thêm conversation memory cho QuestionAnsweringService hiện tại.

Pipeline:
1. Tạo/kiểm tra conversation.
2. Khóa conversation.
3. Tải summary + recent messages.
4. Viết lại câu hỏi thành standalone question.
5. Gọi QuestionAnsweringService hiện tại.
6. Ghi câu hỏi gốc và câu trả lời vào Redis.
"""

from uuid import UUID

from app.schemas_conversation import (
    ConversationQuestionRequest,
    ConversationQuestionResponse,
)
from app.services.conversation_history_compactor import (
    ConversationHistoryCompactor,
)
from app.services.conversation_history_service import (
    ConversationHistoryService,
)
from app.services.conversation_question_rewriter import (
    ConversationQuestionRewriter,
)
from app.services.question_answering_service import (
    QuestionAnsweringService,
)


class ConversationAwareQuestionAnsweringService:
    """
    Không sửa pipeline RAG cũ; chỉ bọc thêm lớp conversation.
    """

    def __init__(
        self,
        question_answering_service: (
            QuestionAnsweringService
        ),
        history_service: (
            ConversationHistoryService
        ),
        question_rewriter: (
            ConversationQuestionRewriter
        ),
        history_compactor: (
            ConversationHistoryCompactor | None
        ) = None,
    ) -> None:
        self.question_answering_service = (
            question_answering_service
        )
        self.history_service = (
            history_service
        )
        self.question_rewriter = (
            question_rewriter
        )
        self.history_compactor = (
            history_compactor
        )

    async def answer_question(
        self,
        payload: ConversationQuestionRequest,
    ) -> ConversationQuestionResponse:
        """
        Xử lý một lượt hội thoại.
        """

        conversation_id = (
            payload.conversation_id
        )

        if conversation_id is None:
            conversation_id = (
                await self.history_service
                .create_conversation(
                    payload.tenant_id
                )
            )
        else:
            await self.history_service.ensure_conversation(
                payload.tenant_id,
                conversation_id,
            )

        # Bảo đảm thứ tự message khi frontend gửi hai câu cùng lúc.
        async with (
            self.history_service
            .conversation_lock(
                payload.tenant_id,
                conversation_id,
            )
        ):
            context = (
                await self.history_service
                .load_context(
                    payload.tenant_id,
                    conversation_id,
                )
            )

            rewritten = (
                await self.question_rewriter
                .rewrite(
                    question=payload.question,
                    context=context,
                )
            )

            internal_request = (
                payload.to_question_request(
                    rewritten
                    .standalone_question
                )
            )

            result = (
                await self
                .question_answering_service
                .answer_question(
                    internal_request
                )
            )

            await self.history_service.append_turn(
                tenant_id=payload.tenant_id,
                conversation_id=(
                    conversation_id
                ),
                user_message=payload.question,
                assistant_message=result.answer,
                metadata={
                    "standalone_question": (
                        rewritten
                        .standalone_question
                    ),
                    "route": result.route,
                    "insufficient_context": (
                        result
                        .insufficient_context
                    ),
                },
            )

            # Chỉ phát sinh thêm một lần gọi Llama khi history vượt ngưỡng.
            # Việc compact diễn ra trong conversation lock để không làm sai
            # thứ tự khi có hai request cùng conversation.
            if self.history_compactor is not None:
                await self.history_compactor.compact_if_needed(
                    tenant_id=payload.tenant_id,
                    conversation_id=conversation_id,
                )

        return ConversationQuestionResponse(
            conversation_id=(
                conversation_id
            ),
            original_question=(
                payload.question
            ),
            standalone_question=(
                rewritten.standalone_question
            ),
            used_chat_history=(
                rewritten.used_history
            ),
            result=result,
        )
