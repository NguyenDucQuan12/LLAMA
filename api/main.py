from __future__ import annotations

"""
FastAPI entry point có conversation history.

Chạy:
    uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

Kiến trúc:
- Lifespan tạo một lần các connection pool/model/service dùng chung.
- Redis lưu lịch sử theo conversation_id.
- /v1/ask tự tạo conversation nếu client chưa gửi ID.
- Câu hỏi follow-up được viết lại thành câu hỏi độc lập trước retrieval.
"""

import asyncio
import inspect
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

import redis.asyncio as redis
from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
)
from fastapi.middleware.cors import (
    CORSMiddleware,
)
from fastapi.responses import JSONResponse

from app.clients.ollama_client import (
    OllamaClient,
)
from app.clients.qdrant_repository import (
    QdrantRepository,
)
from app.config import Settings, get_settings
from app.generation.answer_generator import (
    RagAnswerGenerator,
)
from app.generation.context_builder import (
    AnswerContextBuilder,
)
from app.logging_setup import configure_logging
from app.retrieval.reranker import (
    CrossEncoderDocumentReranker,
)
from app.retrieval.retriever import (
    DenseDocumentRetriever,
)
from app.schemas import HealthResponse
from api.schemas_conversation import (
    ConversationHistoryResponse,
    ConversationMessageResponse,
    ConversationQuestionRequest,
    ConversationQuestionResponse,
    CreateConversationRequest,
    CreateConversationResponse,
)
from app.services.conversation_aware_qa_service import (
    ConversationAwareQuestionAnsweringService,
)
from app.services.conversation_history_compactor import (
    ConversationHistoryCompactor,
)
from app.services.conversation_history_service import (
    ConversationBusyError,
    ConversationHistoryService,
    ConversationNotFoundError,
)
from app.services.conversation_question_rewriter import (
    ConversationQuestionRewriter,
)
from app.services.question_answering_service import (
    QuestionAnsweringService,
)
from app.sql.intent_router import (
    QuestionIntentRouter,
)
from app.sql.query_registry import (
    PredefinedSqlQueryRegistry,
)
from app.sql.sql_service import (
    SafeSqlServerService,
)


logger = logging.getLogger(__name__)


@dataclass
class ApplicationContainer:
    """
    Container giữ dependency dùng chung.

    Dùng một object thay vì đặt hàng chục field rời lên app.state.
    """

    settings: Settings
    redis_client: redis.Redis
    ollama_client: OllamaClient
    qdrant_repository: QdrantRepository
    query_registry: PredefinedSqlQueryRegistry
    sql_server_service: SafeSqlServerService
    history_service: ConversationHistoryService
    conversation_service: (
        ConversationAwareQuestionAnsweringService
    )
    ask_semaphore: asyncio.Semaphore


async def _close_if_supported(
    resource: Any,
) -> None:
    """
    Đóng resource nếu có close()/aclose().
    """

    if resource is None:
        return

    for method_name in (
        "aclose",
        "close",
    ):
        method = getattr(
            resource,
            method_name,
            None,
        )

        if not callable(method):
            continue

        result = method()

        if inspect.isawaitable(result):
            await result

        return


def _get_container(
    request: Request,
) -> ApplicationContainer:
    """
    Lấy dependency container đã được lifespan tạo.
    """

    container = getattr(
        request.app.state,
        "container",
        None,
    )

    if not isinstance(
        container,
        ApplicationContainer,
    ):
        raise RuntimeError(
            "Application container chưa sẵn sàng."
        )

    return container


@asynccontextmanager
async def application_lifespan(
    fastapi_application: FastAPI,
) -> AsyncIterator[None]:
    """
    Startup:
    - đọc Settings;
    - tạo Redis/Ollama/Qdrant/SQL/model/service;
    - kiểm tra Redis và Qdrant.

    Shutdown:
    - đóng resource theo thứ tự ngược.
    """

    settings = get_settings()
    configure_logging(
        settings.log_level
    )

    resources: list[Any] = []

    try:
        # ----------------------------------------------------
        # REDIS
        # ----------------------------------------------------

        redis_url = getattr(
            settings,
            "redis_url",
            "redis://localhost:6379/0",
        )

        redis_client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=float(
                getattr(
                    settings,
                    "redis_connect_timeout_seconds",
                    3.0,
                )
            ),
            socket_timeout=float(
                getattr(
                    settings,
                    "redis_command_timeout_seconds",
                    5.0,
                )
            ),
            health_check_interval=30,
        )
        resources.append(redis_client)

        # Fail-fast: conversation API cần Redis.
        await redis_client.ping()

        # ----------------------------------------------------
        # SHARED CLIENTS
        # ----------------------------------------------------

        ollama_client = OllamaClient(
            settings
        )
        resources.append(ollama_client)

        qdrant_repository = (
            QdrantRepository(
                settings
            )
        )
        resources.append(
            qdrant_repository
        )

        await qdrant_repository.ensure_collection()

        # ----------------------------------------------------
        # SQL
        # ----------------------------------------------------

        query_registry = (
            PredefinedSqlQueryRegistry()
        )

        sql_server_service = (
            SafeSqlServerService(
                settings=settings,
                query_registry=(
                    query_registry
                ),
            )
        )
        resources.append(
            sql_server_service
        )

        # ----------------------------------------------------
        # BASE RAG PIPELINE
        # ----------------------------------------------------

        retriever = DenseDocumentRetriever(
            settings=settings,
            ollama_client=ollama_client,
            qdrant_repository=(
                qdrant_repository
            ),
        )

        reranker = (
            CrossEncoderDocumentReranker(
                settings
            )
        )

        router_arguments: dict[str, Any] = {
            "ollama_client": ollama_client,
            "query_registry": query_registry,
            "sql_server_service": sql_server_service,
        }

        # Tương thích cả router cũ (không có settings) và router mới.
        if "settings" in inspect.signature(
            QuestionIntentRouter
        ).parameters:
            router_arguments["settings"] = settings

        intent_router = QuestionIntentRouter(
            **router_arguments
        )

        context_builder = (
            AnswerContextBuilder(
                settings
            )
        )

        answer_generator = (
            RagAnswerGenerator(
                ollama_client
            )
        )

        base_question_service = (
            QuestionAnsweringService(
                settings=settings,
                retriever=retriever,
                reranker=reranker,
                intent_router=intent_router,
                sql_server_service=(
                    sql_server_service
                ),
                context_builder=(
                    context_builder
                ),
                answer_generator=(
                    answer_generator
                ),
            )
        )

        # ----------------------------------------------------
        # CONVERSATION MEMORY
        # ----------------------------------------------------

        history_service = (
            ConversationHistoryService(
                redis_client,
                key_prefix=getattr(
                    settings,
                    "chat_history_key_prefix",
                    "rag:conversation",
                ),
                ttl_seconds=int(
                    getattr(
                        settings,
                        "chat_history_ttl_seconds",
                        604800,
                    )
                ),
                max_stored_messages=int(
                    getattr(
                        settings,
                        "chat_history_max_stored_messages",
                        100,
                    )
                ),
                context_message_count=int(
                    getattr(
                        settings,
                        "chat_history_context_messages",
                        12,
                    )
                ),
                max_message_characters=int(
                    getattr(
                        settings,
                        "chat_history_max_message_characters",
                        12000,
                    )
                ),
                lock_timeout_seconds=int(
                    getattr(
                        settings,
                        "chat_history_lock_timeout_seconds",
                        180,
                    )
                ),
                lock_wait_seconds=float(
                    getattr(
                        settings,
                        "chat_history_lock_wait_seconds",
                        2.0,
                    )
                ),
            )
        )

        question_rewriter = (
            ConversationQuestionRewriter(
                ollama_client,
                timeout_seconds=float(
                    getattr(
                        settings,
                        "conversation_rewrite_timeout_seconds",
                        30.0,
                    )
                ),
                maximum_context_characters=int(
                    getattr(
                        settings,
                        "conversation_context_max_characters",
                        12000,
                    )
                ),
                maximum_output_characters=int(
                    getattr(
                        settings,
                        "question_max_characters",
                        4000,
                    )
                ),
            )
        )

        history_compactor = ConversationHistoryCompactor(
            history_service=history_service,
            ollama_client=ollama_client,
            compact_trigger_messages=int(
                getattr(
                    settings,
                    "chat_history_compact_trigger_messages",
                    24,
                )
            ),
            keep_recent_messages=int(
                getattr(
                    settings,
                    "chat_history_keep_recent_messages",
                    10,
                )
            ),
            timeout_seconds=float(
                getattr(
                    settings,
                    "chat_history_compact_timeout_seconds",
                    45.0,
                )
            ),
            maximum_summary_characters=int(
                getattr(
                    settings,
                    "chat_history_max_summary_characters",
                    6000,
                )
            ),
        )

        conversation_service = (
            ConversationAwareQuestionAnsweringService(
                question_answering_service=(
                    base_question_service
                ),
                history_service=(
                    history_service
                ),
                question_rewriter=(
                    question_rewriter
                ),
                history_compactor=(
                    history_compactor
                ),
            )
        )

        maximum_concurrent_questions = int(
            getattr(
                settings,
                "api_max_concurrent_questions",
                2,
            )
        )

        if maximum_concurrent_questions <= 0:
            raise ValueError(
                "api_max_concurrent_questions phải > 0."
            )

        fastapi_application.state.container = (
            ApplicationContainer(
                settings=settings,
                redis_client=redis_client,
                ollama_client=(
                    ollama_client
                ),
                qdrant_repository=(
                    qdrant_repository
                ),
                query_registry=(
                    query_registry
                ),
                sql_server_service=(
                    sql_server_service
                ),
                history_service=(
                    history_service
                ),
                conversation_service=(
                    conversation_service
                ),
                ask_semaphore=(
                    asyncio.Semaphore(
                        maximum_concurrent_questions
                    )
                ),
            )
        )

        logger.info(
            "Application startup hoàn tất."
        )

        yield

    finally:
        # Đóng ngược thứ tự tạo.
        for resource in reversed(
            resources
        ):
            try:
                await _close_if_supported(
                    resource
                )
            except Exception:
                logger.exception(
                    "Không đóng được resource %s.",
                    type(resource).__name__,
                )

        if hasattr(
            fastapi_application.state,
            "container",
        ):
            del fastapi_application.state.container


def create_application() -> FastAPI:
    """
    Factory tạo FastAPI app.

    get_settings() ở đây chỉ dùng metadata/CORS.
    Resource mạng vẫn chỉ được tạo trong lifespan.
    """

    settings = get_settings()

    application = FastAPI(
        title=settings.application_name,
        version="2.0.0",
        lifespan=application_lifespan,
    )

    allowed_origins = (
        settings
        .get_cors_allowed_origins()
    )

    allow_credentials = bool(
        getattr(
            settings,
            "cors_allow_credentials",
            True,
        )
    )

    if (
        allow_credentials
        and "*" in allowed_origins
    ):
        raise ValueError(
            "Không nên dùng CORS origin '*' cùng credentials=True. "
            "Hãy khai báo origin frontend cụ thể."
        )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=(
            allow_credentials
        ),
        allow_methods=[
            "GET",
            "POST",
            "DELETE",
        ],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Request-ID",
        ],
        expose_headers=[
            "X-Request-ID",
            "X-Process-Time-Ms",
        ],
    )

    # --------------------------------------------------------
    # REQUEST ID + PROCESS TIME
    # --------------------------------------------------------

    @application.middleware("http")
    async def request_context_middleware(
        request: Request,
        call_next: Any,
    ) -> Response:
        request_id = (
            request.headers.get(
                "X-Request-ID"
            )
            or str(uuid4())
        )

        request.state.request_id = (
            request_id
        )

        started_at = (
            time.perf_counter()
        )

        response = await call_next(
            request
        )

        elapsed_ms = (
            time.perf_counter()
            - started_at
        ) * 1000

        response.headers[
            "X-Request-ID"
        ] = request_id

        response.headers[
            "X-Process-Time-Ms"
        ] = f"{elapsed_ms:.2f}"

        return response

    # --------------------------------------------------------
    # EXCEPTION HANDLERS
    # --------------------------------------------------------

    @application.exception_handler(
        ConversationNotFoundError
    )
    async def conversation_not_found_handler(
        request: Request,
        exception: ConversationNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "detail": str(exception),
                "request_id": getattr(
                    request.state,
                    "request_id",
                    None,
                ),
            },
        )

    @application.exception_handler(
        ConversationBusyError
    )
    async def conversation_busy_handler(
        request: Request,
        exception: ConversationBusyError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "detail": str(exception),
                "request_id": getattr(
                    request.state,
                    "request_id",
                    None,
                ),
            },
        )

    @application.exception_handler(
        ValueError
    )
    async def value_error_handler(
        request: Request,
        exception: ValueError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "detail": str(exception),
                "request_id": getattr(
                    request.state,
                    "request_id",
                    None,
                ),
            },
        )

    @application.exception_handler(
        RuntimeError
    )
    async def runtime_error_handler(
        request: Request,
        exception: RuntimeError,
    ) -> JSONResponse:
        logger.exception(
            "Runtime error tại request %s.",
            getattr(
                request.state,
                "request_id",
                None,
            ),
            exc_info=exception,
        )

        expose_details = bool(
            getattr(
                settings,
                "api_expose_internal_errors",
                False,
            )
        )

        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    str(exception)
                    if expose_details
                    else (
                        "Dịch vụ tạm thời không sẵn sàng."
                    )
                ),
                "request_id": getattr(
                    request.state,
                    "request_id",
                    None,
                ),
            },
        )

    # --------------------------------------------------------
    # HEALTH
    # --------------------------------------------------------

    @application.get(
        "/health/live",
    )
    async def liveness() -> dict[
        str,
        str,
    ]:
        """
        Chỉ xác minh process FastAPI còn sống.
        Không gọi dịch vụ bên ngoài.
        """

        return {
            "status": "ok",
        }

    @application.get(
        "/health",
        response_model=HealthResponse,
    )
    async def health(
        request: Request,
    ) -> HealthResponse:
        """
        Health tương thích endpoint cũ.
        """

        container = _get_container(
            request
        )

        return HealthResponse(
            status="ok",
            application=(
                container
                .settings
                .application_name
            ),
            qdrant_configured=bool(
                container
                .settings
                .qdrant_url
            ),
            sql_server_enabled=(
                container
                .sql_server_service
                .is_enabled()
            ),
            reranker_enabled=(
                container
                .settings
                .reranker_enabled
            ),
        )

    @application.get(
        "/health/ready",
    )
    async def readiness(
        request: Request,
    ) -> dict[str, Any]:
        """
        Kiểm tra Redis và Qdrant với timeout ngắn.

        Không gọi embedding/chat model vì có thể tốn tài nguyên;
        Ollama được báo theo cấu hình.
        """

        container = _get_container(
            request
        )

        checks: dict[str, Any] = {}

        try:
            async with asyncio.timeout(2.0):
                checks["redis"] = bool(
                    await container
                    .redis_client
                    .ping()
                )
        except Exception:
            checks["redis"] = False

        try:
            async with asyncio.timeout(3.0):
                await container.qdrant_repository.ensure_collection()

            checks["qdrant"] = True
        except Exception:
            checks["qdrant"] = False

        checks["sql_enabled"] = (
            container
            .sql_server_service
            .is_enabled()
        )

        ready = bool(
            checks["redis"]
            and checks["qdrant"]
        )

        if not ready:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "checks": checks,
                },
            )

        return {
            "status": "ready",
            "checks": checks,
        }

    # --------------------------------------------------------
    # SQL CATALOG
    # --------------------------------------------------------

    @application.get(
        "/v1/sql/query-catalog",
    )
    async def list_predefined_sql_queries(
        request: Request,
    ) -> dict[str, object]:
        """
        Chỉ trả catalog router an toàn, không trả SQL text.
        """

        container = _get_container(
            request
        )

        return {
            "queries": (
                container
                .query_registry
                .list_for_router()
            )
        }

    # --------------------------------------------------------
    # CONVERSATION ENDPOINTS
    # --------------------------------------------------------

    @application.post(
        "/v1/conversations",
        response_model=(
            CreateConversationResponse
        ),
        status_code=201,
    )
    async def create_conversation(
        payload: CreateConversationRequest,
        request: Request,
    ) -> CreateConversationResponse:
        container = _get_container(
            request
        )

        conversation_id = (
            await container
            .history_service
            .create_conversation(
                payload.tenant_id
            )
        )

        return CreateConversationResponse(
            conversation_id=(
                conversation_id
            ),
            tenant_id=payload.tenant_id,
        )

    @application.get(
        "/v1/conversations/{conversation_id}",
        response_model=(
            ConversationHistoryResponse
        ),
    )
    async def get_conversation_history(
        conversation_id: UUID,
        request: Request,
        tenant_id: str = Query(
            min_length=1,
            max_length=200,
        ),
    ) -> ConversationHistoryResponse:
        container = _get_container(
            request
        )

        context = (
            await container
            .history_service
            .get_full_history(
                tenant_id,
                conversation_id,
            )
        )

        return ConversationHistoryResponse(
            conversation_id=(
                conversation_id
            ),
            tenant_id=tenant_id,
            summary=context.summary,
            messages=[
                ConversationMessageResponse(
                    role=message.role,
                    content=message.content,
                    created_at=(
                        message.created_at
                    ),
                )
                for message in (
                    context.recent_messages
                )
            ],
        )

    @application.delete(
        "/v1/conversations/{conversation_id}",
        status_code=204,
    )
    async def delete_conversation(
        conversation_id: UUID,
        request: Request,
        tenant_id: str = Query(
            min_length=1,
            max_length=200,
        ),
    ) -> Response:
        container = _get_container(
            request
        )

        deleted = (
            await container
            .history_service
            .delete_conversation(
                tenant_id,
                conversation_id,
            )
        )

        if deleted == 0:
            raise ConversationNotFoundError(
                "Không tìm thấy conversation."
            )

        return Response(
            status_code=204
        )

    # --------------------------------------------------------
    # ASK ENDPOINT
    # --------------------------------------------------------

    @application.post(
        "/v1/ask",
        response_model=(
            ConversationQuestionResponse
        ),
    )
    async def ask_question(
        payload: ConversationQuestionRequest,
        request: Request,
    ) -> ConversationQuestionResponse:
        """
        Endpoint hỏi đáp có lịch sử.

        Semaphore bảo vệ CPU/GPU khỏi quá nhiều câu hỏi chạy cùng lúc.
        """

        container = _get_container(
            request
        )

        queue_timeout = float(
            getattr(
                container.settings,
                "api_queue_wait_seconds",
                0.1,
            )
        )

        try:
            await asyncio.wait_for(
                container
                .ask_semaphore
                .acquire(),
                timeout=queue_timeout,
            )
        except TimeoutError as exception:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Hệ thống đang xử lý quá nhiều câu hỏi. "
                    "Vui lòng thử lại sau."
                ),
            ) from exception

        try:
            return await container.conversation_service.answer_question(
                payload
            )
        finally:
            container.ask_semaphore.release()

    return application


app = create_application()
