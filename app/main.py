from __future__ import annotations

"""
FastAPI entry point.

Chạy bằng:
    uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.clients.ollama_client import OllamaClient
from app.clients.qdrant_repository import QdrantRepository
from app.config import Settings, get_settings
from app.generation.answer_generator import RagAnswerGenerator
from app.generation.context_builder import AnswerContextBuilder
from app.logging_setup import configure_logging
from app.retrieval.reranker import CrossEncoderDocumentReranker
from app.retrieval.retriever import DenseDocumentRetriever
from app.schemas import (
    HealthResponse,
    QuestionRequest,
    QuestionResponse,
)
from app.services.question_answering_service import (
    QuestionAnsweringService,
)
from app.sql.intent_router import QuestionIntentRouter
from app.sql.query_registry import PredefinedSqlQueryRegistry
from app.sql.sql_service import SafeSqlServerService


@asynccontextmanager
async def application_lifespan(
    fastapi_application: FastAPI,
) -> AsyncIterator[None]:
    """
    Tạo dependency dùng chung khi process khởi động và đóng khi shutdown.
    """

    settings = get_settings()
    configure_logging(settings.log_level)

    ollama_client = OllamaClient(settings)
    qdrant_repository = QdrantRepository(settings)
    query_registry = PredefinedSqlQueryRegistry()
    sql_server_service = SafeSqlServerService(
        settings=settings,
        query_registry=query_registry,
    )
    retriever = DenseDocumentRetriever(
        settings=settings,
        ollama_client=ollama_client,
        qdrant_repository=qdrant_repository,
    )
    reranker = CrossEncoderDocumentReranker(settings)
    intent_router = QuestionIntentRouter(
        ollama_client=ollama_client,
        query_registry=query_registry,
        sql_server_service=sql_server_service,
    )
    context_builder = AnswerContextBuilder(settings)
    answer_generator = RagAnswerGenerator(ollama_client)
    question_answering_service = QuestionAnsweringService(
        settings=settings,
        retriever=retriever,
        reranker=reranker,
        intent_router=intent_router,
        sql_server_service=sql_server_service,
        context_builder=context_builder,
        answer_generator=answer_generator,
    )

    fastapi_application.state.settings = settings
    fastapi_application.state.ollama_client = ollama_client
    fastapi_application.state.qdrant_repository = qdrant_repository
    fastapi_application.state.query_registry = query_registry
    fastapi_application.state.sql_server_service = sql_server_service
    fastapi_application.state.question_answering_service = (
        question_answering_service
    )

    try:
        # Tạo collection sớm để lỗi Qdrant xuất hiện ngay khi khởi động.
        await qdrant_repository.ensure_collection()
        yield
    finally:
        await sql_server_service.close()
        await qdrant_repository.close()
        await ollama_client.close()


settings_for_application = get_settings()

app = FastAPI(
    title=settings_for_application.application_name,
    version="1.0.0",
    lifespan=application_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=(
        settings_for_application.get_cors_allowed_origins()
    ),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """
    Health endpoint đơn giản.
    """

    settings: Settings = request.app.state.settings

    return HealthResponse(
        status="ok",
        application=settings.application_name,
        qdrant_configured=bool(settings.qdrant_url),
        sql_server_enabled=(
            request.app.state.sql_server_service.is_enabled()
        ),
        reranker_enabled=settings.reranker_enabled,
    )


@app.get("/v1/sql/query-catalog")
async def list_predefined_sql_queries(
    request: Request,
) -> dict[str, object]:
    """
    Hiển thị catalog query để frontend biết query key và parameter.
    """

    return {
        "queries": request.app.state.query_registry.list_for_router()
    }


@app.post("/v1/ask", response_model=QuestionResponse)
async def ask_question(
    payload: QuestionRequest,
    request: Request,
) -> QuestionResponse:
    """
    Endpoint hỏi đáp chính.
    """

    service: QuestionAnsweringService = (
        request.app.state.question_answering_service
    )

    try:
        return await service.answer_question(payload)
    except (ValueError, KeyError) as exception:
        raise HTTPException(
            status_code=400,
            detail=str(exception),
        ) from exception
    except RuntimeError as exception:
        raise HTTPException(
            status_code=503,
            detail=str(exception),
        ) from exception
