from __future__ import annotations

"""
Các model nội bộ của retrieval và reranking.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RetrievedChunk(BaseModel):
    """
    Một chunk do Qdrant trả về ở bước top 20.
    """

    model_config = ConfigDict(extra="ignore")

    point_id: str
    dense_score: float
    document_id: str | None = None
    source_file: str | None = None
    source_hash: str | None = None
    chunk_index: int | None = None
    headings: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    doc_item_refs: list[str] = Field(default_factory=list)
    text: str
    contextualized_text: str
    payload: dict[str, Any] = Field(default_factory=dict)


class RerankedChunk(RetrievedChunk):
    """
    Một chunk sau khi cross-encoder chấm lại.
    """

    reranker_score: float | None = None
    final_score: float
