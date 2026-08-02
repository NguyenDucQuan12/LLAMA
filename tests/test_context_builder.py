from __future__ import annotations

"""
Kiểm tra context builder gắn nhãn nguồn và giữ giới hạn.
"""

from app.config import Settings
from app.generation.context_builder import AnswerContextBuilder
from app.retrieval.models import RerankedChunk


def test_document_sources_receive_d_labels() -> None:
    settings = Settings(
        answer_max_context_characters=10000,
        answer_max_characters_per_document=5000,
    )
    context_builder = AnswerContextBuilder(settings)

    chunks = [
        RerankedChunk(
            point_id="point-1",
            dense_score=0.8,
            reranker_score=0.9,
            final_score=0.885,
            document_id="guide",
            source_file="guide.docx",
            chunk_index=1,
            headings=["Nhận hàng", "Gọi Robot"],
            page_numbers=[],
            doc_item_refs=["#/texts/10"],
            text="Nhấn gọi robot.",
            contextualized_text=(
                "Nhận hàng\nGọi Robot\nNhấn gọi robot."
            ),
            payload={},
        )
    ]

    sources = context_builder.build_sources(
        document_chunks=chunks,
        sql_result=None,
    )

    assert len(sources) == 1
    assert sources[0].source_label == "D1"
    assert sources[0].source_type == "document"

    context_text = context_builder.build_context_text(sources)

    assert '<source label="D1" type="document">' in context_text
    assert "Nhấn gọi robot" in context_text
