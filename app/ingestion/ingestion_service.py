from __future__ import annotations

"""
Điều phối pipeline:

    File -> Docling chunks -> Ollama embeddings -> Qdrant points

Chạy test:

    # Không cần Ollama/Qdrant:
    python3 -m app.ingestion.document_ingestion_service --mode unit

    # Chạy một file thật:
    python3 -m app.ingestion.document_ingestion_service \
        --mode full \
        --source "./document/farbic_warehouse_document.docx" \
        --tenant-id "wms" \
        --document-id "fabric-warehouse-guide"
"""

import argparse
import asyncio
import hashlib
import inspect
import json
import logging
import math
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from qdrant_client import models

# Mở comment 3 dòng bên dưới mỗi khi test (Chạy trực tiếp hàm if __main__)
import os,sys
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from clients.ollama_client import OllamaClient
from clients.qdrant_repository import QdrantRepository
from config import Settings, get_settings
from ingestion.chunking_service import ChunkRecord,ChunkingResult,DocumentChunkingService
from schemas import IngestedDocumentSummary


logger = logging.getLogger(__name__)


class DocumentIngestionService:
    """
    Điều phối chunking, embedding và lưu Qdrant.

    Mỗi lớp chỉ giữ một trách nhiệm:
    - DocumentChunkingService: chuyển file và tạo chunk.
    - OllamaClient: gọi /api/embed.
    - QdrantRepository: giao tiếp với Qdrant.
    - DocumentIngestionService: phối hợp các bước trên.
    """

    def __init__(self, settings: Settings, ollama_client: OllamaClient, qdrant_repository: QdrantRepository, chunking_service: DocumentChunkingService) -> None:
        """
        Lưu các dependency và kiểm tra cấu hình ngay từ đầu.

        Việc kiểm tra sớm tránh trường hợp đã chunk/embedding một phần
        mới phát hiện batch_size bằng 0 hoặc dimension không hợp lệ.
        """

        self.settings = settings
        self.ollama_client = ollama_client
        self.qdrant_repository = qdrant_repository
        self.chunking_service = chunking_service
        # Xác thực các giá trị đầu vào
        self._validate_configuration()

    async def ingest_document(self, source_file: Path, tenant_id: str, document_id: str | None = None, remove_older_versions: bool = True, run_chunking_in_thread: bool = True) -> IngestedDocumentSummary:
        """
        Ingest một tệp và trả về thông tin tổng kết.  
        ----------
        source_file:
            Tệp PDF/DOCX cần ingest.

        tenant_id:
            ID phân vùng dữ liệu, ví dụ "wms".
        document_id:
            ID nghiệp vụ ổn định. Có thể giữ nguyên khi file được cập nhật.
        remove_older_versions:
            Chỉ xóa version cũ sau khi version mới đã được xác minh đầy đủ.
        run_chunking_in_thread:
            True khi gọi trong FastAPI để Docling không chặn event loop.
        """
        # chuẩn hoá tenant id
        normalized_tenant_id = self._normalize_identifier(tenant_id, "tenant_id",)
        # Chuẩn hoá id document
        normalized_document_id = (self._normalize_optional_identifier(document_id, "document_id",))

        # Lấy đường dẫn file cần xử lý và thư mục output
        source_file = Path(source_file)
        output_directory = Path(self.settings.output_directory)

        logger.info("Bắt đầu ingest: source=%s tenant=%s document_id=%s", source_file, normalized_tenant_id, normalized_document_id)

        # Docling là xử lý đồng bộ và có thể mất nhiều thời gian.
        if run_chunking_in_thread:
            chunking_result = await asyncio.to_thread(self.chunking_service.process_document, source_file, output_directory, normalized_document_id)
        else:
            chunking_result = self.chunking_service.process_document(source_file=source_file, output_directory=output_directory, document_id=normalized_document_id)

        if not chunking_result.chunks:
            raise RuntimeError("ChunkingService trả danh sách chunk rỗng.")
        
        # Đếm số lượng chunk được tạo ra
        expected_point_count = len(chunking_result.chunks)
        # Tạo index phiên bản của tài liệu
        index_version = self._build_index_version(
            source_hash=chunking_result.source_hash,
            chunk_schema_version=(
                chunking_result
                .chunks[0]
                .chunk_schema_version
            ),
        )

        # Collection phải tồn tại trước khi upsert.
        await self.qdrant_repository.ensure_collection()

        # Số point tồn tại trước ingest dùng cho rollback best-effort.
        preexisting_count = (
            await self.qdrant_repository
            .count_document_version(
                tenant_id=normalized_tenant_id,
                document_id=(chunking_result.document_id),
                source_hash=(chunking_result.source_hash),
            )
        )

        # Tạo bộ đệm để upsert từng batch vector vào qdrant
        point_buffer: list[models.PointStruct] = []
        embedded_count = 0
        upserted_count = 0

        try:
            # Tạo từng batch để upsert
            batches = self._create_embedding_batches(
                chunks=chunking_result.chunks,
                maximum_items=(self.settings.embedding_batch_size),
                maximum_total_tokens=(self.settings.embedding_batch_total_token_limit),
            )
            
            # Xử lý lần lượt từng batch một
            for batch_number, chunk_batch in enumerate(batches, start=1):
                if not chunk_batch:
                    raise RuntimeError(f"Embedding batch {batch_number} rỗng.")
                # Lấy danh sách embedding text ở trong batch này
                embedding_inputs = [
                    chunk.embedding_text
                    for chunk in chunk_batch
                ]

                # Tạo danh sách vector tương ứng với batch
                vectors = await self.ollama_client.embed_texts(embedding_inputs)
                # Xác thực lại các vector đã được tạo trước khi đưa vào qdrant
                self._validate_embedding_vectors(vectors=vectors, expected_count=len(chunk_batch),)

                # Xử lý từng cặp chunk và vector (strict yêu cầu số lượng phần tử trong chunk_batch và vectors) phải bằng nhau, nếu không sẽ lỗi
                for chunk, vector in zip(chunk_batch, vectors, strict=True):
                    # Tạo PointStruct để lưu vào collection của Qdrant
                    point_buffer.append(
                        models.PointStruct(
                            id=chunk.id,
                            vector=vector,
                            payload=self._build_qdrant_payload(
                                chunk=chunk,
                                tenant_id=normalized_tenant_id,
                                index_version=index_version,
                            ),
                        )
                    )
                # Tính số lượng embedding
                embedded_count += len(chunk_batch)

                logger.info("Embedding batch %s: %s chunk, tổng %s/%s", batch_number, len(chunk_batch), embedded_count, expected_point_count)

                # Gửi đúng từng block qdrant_upsert_batch_size.
                upserted_count += await self._flush_full_qdrant_batches(point_buffer)

            # Gửi phần dư cuối cùng.
            if point_buffer:
                final_count = len(point_buffer)

                await self.qdrant_repository.upsert_points(point_buffer)

                point_buffer.clear()
                upserted_count += final_count

            # Truy vấn lại số point đã được embedding vào qdrant
            actual_point_count = (
                await self.qdrant_repository
                .count_document_version(
                    tenant_id=normalized_tenant_id,
                    document_id=chunking_result.document_id,
                    source_hash=chunking_result.source_hash,
                )
            )
            # Nếu só point được đưa vào collection khác với số point đáng ra được tạo thì thông báo lỗi
            if actual_point_count != expected_point_count:
                raise RuntimeError(
                    "Số point Qdrant không khớp số chunk: "
                    f"actual={actual_point_count}, "
                    f"expected={expected_point_count}, "
                    f"embedded={embedded_count}, "
                    f"upserted={upserted_count}."
                )

            # Chỉ xóa version cũ sau khi version mới hoàn chỉnh.
            #Tránh xoá trước, vì nếu đã xoá trước, sau đó mới embedding, trong quá trình embedding nếu có lỗi thì không còn phiên bản cũ nữa
            if remove_older_versions:
                await (
                    self.qdrant_repository.delete_older_document_versions(
                        tenant_id=normalized_tenant_id,
                        document_id=chunking_result.document_id,
                        current_source_hash=chunking_result.source_hash,
                    )
                )

        except Exception:
            logger.exception(
                "Ingest thất bại: document=%s hash=%s",
                chunking_result.document_id,
                chunking_result.source_hash,
            )

            # Nếu version này chưa tồn tại trước đó, Trong quá trình ghi dữ liệu thì gặp lỗi, một số point đã có, một số point chưa được ghi
            # Thử xóa các poin mới ghi dở. Không che mất exception gốc nếu rollback lỗi.
            if preexisting_count == 0:
                await self._rollback_best_effort(
                    tenant_id=normalized_tenant_id,
                    document_id=chunking_result.document_id,
                    source_hash=chunking_result.source_hash,
                )

            raise
        
        # Trả về thông tin vừa embedding tệp tin
        return IngestedDocumentSummary(
            document_id=chunking_result.document_id,
            source_file=chunking_result.source_file,
            source_hash=chunking_result.source_hash,
            chunk_count=expected_point_count,
            collection_name=self.settings.qdrant_collection_name,
            chunks_jsonl_path=str(chunking_result.chunks_jsonl_path),
            document_json_path=str(chunking_result.document_json_path),
        )

    def _create_embedding_batches(self, chunks: Sequence[ChunkRecord], maximum_items: int, maximum_total_tokens: int) -> Iterator[list[ChunkRecord]]:
        """
        Chia batch theo số item và tổng token.

        Ví dụ:
            token = [200, 250, 300, 100]
            maximum_items = 3
            maximum_total_tokens = 700

        Kết quả:
            [200, 250]
            [300, 100]

        Chunk 300 không được đưa vào batch đầu vì tổng sẽ là 750.
        """

        if maximum_items <= 0:
            raise ValueError("maximum_items phải lớn hơn 0.")

        if maximum_total_tokens <= 0:
            raise ValueError("maximum_total_tokens phải lớn hơn 0.")

        # Tạo danh sách chứa các chunk thành 1 batch
        current_batch: list[ChunkRecord] = []
        current_tokens = 0
        # Xử lý từng chunk 
        for chunk in chunks:
            # Tính token cho chunk này
            token_count = int(chunk.embedding_token_count)

            if token_count <= 0:
                raise ValueError(f"Chunk {chunk.chunk_index} có token_count={token_count}.")

            # Một chunk đơn lẻ không thể lớn hơn ngân sách cả batch.
            if token_count > maximum_total_tokens:
                raise ValueError(
                    f"Chunk {chunk.chunk_index} có "
                    f"{token_count} token, vượt batch token limit "
                    f"{maximum_total_tokens}."
                )

            exceeds_items = len(current_batch) >= maximum_items

            exceeds_tokens = (bool(current_batch) and (current_tokens + token_count > maximum_total_tokens))

            if exceeds_items or exceeds_tokens:
                yield current_batch
                current_batch = []
                current_tokens = 0

            # Thêm chunk này vào batch hiện tại
            current_batch.append(chunk)
            # Tính tổng lại số token
            current_tokens += token_count

        if current_batch:
            yield current_batch

    def _validate_embedding_vectors(self, vectors: Any, expected_count: int) -> None:
        """
        Kiểm tra vector trước khi tạo Qdrant point.

        Đây là defense-in-depth: OllamaClient có thể đã kiểm tra,
        nhưng service vẫn xác minh dữ liệu ở ranh giới module.
        """
        # Kiểm tra kiểu dữ liệu
        if not isinstance(vectors, list):
            raise TypeError("Embedding output phải là list.")
        # Kiểm tra số lượng vecto được tạo ra có khớp với số lượng chunk ban đầu không
        if len(vectors) != expected_count:
            raise RuntimeError(f"Số vector không khớp số input: {len(vectors)} != {expected_count}.")

        # Kiểm tra chi tiết từng vector
        expected_dimension = int(self.settings.embedding_vector_dimensions)
        for vector_index, vector in enumerate(vectors):
            if not isinstance(vector, list):
                raise TypeError(f"Vector {vector_index} không phải list.")
            # Kiểm tra số chiều của vector có khớp không
            if len(vector) != expected_dimension:
                raise RuntimeError(
                    f"Vector {vector_index} có "
                    f"{len(vector)} chiều, cần "
                    f"{expected_dimension}."
                )

            # Kiểm tra từng giá trị vector trong một vector [0.21, -0.34, 0.32, ...]
            squared_norm = 0.0
            for value_index, value in enumerate(vector):
                # bool là subclass của int nhưng không phải số vector.
                if isinstance(value, bool) or not isinstance(value, (int, float),):
                    raise TypeError( f"Vector {vector_index}, vị trí {value_index} không phải số hợp lệ." )
                # Kiểm tra giá trị phải là số, không được Nan hoặc Ìninity 
                number = float(value)
                if not math.isfinite(number):
                    raise RuntimeError(
                        f"Vector {vector_index} chứa NaN/Infinity."
                    )
                # Tính giá trị norm cho vector
                squared_norm += number * number
            # Giá trị norm này được chuẩn hoá về giá trị gần bằng 1, nên nếu nó bé hơn không thì không hợp lý
            if squared_norm <= 0:
                raise RuntimeError(f"Vector {vector_index} là zero vector.")

    async def _flush_full_qdrant_batches(self, point_buffer: list[models.PointStruct]) -> int:
        """
        Gửi các block đủ kích thước, giữ phần dư trong buffer.

        Ví dụ batch size=100, buffer=135:
        - gửi 100;
        - giữ 35 để ghép với batch sau.
        """

        batch_size = int(self.settings.qdrant_upsert_batch_size)
        if batch_size <= 0:
            raise ValueError("qdrant_upsert_batch_size phải > 0.")

        flushed = 0
        while len(point_buffer) >= batch_size:
            points = point_buffer[:batch_size]

            await self.qdrant_repository.upsert_points(points)

            # Chỉ xóa buffer sau khi upsert thành công.
            del point_buffer[:batch_size]
            flushed += len(points)

        return flushed

    def _build_qdrant_payload(self, chunk: ChunkRecord, tenant_id: str, index_version: str) -> dict[str, Any]:
        """
        Tạo metadata lưu cạnh vector.

        Không lưu:
        - ảnh Base64;
        - toàn bộ DoclingDocument;
        - source_path tuyệt đối;
        - embedding_text nếu không cần debug, đã có contextualized và text rồi.
        """

        return {
            "tenant_id": tenant_id,
            "document_id": chunk.document_id,
            "source_file": chunk.source_file,
            "source_hash": chunk.source_hash,

            # Phân biệt cả schema chunk và embedding model.
            "index_version": index_version,

            "chunk_index": chunk.chunk_index,
            "docling_chunk_index": chunk.docling_chunk_index,
            "subchunk_index": chunk.subchunk_index,
            "subchunk_count": chunk.subchunk_count,

            "page_numbers": chunk.page_numbers,
            "page_number_status": chunk.page_number_status,
            "headings": chunk.headings,
            "captions": chunk.captions,
            "doc_item_refs": chunk.doc_item_refs,

            "text": chunk.text,
            "contextualized_text": chunk.contextualized_text,

            # Hash của đúng chuỗi đã embedding, hữu ích cho cache/audit.
            "embedding_content_hash": hashlib.sha256(chunk.embedding_text.encode("utf-8")).hexdigest(),

            "embedding_model": self.settings.embedding_model_name,
            "embedding_dimensions": self.settings.embedding_vector_dimensions,
            "embedding_token_count": chunk.embedding_token_count,
            "chunk_schema_version": chunk.chunk_schema_version,
        }

    def _validate_configuration(self) -> None:
        """
        Kiểm tra các giá trị bắt buộc.
        """

        integer_settings = {
            "embedding_batch_size": (self.settings.embedding_batch_size),
            "embedding_batch_total_token_limit": (self.settings.embedding_batch_total_token_limit),
            "qdrant_upsert_batch_size": (self.settings.qdrant_upsert_batch_size),
            "emedding_vector_dimensions": (self.settings.embedding_vector_dimensions),
        }

        for name, value in integer_settings.items():
            if isinstance(value, bool) or not isinstance(value, int,):
                raise TypeError(f"{name} phải là int.")

            if value <= 0:
                raise ValueError(f"{name} phải lớn hơn 0.")

        if not str(self.settings.qdrant_collection_name).strip():
            raise ValueError(
                "qdrant_collection_name không được rỗng."
            )

    def _normalize_identifier(self, value: str, field_name: str) -> str:
        """
        Chuẩn hóa ID bắt buộc.
        """

        if not isinstance(value, str):
            raise TypeError(f"{field_name} phải là string.")

        normalized = value.strip()

        if not normalized:
            raise ValueError(f"{field_name} không được rỗng.")

        if len(normalized) > 200:
            raise ValueError(f"{field_name} tối đa 200 ký tự.")

        return normalized

    def _normalize_optional_identifier(self, value: str | None, field_name: str) -> str | None:
        """
        Chuẩn hóa ID tùy chọn.
        """

        if value is None:
            return None

        return self._normalize_identifier(
            value,
            field_name,
        )

    def _build_index_version(self, source_hash: str, chunk_schema_version: str) -> str:
        """
        tạo một chuỗi version (SHA‑256 hex) đại diện cho "phiên bản" của vector index  
        Dựa trên các tham số quan trọng — dùng để phát hiện khi nào cần rebuild/reindex.
        """

        identity = {
            "source_hash": source_hash,
            "chunk_schema_version": chunk_schema_version,
            "embedding_model": self.settings.embedding_model_name,
            "embedding_dimensions": self.settings.embedding_vector_dimensions,
        }

        serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))

        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    async def _rollback_best_effort(self, tenant_id: str, document_id: str, source_hash: str) -> None:
        """
        Xóa version mới ghi dở nếu repository hỗ trợ.

        QdrantRepository nên có:

            async def delete_document_version(
                tenant_id: str,
                document_id: str,
                source_hash: str,
            ) -> None
        """

        delete_method = getattr(
            self.qdrant_repository,
            "delete_document_version",
            None,
        )

        if not callable(delete_method):
            logger.warning(
                "Repository chưa có delete_document_version(); "
                "point ghi dở có thể còn lại."
            )
            return

        try:
            result = delete_method(tenant_id=tenant_id, document_id=document_id, source_hash=source_hash)

            if inspect.isawaitable(result):
                await result

        except Exception:
            # Không thay thế exception ingest gốc.
            logger.exception(
                "Rollback Qdrant thất bại."
            )


# ============================================================
# UNIT TEST
# ============================================================


async def _close_if_supported(
    resource: Any,
) -> None:
    close_method = getattr(resource, "close", None)

    if not callable(close_method):
        return

    result = close_method()

    if inspect.isawaitable(result):
        await result


async def test_full_pipeline(
    source_file: Path,
    tenant_id: str,
    document_id: str | None,
    remove_older_versions: bool,
) -> None:
    """
    Test một file thật.
    """

    settings = get_settings()

    ollama_client = OllamaClient(settings)
    qdrant_repository = QdrantRepository(settings)
    chunking_service = DocumentChunkingService(settings)

    service = DocumentIngestionService(
        settings=settings,
        ollama_client=ollama_client,
        qdrant_repository=qdrant_repository,
        chunking_service=chunking_service,
    )

    try:
        summary = await service.ingest_document(
            source_file=source_file,
            tenant_id=tenant_id,
            document_id=document_id,
            remove_older_versions=remove_older_versions,
            run_chunking_in_thread=True,
        )

        print(
            json.dumps(
                {
                    "document_id": summary.document_id,
                    "source_file": summary.source_file,
                    "source_hash": summary.source_hash,
                    "chunk_count": summary.chunk_count,
                    "collection_name": (
                        summary.collection_name
                    ),
                    "chunks_jsonl_path": (
                        summary.chunks_jsonl_path
                    ),
                    "document_json_path": (
                        summary.document_json_path
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    finally:
        await _close_if_supported(ollama_client)
        await _close_if_supported(qdrant_repository)



async def main() -> None:

    await test_full_pipeline(
        source_file=Path("document/farbic_warehouse_document.docx"),
        tenant_id="viva_factory",
        document_id="fabric-warehouse-guide",
        remove_older_versions= True,
    )


if __name__ == "__main__":
    try:
        # Chỉ tạo một event loop cho toàn bộ test.
        asyncio.run(main())

    except KeyboardInterrupt:
        print(
            "Đã dừng bởi người dùng.",
            file=sys.stderr,
        )
        raise SystemExit(130)

    except Exception as exception:
        logger.exception(
            "Document ingestion test thất bại."
        )
        print(
            f"\nLỖI: {exception}",
            file=sys.stderr,
        )
        raise SystemExit(1)