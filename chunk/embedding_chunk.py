from __future__ import annotations

"""
Pipeline embedding production mẫu cho:

    Docling chunks JSONL
        -> Ollama /api/embed
        -> Qdrant

Cài đặt:

    pip install httpx qdrant-client

Ví dụ ingest:

    python production_embedding_pipeline.py ingest \
        --chunks ./outputs/farbic_warehouse_document.chunks.jsonl

Ví dụ truy vấn:

    python production_embedding_pipeline.py search \
        --question "Làm thế nào để gọi robot nhận hàng?"
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import httpx
from qdrant_client import AsyncQdrantClient, models


# ============================================================
# 1. CẤU HÌNH
# ============================================================

@dataclass(frozen=True)
class Settings:
    """
    Toàn bộ cấu hình quan trọng phải được khóa theo từng phiên bản index.

    Không thay đổi model hoặc dimension trong cùng một collection.
    Khi đổi model/dimension/chunking, hãy tạo collection phiên bản mới.
    """

    # URL Ollama.
    ollama_url: str = os.getenv( "OLLAMA_URL", "http://localhost:11434",)

    # Tên model đúng như hiển thị trong `ollama list`.
    embedding_model: str = os.getenv( "EMBEDDING_MODEL", "nomic-embed-text-v2-moe:latest",)

    # Nomic v2 mặc định trả vector 768 chiều.
    # Nếu chủ động dùng Matryoshka 256 chiều thì đổi cả hai:
    # - embedding_dimensions = 256
    # - collection mới có size = 256
    embedding_dimensions: int = int( os.getenv("EMBEDDING_DIMENSIONS", "768"))

    # Đặt tên collection có phiên bản rõ ràng.
    # Khi đổi model/chunking, tạo collection mới thay vì ghi đè.
    collection_name: str = os.getenv( "QDRANT_COLLECTION", "wms_chunks_nomic_v2_768_v1",)

    qdrant_url: str = os.getenv("QDRANT_URL","http://localhost:6333",)

    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")

    # Số văn bản tối đa gửi trong một request embedding.
    # Máy CPU yếu có thể bắt đầu 4-8.
    # GPU có thể thử 16-64 rồi benchmark.
    embedding_batch_size: int = int( os.getenv("EMBEDDING_BATCH_SIZE", "16"))

    # Ngoài giới hạn theo số item, còn giới hạn tổng token của batch.
    # Điều này tránh một batch gồm quá nhiều chunk gần 512 token.
    embedding_batch_token_budget: int = int( os.getenv("EMBEDDING_BATCH_TOKEN_BUDGET", "6000"))

    # Qdrant có thể upsert nhiều point hơn mỗi request.
    # Đây là bộ đệm point trước khi gửi sang Qdrant.
    qdrant_upsert_batch_size: int = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "128"))

    request_timeout_seconds: float = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "180"))

    max_retries: int = int(os.getenv("EMBEDDING_MAX_RETRIES", "4"))

    # Giữ model trong RAM/VRAM để các batch sau không phải load lại.
    keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m",)

    # Phiên bản schema payload do ứng dụng quản lý.
    payload_schema_version: str = "wms-chunk-payload-v1"

    # Phiên bản logic chunking.
    # Đổi giá trị này khi thay đổi cách chunk/contextualize.
    chunk_schema_version: str = "docling-hybrid-v1"


SETTINGS = Settings()


# ============================================================
# 2. ĐỌC VÀ KIỂM TRA CHUNK JSONL
# ============================================================

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Đọc file chunks.jsonl.

    Mỗi dòng phải là một JSON object.
    """

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL lỗi tại dòng {line_number}: {exc}"
                ) from exc

            validate_chunk_record(
                record=record,
                line_number=line_number,
            )

            records.append(record)

    if not records:
        raise ValueError("File JSONL không có chunk hợp lệ.")

    return records


def validate_chunk_record(
    record: dict[str, Any],
    line_number: int,
) -> None:
    """
    Kiểm tra những trường tối thiểu trước khi embedding.

    Quan trọng:
    - Chỉ embedding `embedding_text`.
    - Không embedding toàn bộ JSON.
    """

    required_fields = {
        "id",
        "chunk_index",
        "source_file",
        "source_hash",
        "text",
        "contextualized_text",
        "embedding_text",
        "embedding_token_count",
        "embedding_model_max_tokens",
    }

    missing = required_fields - set(record)

    if missing:
        raise ValueError(
            f"Dòng {line_number} thiếu trường: {sorted(missing)}"
        )

    embedding_text = record["embedding_text"]

    if not isinstance(embedding_text, str):
        raise TypeError(
            f"Dòng {line_number}: embedding_text phải là string."
        )

    embedding_text = embedding_text.strip()

    if not embedding_text:
        raise ValueError(
            f"Dòng {line_number}: embedding_text rỗng."
        )

    # Nomic yêu cầu document dùng prefix search_document:.
    if not embedding_text.startswith("search_document: "):
        raise ValueError(
            f"Dòng {line_number}: embedding_text chưa có "
            "'search_document: ' hoặc prefix bị thêm sai."
        )

    token_count = int(record["embedding_token_count"])
    max_tokens = int(record["embedding_model_max_tokens"])

    if token_count <= 0:
        raise ValueError(
            f"Dòng {line_number}: embedding_token_count không hợp lệ."
        )

    if token_count > max_tokens:
        raise ValueError(
            f"Dòng {line_number}: chunk có {token_count} token, "
            f"vượt giới hạn {max_tokens}."
        )


# ============================================================
# 3. CHIA BATCH THEO SỐ ITEM VÀ TỔNG TOKEN
# ============================================================

def make_embedding_batches(
    records: Sequence[dict[str, Any]],
    max_items: int,
    max_total_tokens: int,
) -> Iterator[list[dict[str, Any]]]:
    """
    Chia batch theo hai điều kiện:

    1. Không vượt quá max_items.
    2. Tổng embedding_token_count không vượt max_total_tokens.

    Ví dụ:
    - batch_size = 16
    - token_budget = 6000

    Nếu 16 chunk đều gần 500 token, chương trình sẽ dừng batch
    sớm hơn để tránh request quá nặng.
    """

    batch: list[dict[str, Any]] = []
    batch_tokens = 0

    for record in records:
        token_count = int(record["embedding_token_count"])

        would_exceed_items = len(batch) >= max_items
        would_exceed_tokens = (
            batch
            and batch_tokens + token_count > max_total_tokens
        )

        if would_exceed_items or would_exceed_tokens:
            yield batch
            batch = []
            batch_tokens = 0

        batch.append(record)
        batch_tokens += token_count

    if batch:
        yield batch


from qdrant_client import AsyncQdrantClient


def create_qdrant_client(
    use_local_mode: bool,
) -> AsyncQdrantClient:
    """
    Tạo Qdrant client.

    use_local_mode=True:
        Không cần Qdrant server.
        Vector được lưu vào thư mục trên máy.

    use_local_mode=False:
        Kết nối đến Qdrant server qua HTTP.
    """

    if use_local_mode:
        return AsyncQdrantClient(
            # Thư mục chứa dữ liệu local Qdrant.
            path="./qdrant_local_storage",
        )

    return AsyncQdrantClient(
        url="http://localhost:6333",
        timeout=60,
    )

# ============================================================
# 4. CLIENT EMBEDDING OLLAMA
# ============================================================

class OllamaEmbeddingClient:
    """
    Client gọi endpoint POST /api/embed của Ollama.

    Các đặc điểm production:
    - Hỗ trợ batch.
    - truncate=False để không âm thầm cắt mất nội dung.
    - Retry với exponential backoff.
    - Kiểm tra số vector, dimension, NaN/Inf và norm.
    """

    RETRYABLE_STATUS_CODES = {
        408,
        429,
        500,
        502,
        503,
        504,
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_url.rstrip("/"),
            timeout=httpx.Timeout(
                settings.request_timeout_seconds
            ),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Embedding một batch văn bản.

        `texts` phải là embedding_text đã có:
            search_document: ...

        Khi truy vấn, caller truyền:
            search_query: ...
        """

        if not texts:
            return []

        request_body: dict[str, Any] = {
            "model": self.settings.embedding_model,
            "input": list(texts),

            # Rất quan trọng:
            # Nếu đầu vào vượt context thì trả lỗi, không cắt âm thầm.
            "truncate": False,

            # Giữ model nóng trong RAM/VRAM.
            "keep_alive": self.settings.keep_alive,
        }

        # Endpoint Ollama hỗ trợ yêu cầu số chiều.
        # Với Nomic v2, 768 là đầy đủ; 256 là tùy chọn tiết kiệm.
        if self.settings.embedding_dimensions:
            request_body["dimensions"] = (
                self.settings.embedding_dimensions
            )

        last_error: Exception | None = None

        for attempt in range(self.settings.max_retries + 1):
            try:
                response = await self._client.post(
                    "/api/embed",
                    json=request_body,
                )

                if ( response.status_code in self.RETRYABLE_STATUS_CODES):
                    raise httpx.HTTPStatusError(
                        message=(
                            "Lỗi tạm thời từ embedding service: "
                            f"HTTP {response.status_code}"
                        ),
                        request=response.request,
                        response=response,
                    )

                # Các lỗi 400 như model không tồn tại hoặc input quá dài
                # phải dừng ngay, không retry vô ích.
                response.raise_for_status()

                data = response.json()
                vectors = data.get("embeddings")

                if not isinstance(vectors, list):
                    raise RuntimeError(
                        "Ollama không trả trường embeddings hợp lệ."
                    )

                self._validate_vectors(
                    vectors=vectors,
                    expected_count=len(texts),
                )

                return vectors

            except (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.HTTPStatusError,
            ) as exc:
                last_error = exc

                # Không retry lỗi HTTP không thuộc nhóm tạm thời.
                if isinstance(exc, httpx.HTTPStatusError):
                    status = exc.response.status_code

                    if status not in self.RETRYABLE_STATUS_CODES:
                        body = exc.response.text[:1000]

                        raise RuntimeError(
                            "Embedding thất bại và không nên retry. "
                            f"HTTP {status}: {body}"
                        ) from exc

                if attempt >= self.settings.max_retries:
                    break

                # 1, 2, 4, 8... giây; tối đa 20 giây.
                delay = min(2**attempt, 20)

                print(
                    f"Embedding tạm lỗi; retry sau {delay}s "
                    f"(lần {attempt + 1}/"
                    f"{self.settings.max_retries}).",
                    file=sys.stderr,
                )

                await asyncio.sleep(delay)

        raise RuntimeError(
            "Embedding thất bại sau toàn bộ số lần retry."
        ) from last_error

    def _validate_vectors(
        self,
        vectors: Any,
        expected_count: int,
    ) -> None:
        """
        Kiểm tra output trước khi lưu vào Qdrant.

        Ollama /api/embed trả vector đã L2-normalized.
        Vì vậy norm dự kiến xấp xỉ 1.
        """

        if len(vectors) != expected_count:
            raise RuntimeError(
                "Số vector không khớp số input: "
                f"{len(vectors)} != {expected_count}"
            )

        for vector_index, vector in enumerate(vectors):
            if not isinstance(vector, list):
                raise TypeError(
                    f"Vector {vector_index} không phải list."
                )

            if len(vector) != self.settings.embedding_dimensions:
                raise RuntimeError(
                    f"Vector {vector_index} có dimension "
                    f"{len(vector)}, expected "
                    f"{self.settings.embedding_dimensions}."
                )

            if not all(
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in vector
            ):
                raise RuntimeError(
                    f"Vector {vector_index} chứa NaN, Inf "
                    "hoặc giá trị không phải số."
                )

            norm = math.sqrt(
                sum(float(value) ** 2 for value in vector)
            )

            if not 0.98 <= norm <= 1.02:
                raise RuntimeError(
                    f"Vector {vector_index} không được "
                    f"L2-normalize như dự kiến; norm={norm:.6f}."
                )


# ============================================================
# 5. QDRANT COLLECTION VÀ PAYLOAD INDEX
# ============================================================

async def ensure_collection(
    client: AsyncQdrantClient,
    settings: Settings,
) -> None:
    """
    Tạo collection nếu chưa có.

    Nếu collection đã có thì kiểm tra dimension.
    Không tự xóa collection production.
    """

    exists = await client.collection_exists(
        collection_name=settings.collection_name
    )

    if not exists:
        await client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimensions,
                distance=models.Distance.COSINE,
            ),
        )

        print(
            "Đã tạo collection:",
            settings.collection_name,
        )

    collection_info = await client.get_collection(
        collection_name=settings.collection_name
    )

    # Tùy phiên bản qdrant-client, vectors config có thể là
    # VectorParams hoặc dictionary named vectors.
    vectors_config = collection_info.config.params.vectors

    actual_size: int | None = None

    if hasattr(vectors_config, "size"):
        actual_size = int(vectors_config.size)

    elif isinstance(vectors_config, dict):
        # Nếu dùng named vector, bạn cần chọn đúng tên vector.
        # Script hiện dùng một vector mặc định nên không mong đợi dict.
        raise RuntimeError(
            "Collection đang dùng named vectors nhưng script "
            "đang cấu hình single vector."
        )

    if actual_size != settings.embedding_dimensions:
        raise RuntimeError(
            "Dimension collection không khớp model: "
            f"{actual_size} != {settings.embedding_dimensions}. "
            "Hãy tạo collection phiên bản mới."
        )

    await ensure_payload_indexes(
        client=client,
        settings=settings,
    )


async def ensure_payload_indexes(
    client: AsyncQdrantClient,
    settings: Settings,
) -> None:
    """
    Chỉ tạo index cho các trường thực sự dùng để lọc.

    Không cần index:
    - text
    - contextualized_text
    - headings nếu chưa dùng filter

    Nên index:
    - document_id
    - source_hash
    - tenant_id
    - embedding_model
    """

    index_specs = [
        (
            "document_id",
            models.PayloadSchemaType.KEYWORD,
        ),
        (
            "source_hash",
            models.PayloadSchemaType.KEYWORD,
        ),
        (
            "tenant_id",
            models.PayloadSchemaType.KEYWORD,
        ),
        (
            "embedding_model",
            models.PayloadSchemaType.KEYWORD,
        ),
    ]

    for field_name, field_schema in index_specs:
        try:
            await client.create_payload_index(
                collection_name=settings.collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )

        except Exception as exc:
            # create_payload_index có thể báo index đã tồn tại.
            # Chỉ bỏ qua đúng trường hợp đó.
            message = str(exc).lower()

            if (
                "already exists" not in message
                and "already indexed" not in message
            ):
                raise


# ============================================================
# 6. TẠO PAYLOAD QDRANT
# ============================================================

def sha256_text(text: str) -> str:
    """
    Hash nội dung thực tế đã embedding.

    Có thể dùng làm cache key:
        model + dimensions + embedding_content_hash
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def build_payload(
    record: dict[str, Any],
    settings: Settings,
    tenant_id: str,
) -> dict[str, Any]:
    """
    Chỉ lưu metadata hữu ích cho retrieval và quản trị.

    Không nên lưu toàn bộ Base64 hoặc toàn bộ DoclingDocument JSON
    vào payload của mỗi point.
    """

    source_file = str(record["source_file"])

    # Trong production, document_id nên là ID nghiệp vụ ổn định
    # từ DB của bạn. Ở ví dụ này dùng tên file làm fallback.
    document_id = str(
        record.get("document_id") or source_file
    )

    embedding_text = str(record["embedding_text"])

    return {
        # Phân vùng dữ liệu.
        "tenant_id": tenant_id,

        # Định danh tài liệu ổn định.
        "document_id": document_id,

        # Phiên bản nội dung file.
        "source_hash": str(record["source_hash"]),

        "source_file": source_file,

        # Liên kết ngược về chunks JSONL/DoclingDocument.
        "chunk_index": int(record["chunk_index"]),
        "docling_chunk_index": int(
            record.get(
                "docling_chunk_index",
                record["chunk_index"],
            )
        ),
        "subchunk_index": int(
            record.get("subchunk_index", 0)
        ),

        # Nguồn và cấu trúc.
        "page_numbers": list(
            record.get("page_numbers", [])
        ),
        "headings": list(
            record.get(
                "headings",
                record.get(
                    "docling_metadata",
                    {},
                ).get("headings", []),
            )
            or []
        ),

        # Text gốc dùng để trích dẫn cho người dùng.
        "text": str(record["text"]),

        # Text có heading; phù hợp đưa vào context của LLM.
        "contextualized_text": str(
            record["contextualized_text"]
        ),

        # Dùng kiểm tra/caching; có thể bỏ embedding_text khỏi
        # payload nếu muốn giảm dung lượng vì đã có
        # contextualized_text.
        "embedding_content_hash": sha256_text(
            embedding_text
        ),

        # Version hóa để biết point được tạo bằng model nào.
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": (
            settings.embedding_dimensions
        ),
        "payload_schema_version": (
            settings.payload_schema_version
        ),
        "chunk_schema_version": (
            settings.chunk_schema_version
        ),
    }


# ============================================================
# 7. INGEST: EMBEDDING + UPSERT QDRANT
# ============================================================

async def ingest_chunks(
    chunks_path: Path,
    settings: Settings,
    tenant_id: str,
    remove_old_versions: bool,
) -> None:
    """
    Ingest an toàn theo thứ tự:

    1. Đọc và validate chunks.
    2. Tạo collection/index nếu cần.
    3. Embedding theo batch.
    4. Kiểm tra vector.
    5. Upsert Qdrant bằng ID ổn định.
    6. Kiểm tra số point phiên bản mới.
    7. Sau khi thành công mới xóa phiên bản cũ.

    Không xóa dữ liệu cũ trước khi phiên bản mới ingest xong.
    """

    records = read_jsonl(chunks_path)

    # Ví dụ này giả định một file JSONL thuộc một document version.
    source_hashes = {
        str(record["source_hash"])
        for record in records
    }

    document_ids = {
        str(
            record.get("document_id")
            or record["source_file"]
        )
        for record in records
    }

    if len(source_hashes) != 1:
        raise ValueError(
            "Một file chunks.jsonl nên chỉ chứa một source_hash."
        )

    if len(document_ids) != 1:
        raise ValueError(
            "Một file chunks.jsonl nên chỉ chứa một document_id."
        )

    source_hash = next(iter(source_hashes))
    document_id = next(iter(document_ids))

    qdrant = create_qdrant_client(
        use_local_mode=True
    )

    embedder = OllamaEmbeddingClient(settings)

    try:
        await ensure_collection(
            client=qdrant,
            settings=settings,
        )

        qdrant_buffer: list[models.PointStruct] = []
        embedded_count = 0

        batches = make_embedding_batches(
            records=records,
            max_items=settings.embedding_batch_size,
            max_total_tokens=(
                settings.embedding_batch_token_budget
            ),
        )

        for batch_number, batch in enumerate(
            batches,
            start=1,
        ):
            texts = [
                str(record["embedding_text"])
                for record in batch
            ]

            vectors = await embedder.embed(texts)

            for record, vector in zip(
                batch,
                vectors,
                strict=True,
            ):
                point = models.PointStruct(
                    # ID UUID ổn định đã được pipeline chunk tạo.
                    # Upsert lại cùng ID không tạo bản trùng.
                    id=str(record["id"]),
                    vector=vector,
                    payload=build_payload(
                        record=record,
                        settings=settings,
                        tenant_id=tenant_id,
                    ),
                )

                qdrant_buffer.append(point)

            embedded_count += len(batch)

            print(
                f"Embedded batch {batch_number}: "
                f"{len(batch)} chunk; "
                f"tổng={embedded_count}/{len(records)}"
            )

            if (
                len(qdrant_buffer)
                >= settings.qdrant_upsert_batch_size
            ):
                await qdrant.upsert(
                    collection_name=settings.collection_name,
                    points=qdrant_buffer,
                    wait=True,
                )

                print(
                    "Upsert Qdrant:",
                    len(qdrant_buffer),
                    "point",
                )

                qdrant_buffer = []

        # Gửi nốt point còn lại.
        if qdrant_buffer:
            await qdrant.upsert(
                collection_name=settings.collection_name,
                points=qdrant_buffer,
                wait=True,
            )

            print(
                "Upsert Qdrant:",
                len(qdrant_buffer),
                "point cuối",
            )

        # Kiểm tra đủ point của phiên bản mới.
        new_version_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(
                        value=tenant_id
                    ),
                ),
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(
                        value=document_id
                    ),
                ),
                models.FieldCondition(
                    key="source_hash",
                    match=models.MatchValue(
                        value=source_hash
                    ),
                ),
            ]
        )

        count_result = await qdrant.count(
            collection_name=settings.collection_name,
            count_filter=new_version_filter,
            exact=True,
        )

        if count_result.count != len(records):
            raise RuntimeError(
                "Số point trong Qdrant không khớp JSONL: "
                f"{count_result.count} != {len(records)}"
            )

        print(
            f"Xác minh thành công: {count_result.count} point."
        )

        # Chỉ xóa version cũ sau khi version mới đã hoàn chỉnh.
        if remove_old_versions:
            old_versions_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(
                            value=tenant_id
                        ),
                    ),
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(
                            value=document_id
                        ),
                    ),
                ],
                must_not=[
                    models.FieldCondition(
                        key="source_hash",
                        match=models.MatchValue(
                            value=source_hash
                        ),
                    ),
                ],
            )

            await qdrant.delete(
                collection_name=settings.collection_name,
                points_selector=models.FilterSelector(
                    filter=old_versions_filter
                ),
                wait=True,
            )

            print(
                "Đã xóa các vector phiên bản cũ của:",
                document_id,
            )

    finally:
        await embedder.close()
        await qdrant.close()


# ============================================================
# 8. TRUY VẤN DENSE VECTOR
# ============================================================

def normalize_query_text(text: str) -> str:
    """
    Chỉ chuẩn hóa Unicode và khoảng trắng.

    Không:
    - lowercase cưỡng bức;
    - xóa dấu tiếng Việt;
    - xóa dấu gạch dưới;
    - sửa mã pallet/procedure.

    Các chuỗi như usp_wms_custom_in_submit hoặc F3-29
    phải được giữ nguyên.
    """

    normalized = unicodedata.normalize(
        "NFC",
        text,
    )

    return " ".join(normalized.split())


async def semantic_search(
    question: str,
    settings: Settings,
    tenant_id: str,
    limit: int,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Query phải dùng prefix search_query:.

    Dùng đúng cùng:
    - embedding model;
    - embedding dimension;
    - collection/vector space;
    - logic prefix.
    """

    question = normalize_query_text(question)

    if not question:
        raise ValueError("Câu hỏi rỗng.")

    query_embedding_text = (
        f"search_query: {question}"
    )

    embedder = OllamaEmbeddingClient(settings)

    qdrant = create_qdrant_client(
        use_local_mode=True
    )

    try:
        query_vector = (
            await embedder.embed(
                [query_embedding_text]
            )
        )[0]

        must_conditions: list[models.Condition] = [
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(
                    value=tenant_id
                ),
            ),
            models.FieldCondition(
                key="embedding_model",
                match=models.MatchValue(
                    value=settings.embedding_model
                ),
            ),
        ]

        if document_id:
            must_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(
                        value=document_id
                    ),
                )
            )

        result = await qdrant.query_points(
            collection_name=settings.collection_name,
            query=query_vector,
            query_filter=models.Filter(
                must=must_conditions
            ),
            with_payload=True,
            with_vectors=False,

            # Retrieval ban đầu thường lấy nhiều hơn số chunk
            # cuối cùng gửi cho LLM để còn rerank/deduplicate.
            limit=limit,
        )

        output: list[dict[str, Any]] = []

        for point in result.points:
            payload = dict(point.payload or {})

            output.append(
                {
                    "id": str(point.id),
                    "score": float(point.score),
                    "source_file": payload.get(
                        "source_file"
                    ),
                    "chunk_index": payload.get(
                        "chunk_index"
                    ),
                    "headings": payload.get(
                        "headings",
                        [],
                    ),
                    "page_numbers": payload.get(
                        "page_numbers",
                        [],
                    ),
                    "text": payload.get("text"),
                    "contextualized_text": payload.get(
                        "contextualized_text"
                    ),
                }
            )

        return output

    finally:
        await embedder.close()
        await qdrant.close()


# ============================================================
# 9. CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Embedding Docling chunks bằng Ollama "
            "và lưu vào Qdrant."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Embedding và ingest chunks JSONL.",
    )

    ingest_parser.add_argument(
        "--chunks",
        type=Path,
        required=True,
        help="Đường dẫn file *.chunks.jsonl",
    )

    ingest_parser.add_argument(
        "--tenant-id",
        default="default",
        help="Tenant dùng để phân vùng dữ liệu.",
    )

    ingest_parser.add_argument(
        "--remove-old-versions",
        action="store_true",
        help=(
            "Sau khi version mới ingest và xác minh thành công, "
            "xóa các vector version cũ của cùng document_id."
        ),
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Semantic search thử nghiệm.",
    )

    search_parser.add_argument(
        "--question",
        required=True,
    )

    search_parser.add_argument(
        "--tenant-id",
        default="default",
    )

    search_parser.add_argument(
        "--document-id",
        default=None,
    )

    search_parser.add_argument(
        "--limit",
        type=int,
        default=20,
    )

    return parser


async def async_main() -> int:
    args = build_parser().parse_args()

    if args.command == "ingest":
        await ingest_chunks(
            chunks_path=args.chunks,
            settings=SETTINGS,
            tenant_id=args.tenant_id,
            remove_old_versions=(
                args.remove_old_versions
            ),
        )

        return 0

    if args.command == "search":
        results = await semantic_search(
            question=args.question,
            settings=SETTINGS,
            tenant_id=args.tenant_id,
            limit=args.limit,
            document_id=args.document_id,
        )

        print(
            json.dumps(
                results,
                ensure_ascii=False,
                indent=2,
            )
        )

        return 0

    return 1


def main() -> int:
    try:
        return asyncio.run(async_main())

    except KeyboardInterrupt:
        print("Đã dừng bởi người dùng.", file=sys.stderr)
        return 130

    except Exception as exc:
        print(
            f"Lỗi: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())