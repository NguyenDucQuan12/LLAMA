from __future__ import annotations

"""
Repository quản lý toàn bộ thao tác với Qdrant.

Tầng dịch vụ không nên gọi Qdrant client trực tiếp ở nhiều nơi vì điều đó
làm phân tán logic collection, payload index, filter và version hóa.
"""

import logging
from collections.abc import Sequence
from typing import Any
import asyncio

from qdrant_client import AsyncQdrantClient, models

from config import Settings, get_settings


logger = logging.getLogger(__name__)


class QdrantRepository:
    """
    Cung cấp API cấp ứng dụng cho collection vector.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.client = self._create_qdrant_client()

    def _create_qdrant_client( self, use_local_mode: bool = True,) -> AsyncQdrantClient:
        """
        Tạo Qdrant client.

        use_local_mode=True:
            Không cần Qdrant server.
            Vector được lưu vào thư mục trên máy.

        use_local_mode=False:
            Kết nối đến Qdrant server qua HTTP.
        """
        # Ngăn nhiều coroutine trong cùng process đồng thời tạo
        # collection và payload indexes.
        self._collection_lock = asyncio.Lock()
        self.collection_name = self.settings.qdrant_collection_name
        self.vector_dimensions = self.settings.embedding_vector_dimensions

        if use_local_mode:
            return AsyncQdrantClient(
                # Thư mục chứa dữ liệu local Qdrant.
                path="./qdrant_local_storage",
            )

        return AsyncQdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            timeout=60
        )

    async def close(self) -> None:
        """
        Đóng Qdrant client.
        """

        await self.client.close()

    async def ensure_collection(self) -> None:
        """
        Tạo collection và payload index khi chưa tồn tại.

        Không tự động xóa collection nếu dimension sai. Trong production,
        đổi embedding model phải tạo collection phiên bản mới để có thể
        rollback an toàn.  
        Lock ngăn hai coroutine trong cùng process chạy bước tạo đồng thời.
        Nếu có race giữa nhiều process, exception tạo collection chỉ được
        bỏ qua khi kiểm tra lại cho thấy collection thực sự đã tồn tại.
        """
        async with self._collection_lock:
            # Kiểm tra xem collection có tên này đã tồn tại hay chưa
            exists = await self.client.collection_exists(collection_name=self.collection_name)

            if not exists:
                # Nếu chưa tồn tại thì tiến hành tạo collection mới theo tên đã chọn, với số chiều cố định
                try:
                    await self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(
                            size=self.vector_dimensions,
                            distance=models.Distance.COSINE,
                        ),
                    )
                except Exception:
                    # Có thể process khác vừa tạo collection.
                    exists_after_error = (
                        await self.client.collection_exists(collection_name=self.collection_name)
                    )

                    if not exists_after_error:
                        raise

                    logger.info("Collection `%s` đã được process khác tạo.",self.collection_name,)

            # Lấy thông tin collection
            info = await self.client.get_collection(collection_name=self.collection_name)

            # so sánh vector có khớp không
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, dict):
                raise RuntimeError(
                    "Collection đang dùng named vectors nhưng "
                    "repository đang dùng single vector."
                )

            # Lấy thông tin số chiều của vector
            actual_dimension = getattr(vectors_config, "size", None)

            if actual_dimension is None:
                raise RuntimeError("Không đọc được dimension collection.")
            # NẾu số chiều vector khác nhau thì không hợp lệ
            if int(actual_dimension) != self.vector_dimensions:
                raise RuntimeError(
                    "Dimension collection không khớp: "
                    f"collection={actual_dimension}, "
                    f"settings={self.vector_dimensions}."
                )

            await self._ensure_payload_indexes(info)

    async def upsert_points(self, points: Sequence[models.PointStruct]) -> None:
        """
        Upsert một batch point và chờ Qdrant xác nhận hoàn tất.
        """

        if not points:
            return
        
        seen_ids: set[str] = set()

        # Xác thực point id có trùng lặp không
        for point_index, point in enumerate(points):
            point_id = str(point.id)

            if point_id in seen_ids:
                raise ValueError(f"Point ID trùng trong batch: {point_id}")

            seen_ids.add(point_id)

            vector = point.vector
            if not isinstance(vector, list):
                raise TypeError(f"Point {point_index}: vector phải là list.")

            # Kiểm tra số chiều vector, số chiều bằng số lượng phần tử trong nó
            if len(vector) != self.vector_dimensions:
                raise ValueError(
                    f"Point {point_index}: dimension "
                    f"{len(vector)} != {self.vector_dimensions}."
                )

            for value in vector:
                if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))):
                    raise ValueError(f"Point {point_index}: vector không hợp lệ.")

            payload = point.payload
            if not isinstance(payload, dict):
                raise TypeError(f"Point {point_index}: payload phải là dict.")
            
            # Các trường bắt buộc phải có
            required_fields = {"tenant_id", "document_id", "source_hash"}
            # So sánh các trường thiếu
            missing = required_fields - set(payload)
            if missing:
                raise ValueError(f"Point {point_index} thiếu payload: {sorted(missing)}")

        # Tiến hành upsert 1 batch point vào adrant
        await self.client.upsert(
            collection_name=self.collection_name,
            points=list(points),
            wait=True,
        )

    async def search_chunks( self, query_vector: list[float], tenant_id: str, 
                            top_k: int, document_id: str | None = None, ) -> list[models.ScoredPoint]:
        """
        Tìm các chunk gần nhất với query vector.

        `with_payload=True` là bắt buộc vì Llama cần text và metadata.  
        Hoặc có thể chỉ định các trường cụ thể muốn lấy trong payload. Ví dụ:  
        ```python
        with_payload=[
            "document_id",
            "source_file",
            "source_hash",
            "chunk_index",
            "headings",
            "page_numbers",
            "doc_item_refs",
            "text",
            "contextualized_text",
        ]
        ```
        Vector của point không cần trả về nên `with_vectors=False` để giảm
        dữ liệu truyền qua mạng.
        """
        # Tạo điều kiện để lọc dữ liệu tốt hơn
        filter_conditions: list[models.Condition] = [
            # Là điều kiện bắt buộc để phân tách dữ liệu của các tenant khác nhau
            models.FieldCondition(
                key="tenant_id",                           # Tìm trường tenant_id trong CSDL
                match=models.MatchValue(value=tenant_id),  # Lọc lấy giá trị chính xác bằng giá trị biến này truyền vào
            ),
            models.FieldCondition(
                key="embedding_model",
                match=models.MatchValue( value=self.settings.embedding_model_name), # Lấy đúng giá trị model này để tránh lúc embeding 1 model, và search 1 model khác
            ),
        ]

        if document_id is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            )

        query_result = await self.client.query_points(
            collection_name=self.settings.qdrant_collection_name,
            query=query_vector,
            query_filter=models.Filter(must=filter_conditions),
            with_payload=True,
            with_vectors=False,
            limit=top_k,
        )

        return list(query_result.points)

    async def count_document_version( self, tenant_id: str, document_id: str, source_hash: str, ) -> int:
        """
        Đếm chính xác số point của một phiên bản tài liệu.
        """

        count_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id),
                ),
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                ),
                models.FieldCondition(
                    key="source_hash",
                    match=models.MatchValue(value=source_hash),
                ),
            ]
        )

        count_result = await self.client.count(
            collection_name=self.settings.qdrant_collection_name,
            count_filter=count_filter,
            exact=True,
        )

        return int(count_result.count)
    
    async def delete_document_version(self, tenant_id: str, document_id: str, source_hash: str, verify_after_delete: bool = True) -> int:
        """
        Xóa đúng một version theo ba khóa:

            tenant_id
            AND document_id
            AND source_hash

        Trả về số point tồn tại trước khi xóa.

        Quy trình:
        1. Tạo filter dùng chung.
        2. Count trước.
        3. Nếu 0 thì kết thúc an toàn.
        4. Delete bằng FilterSelector.
        5. wait=True để chờ thao tác.
        6. Count lại và yêu cầu bằng 0.
        """

        point_filter = self._document_version_filter(tenant_id, document_id, source_hash)

        count_before = await self._count_filter(point_filter)

        if count_before == 0:
            logger.info("Không có point cần xóa: tenant=%s document=%s hash=%s", tenant_id, document_id, source_hash)
            return 0

        logger.warning(
            "Xóa document version: tenant=%s document=%s "
            "hash=%s points=%s",
            tenant_id,
            document_id,
            source_hash,
            count_before,
        )

        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=point_filter),
            wait=True,
        )

        if verify_after_delete:
            count_after = await self._count_filter(point_filter)

            if count_after != 0:
                raise RuntimeError(
                    "Xóa chưa hoàn tất: "
                    f"before={count_before}, after={count_after}."
                )

        return count_before

    async def delete_older_document_versions(self, tenant_id: str, document_id: str, current_source_hash: str) -> None:
        """
        Xóa các point phiên bản cũ sau khi phiên bản mới đã ingest đủ.
        """

        old_versions_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id),
                ),
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                ),
            ],
            must_not=[
                models.FieldCondition(
                    key="source_hash",
                    match=models.MatchValue(value=current_source_hash),
                )
            ],
        )

        await self.client.delete(
            collection_name=self.settings.qdrant_collection_name,
            points_selector=models.FilterSelector(
                filter=old_versions_filter
            ),
            wait=True,
        )

    async def count_index_version(
        self,
        tenant_id: str,
        document_id: str,
        index_version: str,
    ) -> int:
        """
        Đếm bằng index_version.

        index_version nên bao gồm:
        - source_hash;
        - chunk schema;
        - embedding model;
        - dimensions.
        """

        point_filter = models.Filter(
            must=[
                self._match(
                    "tenant_id",
                    self._normalize_identifier(
                        tenant_id,
                        "tenant_id",
                    ),
                ),
                self._match(
                    "document_id",
                    self._normalize_identifier(
                        document_id,
                        "document_id",
                    ),
                ),
                self._match(
                    "index_version",
                    self._normalize_identifier(
                        index_version,
                        "index_version",
                    ),
                ),
            ]
        )

        return await self._count_filter(point_filter)

    async def delete_other_index_versions(self, tenant_id: str, document_id: str, current_index_version: str) -> int:
        """
        Cải tiến so với source_hash:
        xóa index cũ khi model hoặc chunk schema thay đổi.
        """

        tenant_id = self._normalize_identifier(
            tenant_id,
            "tenant_id",
        )
        document_id = self._normalize_identifier(
            document_id,
            "document_id",
        )
        current_index_version = self._normalize_identifier(
            current_index_version,
            "current_index_version",
        )

        old_filter = models.Filter(
            must=[
                self._match(
                    "tenant_id",
                    tenant_id,
                ),
                self._match(
                    "document_id",
                    document_id,
                ),
            ],
            must_not=[
                self._match(
                    "index_version",
                    current_index_version,
                )
            ],
        )

        count_before = await self._count_filter(
            old_filter
        )

        if count_before == 0:
            return 0

        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=old_filter
            ),
            wait=True,
        )

        count_after = await self._count_filter(
            old_filter
        )

        if count_after != 0:
            raise RuntimeError(
                "Vẫn còn index version cũ: "
                f"before={count_before}, after={count_after}."
            )

        return count_before

    async def _ensure_payload_indexes(self, collection_info: Any) -> None:
        """
        Tạo index cho các trường thường xuyên xuất hiện trong filter.  
        Payload index giúp các filter tenant/document/version hiệu quả hơn.

        Không tạo index cho `text` hoặc `contextualized_text` vì hai trường
        này không được dùng làm keyword filter trong pipeline hiện tại.
        """

        payload_schema = getattr(
            collection_info,
            "payload_schema",
            {},
        )

        existing_fields = (
            set(payload_schema)
            if isinstance(payload_schema, dict)
            else set()
        )

        fields = (
            "tenant_id",
            "document_id",
            "source_hash",
            "index_version",
            "embedding_model",
        )

        for field_name in fields:
            if field_name in existing_fields:
                continue

            try:
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=(
                        models.PayloadSchemaType.KEYWORD
                    ),
                    wait=True,
                )
            except Exception:
                # Process khác có thể vừa tạo cùng index.
                refreshed_info = await self.client.get_collection(
                    collection_name=self.collection_name
                )
                refreshed_schema = getattr(
                    refreshed_info,
                    "payload_schema",
                    {},
                )

                if (
                    not isinstance(refreshed_schema, dict)
                    or field_name not in refreshed_schema
                ):
                    raise

                logger.info(
                    "Payload index `%s` đã được process khác tạo.",
                    field_name,
                )
    
    def _document_version_filter(self, tenant_id: str, document_id: str, source_hash: str) -> models.Filter:
        """
        Dùng cùng filter cho count và delete để tránh lệch điều kiện.
        """

        tenant_id = self._normalize_identifier(
            tenant_id,
            "tenant_id",
        )
        document_id = self._normalize_identifier(
            document_id,
            "document_id",
        )
        source_hash = self._normalize_identifier(
            source_hash,
            "source_hash",
        )

        return models.Filter(
            must=[
                self._match(
                    "tenant_id",
                    tenant_id,
                ),
                self._match(
                    "document_id",
                    document_id,
                ),
                self._match(
                    "source_hash",
                    source_hash,
                ),
            ]
        )
    
    def _match(self, key: str, value: str) -> models.FieldCondition:
        """
        Tạo payload MatchValue condition.
        """

        return models.FieldCondition(
            key=key,
            match=models.MatchValue(
                value=value
            ),
        )

    def _normalize_identifier(self, value: str, field_name: str) -> str:
        """
        Chặn None, chuỗi rỗng và identifier quá dài.
        """

        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} phải là string."
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                f"{field_name} không được rỗng."
            )

        if len(normalized) > 500:
            raise ValueError(
                f"{field_name} tối đa 500 ký tự."
            )

        return normalized
    
    def _positive_int(self, value: Any, field_name: str) -> int:
        """
        bool bị loại vì bool là subclass của int trong Python.
        """

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{field_name} phải là int."
            )

        if value <= 0:
            raise ValueError(
                f"{field_name} phải lớn hơn 0."
            )

        return value
    
    async def _count_filter(self, point_filter: models.Filter) -> int:
        """
        Hàm count dùng chung.
        """

        result = await self.client.count(
            collection_name=self.collection_name,
            count_filter=point_filter,
            exact=True,
        )

        return int(result.count)
    
if __name__ == "__main__":
    # ============================================================
    # TEST XÓA DỮ LIỆU THẬT
    # ============================================================

    async def test_real_delete(
        tenant_id: str,
        document_id: str,
        source_hash: str,
        confirmed: bool,
    ) -> None:
        """
        Chỉ xóa khi có --confirm-delete.
        """

        if not confirmed:
            raise RuntimeError(
                "Thiếu --confirm-delete. "
                "Không thực hiện xóa dữ liệu thật."
            )

        repository = QdrantRepository(
            get_settings()
        )

        try:
            await repository.ensure_collection()

            before = (
                await repository.count_document_version(
                    tenant_id,
                    document_id,
                    source_hash,
                )
            )

            deleted = (
                await repository.delete_document_version(
                    tenant_id,
                    document_id,
                    source_hash,
                )
            )

            after = (
                await repository.count_document_version(
                    tenant_id,
                    document_id,
                    source_hash,
                )
            )

            print(
                json.dumps(
                    {
                        "before": before,
                        "deleted": deleted,
                        "after": after,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

        finally:
            await repository.close()

    async def main():
        await test_real_delete(
            tenant_id="tenant_id",
            document_id="document_id",
            source_hash="source_hash",
            confirmed=True,
        )
    
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print(
            "Đã dừng bởi người dùng.",
            file=sys.stderr,
        )
        raise SystemExit(130)

    except Exception as exception:
        logger.exception(
            "Test QdrantRepository thất bại."
        )
        print(
            f"\nLỖI: {exception}",
            file=sys.stderr,
        )
        raise SystemExit(1)
