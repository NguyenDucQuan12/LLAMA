from __future__ import annotations

"""
Dịch vụ chunking tài liệu
---------------
1. Chuyển PDF/DOCX/... thành DoclingDocument.
2. Xuất Markdown để con người kiểm tra kết quả chuyển đổi.
3. Xuất DoclingDocument JSON với tiếng Việt dễ đọc.
4. Dùng HybridChunker để tạo chunk theo cấu trúc tài liệu.
5. Giới hạn token bằng tokenizer của embedding model.
6. Kiểm tra lại embedding_text trước khi cho phép ingest.
7. Ghi chunks thành JSONL.
8. Đọc lại JSONL và xác minh tính toàn vẹn
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from docling_core.types.doc import ImageRefMode

# Mở comment 3 dòng bên dưới mỗi khi test (Chạy trực tiếp hàm if __main__)
import os,sys
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from config import Settings, get_settings


logger = logging.getLogger(__name__)


# ============================================================
# DATA CLASS
# ============================================================

@dataclass(frozen=True)
class ChunkRecord:
    """
    Cấu trúc 1 chunk hoàn chỉnh để đưa vào embedding và qdrant  
    Ví dụ:  
    ```python
    {
        "id": "23bac458-...",
        "chunk_index": 21,
        "docling_chunk_index": 21,
        "subchunk_index": 0,
        "subchunk_count": 1,
        "document_id": "fabric-warehouse-guide",
        "source_file": "farbic_warehouse_document.docx",
        "source_hash": "95e32e...",
        "headings": [
            "8 Các lỗi có thể xảy ra"
        ],
        "text": "Nội dung gốc",
        "contextualized_text": "Tiêu đề\nNội dung gốc",
        "embedding_text": "search_document: Tiêu đề\nNội dung gốc",
        "embedding_token_count": 35
    }
    ```
    """

    id: str
    chunk_index: int               # Số thứ tự của chunk
    docling_chunk_index: int       # Số thứ tự của chunk gốc được tạo bới Docling
    subchunk_index: int            # Nếu chunk gốc quá dài, nó sẽ được tách ra và sẽ có giá trị subchunk
    subchunk_count: int

    document_id: str
    source_file: str
    source_path: str
    source_hash: str

    page_numbers: list[int]
    page_number_status: str
    headings: list[str]
    captions: list[str]
    doc_item_refs: list[str]

    text: str
    contextualized_text: str
    embedding_text: str

    embedding_token_count: int
    embedding_model_max_tokens: int
    chunk_content_budget: int
    chunk_schema_version: str

    docling_metadata: dict[str, Any]


@dataclass(frozen=True)
class ChunkingResult:
    """
    Kết quả trả về sau khi xử lý xong một tệp.

    `chunks` được giữ trong RAM để caller có thể:
    - xem trước;
    - embedding ngay;
    - tính thống kê;
    - viết unit test.
    """

    document_id: str
    source_file: str
    source_hash: str

    document_json_path: Path
    markdown_path: Path
    chunks_jsonl_path: Path

    chunks: list[ChunkRecord]


@dataclass(frozen=True)
class ChunkingRuntime:
    """
    Các object nặng được tạo một lần và dùng lại.

    AutoTokenizer và HybridChunker không nên được tải lại cho từng chunk.
    """

    huggingface_tokenizer: PreTrainedTokenizerBase
    docling_tokenizer: HuggingFaceTokenizer
    chunker: HybridChunker
    chunk_content_budget: int                        # Số lượng ký tự tối đa cho chunk


@dataclass(frozen=True)
class ChunkValidationReport:
    """
    Báo cáo kiểm tra chunks sau khi tạo và sau khi đọc lại JSONL.
    """

    chunk_count: int
    minimum_tokens: int
    maximum_tokens: int
    average_tokens: float
    chunks_with_pages: int
    chunks_without_pages: int
    fallback_subchunks: int


# ============================================================
# 2. SERVICE CHUNKING
# ============================================================

class DocumentChunkingService:
    """
    Chuyển một tài liệu thành các chunk an toàn theo token.

    Service này đồng bộ vì Docling và tokenizer chủ yếu là synchronous.
    Trong FastAPI nên gọi qua:
        await asyncio.to_thread(service.process_document, ...)
    hoặc chạy bằng worker queue.
    """

    chunk_schema_version = "docling-hybrid-token-safe-v3"

    def __init__(self,settings: Settings,converter: DocumentConverter | None = None,) -> None:
        """
        settings:
            Cấu hình model/token/chunk.

        converter:
            Cho phép truyền DocumentConverter từ ngoài vào.
            Điều này giúp:
            - tái sử dụng converter;
            - dễ mock trong unit test;
            - dễ cấu hình OCR riêng trong production.
        """

        self.settings = settings

        # Tạo converter một lần thay vì tạo lại trong mỗi process_document().
        self.converter = converter or DocumentConverter()
        # Tokenizer và HybridChunker cũng được tạo một lần.
        self.runtime = self._create_chunking_runtime()

    def process_document(self, source_file: Path, output_directory: Path, document_id: str | None = None,) -> ChunkingResult:
        """
        Chuyển đổi và tạo ch từunk một tệp tin.
        ---------
        1. Chuẩn hóa/kiểm tra đường dẫn.
        2. Tính SHA-256 file nguồn.
        3. Chuyển file thành DoclingDocument.
        4. Xuất Markdown và JSON.
        5. Tạo ChunkRecord.
        6. Kiểm tra tính hợp lệ.
        7. Ghi JSONL theo cách atomic.
        8. Đọc lại JSONL và kiểm tra lần cuối.
        """
        # xử lý đường dẫn file, và thư mục output 
        resolved_source_file = source_file.expanduser().resolve()
        resolved_output_directory = output_directory.expanduser().resolve()

        # Xác thực tệp tin trước khi xử lý
        self._validate_source_file(resolved_source_file)
        # Tạo thư mục output nếu nó chưa tồn tại
        resolved_output_directory.mkdir( parents=True, exist_ok=True, )

        selected_document_id = self._select_document_id( document_id=document_id, source_file=resolved_source_file, )
        # Tính mã hash cho tệp tin này
        source_hash = self._calculate_file_sha256(resolved_source_file)

        logger.info("Bắt đầu Docling conversion: %s",resolved_source_file,)

        # Tiến hành đọc file bằng docling
        try:
            conversion_result = self.converter.convert(resolved_source_file)
            document = conversion_result.document
        except Exception as exception:
            raise RuntimeError(
                "Docling không thể chuyển đổi tài liệu "
                f"`{resolved_source_file}`. "
                "Hãy kiểm tra định dạng file, quyền đọc, "
                "cấu hình OCR và log gốc."
            ) from exception

        # Tạo các thư mục chứa tệp đầu ra
        document_json_path, markdown_path, chunks_jsonl_path, artifacts_directory = self._build_output_paths( source_file=resolved_source_file, output_directory=resolved_output_directory)

        # Markdown giúp kiểm tra trực quan Docling đã đọc đúng nội dung chưa.
        # Trước khi kiểm tra từng chunk thì xem docling có phân tích tốt nội dung document bằng tệp markdown này không
        # Nếu tệp markdown này lỗi thì chắc chắn chunk cũng lỗi theo, còn nếu markdown tốt thì ta phân tích cách tạo chunk
        try:
            document.save_as_markdown( filename=markdown_path, image_mode=ImageRefMode.REFERENCED, artifacts_dir=artifacts_directory, )
        except Exception as exception:
            raise RuntimeError(
                f"Không thể xuất Markdown: {markdown_path}"
            ) from exception

        # JSON dùng PLACEHOLDER để tránh nhúng ảnh Base64 rất lớn.
        try:
            document.save_as_json( filename=document_json_path, image_mode=ImageRefMode.PLACEHOLDER, indent=2, )
            # Sau khi lưu thì têp lỗi ký tự tiếng Việt, ta tiến hành đọc lại tệp này và chuyển đổi nó đúng với lý tự unicode
            self._rewrite_json_with_readable_unicode(document_json_path)
        except Exception as exception:
            raise RuntimeError(
                f"Không thể xuất Docling JSON: {document_json_path}"
            ) from exception

        # Tạo chunk record theo đúng cấu trúc
        chunk_records = self._create_chunk_records( document=document, source_file=resolved_source_file, source_hash=source_hash, document_id=selected_document_id, )

        # Kiểm tra object trong RAM trước khi ghi.
        self._validate_chunk_records(chunk_records)

        # Ghi atomic: ghi file .tmp trước, thành công mới replace file chính.
        self._write_chunks_jsonl_atomic(
            chunks_jsonl_path=chunks_jsonl_path,
            chunks=chunk_records,
        )

        # Đọc lại file trên ổ đĩa để phát hiện JSONL bị lỗi hoặc thiếu dòng.
        self._validate_written_jsonl( chunks_jsonl_path=chunks_jsonl_path, expected_chunks=chunk_records, )

        logger.info( "Chunking hoàn tất. document_id=%s, chunks=%s.", selected_document_id, len(chunk_records), )

        return ChunkingResult(
            document_id=selected_document_id,
            source_file=resolved_source_file.name,
            source_hash=source_hash,
            document_json_path=document_json_path,
            markdown_path=markdown_path,
            chunks_jsonl_path=chunks_jsonl_path,
            chunks=chunk_records,
        )

    # --------------------------------------------------------
    # TẠO TOKENIZER VÀ HYBRID CHUNKER
    # --------------------------------------------------------

    def _create_chunking_runtime(self) -> ChunkingRuntime:
        """
        Tải tokenizer và tính ngân sách token nội dung.

        Công thức:
            **content_budget = model_max_tokens - prefix_tokens - special_tokens - safety_margin**
        """

        try:
            # Tải tokenizer
            tokenizer = AutoTokenizer.from_pretrained( self.settings.embedding_tokenizer_name, trust_remote_code=True, )
        except Exception as exception:
            raise RuntimeError(
                "Không thể tải tokenizer "
                f"`{self.settings.embedding_tokenizer_name}`. "
                "Hãy kiểm tra Internet/cache Hugging Face và tên model."
            ) from exception

        # Tính toán token cho prefix: search_document
        prefix_token_count = self._count_tokens( tokenizer=tokenizer, text=self.settings.embedding_document_prefix, add_special_tokens=False, )
        # Tính một số token cho các ký tự đặc biệt
        special_token_count = (
            tokenizer.num_special_tokens_to_add(pair=False)
        )

        # Tính toán lượng token còn laị có thể dành cho nội dung embedding
        chunk_content_budget = (
            self.settings.embedding_model_max_tokens
            - prefix_token_count
            - special_token_count
            - self.settings.chunk_token_safety_margin     # Một lượng token để đamr bảo cho margin
        )

        if self.settings.embedding_model_max_tokens <= 0:
            raise ValueError("embedding_model_max_tokens phải lớn hơn 0.")

        if self.settings.chunk_token_safety_margin < 0:
            raise ValueError("chunk_token_safety_margin không được âm.")

        if chunk_content_budget <= 0:
            raise ValueError(
                "Cấu hình token không hợp lệ: "
                f"model={self.settings.embedding_model_max_tokens}, "
                f"prefix={prefix_token_count}, "
                f"special={special_token_count}, "
                f"safety={self.settings.chunk_token_safety_margin}. "
                "Không còn ngân sách cho nội dung chunk."
            )

        # Khởi tạo các model nặng một lần
        docling_tokenizer = HuggingFaceTokenizer(
            tokenizer=tokenizer,
            max_tokens=chunk_content_budget,
        )

        chunker = HybridChunker(
            tokenizer=docling_tokenizer,
            merge_peers=True,
            repeat_table_header=True,
            omit_header_on_overflow=False,
        )

        logger.info(
            "Token budget: model=%s, prefix=%s, special=%s, "
            "safety=%s, content=%s.",
            self.settings.embedding_model_max_tokens,
            prefix_token_count,
            special_token_count,
            self.settings.chunk_token_safety_margin,
            chunk_content_budget,
        )

        return ChunkingRuntime(
            huggingface_tokenizer=tokenizer,
            docling_tokenizer=docling_tokenizer,
            chunker=chunker,
            chunk_content_budget=chunk_content_budget,
        )

    # --------------------------------------------------------
    # TẠO CÁC CHUNK RECORD
    # --------------------------------------------------------

    def _create_chunk_records(self, document: Any, source_file: Path, source_hash: str, document_id: str) -> list[ChunkRecord]:
        """
        Duyệt chunk Docling và chuyển thành ChunkRecord.
        """
        # Khai báo danh sách chứa chunk record sau cùng
        final_records: list[ChunkRecord] = []
        output_chunk_index = 0

        try:
            chunk_iterator = self.runtime.chunker.chunk(dl_doc=document)

            for docling_chunk_index, chunk in enumerate(chunk_iterator):
                raw_text = str(chunk.text).strip()

                if not raw_text:
                    logger.debug("Bỏ Docling chunk %s vì raw_text rỗng.",docling_chunk_index,)
                    continue

                contextualized_text = (self.runtime.chunker.contextualize(chunk=chunk).strip())

                if not contextualized_text:
                    logger.debug("Bỏ Docling chunk %s vì contextualized_text rỗng.",docling_chunk_index,)
                    continue

                metadata = chunk.meta.export_json_dict()

                headings = self._normalize_string_list(metadata.get("headings"))
                captions = self._normalize_string_list(metadata.get("captions"))
                page_numbers = sorted(self._find_page_numbers(metadata))
                doc_item_refs = self._extract_doc_item_refs(metadata)

                embedding_text = (self.settings.embedding_document_prefix + contextualized_text)

                embedding_token_count = self._count_tokens(
                    tokenizer=self.runtime.huggingface_tokenizer,
                    text=embedding_text,
                    add_special_tokens=True,
                )

                if (embedding_token_count <= self.settings.embedding_model_max_tokens):
                    text_parts = [contextualized_text]
                else:
                    logger.warning(
                        "Docling chunk %s có %s token sau prefix; "
                        "kích hoạt fallback split.",
                        docling_chunk_index,
                        embedding_token_count,
                    )

                    text_parts = (
                        self._hard_split_contextualized_text(
                            contextualized_text
                        )
                    )

                if not text_parts:
                    raise RuntimeError(f"Docling chunk {docling_chunk_index} không tạo được text_part.")

                for subchunk_index, text_part in enumerate(text_parts):
                    final_embedding_text = (self.settings.embedding_document_prefix + text_part)

                    final_token_count = self._count_tokens(
                        tokenizer=self.runtime.huggingface_tokenizer,
                        text=final_embedding_text,
                        add_special_tokens=True,
                    )

                    if (final_token_count > self.settings.embedding_model_max_tokens):
                        raise RuntimeError(
                            "Chunk cuối vẫn vượt token limit: "
                            f"docling_chunk={docling_chunk_index}, "
                            f"subchunk={subchunk_index}, "
                            f"tokens={final_token_count}, "
                            f"limit="
                            f"{self.settings.embedding_model_max_tokens}."
                        )

                    point_id = self._create_stable_point_id(
                        document_id=document_id,
                        source_hash=source_hash,
                        docling_chunk_index=docling_chunk_index,
                        subchunk_index=subchunk_index,
                    )

                    page_number_status = (
                        "available_from_provenance"
                        if page_numbers
                        else "unavailable_in_source_provenance"
                    )

                    final_records.append(
                        ChunkRecord(
                            id=point_id,
                            chunk_index=output_chunk_index,
                            docling_chunk_index=(
                                docling_chunk_index
                            ),
                            subchunk_index=subchunk_index,
                            subchunk_count=len(text_parts),
                            document_id=document_id,
                            source_file=source_file.name,

                            # Hữu ích để debug local.
                            # Không nên đưa đường dẫn này vào payload
                            # public nếu hệ thống chạy production.
                            source_path=str(source_file),

                            source_hash=source_hash,
                            page_numbers=page_numbers,
                            page_number_status=page_number_status,
                            headings=headings,
                            captions=captions,
                            doc_item_refs=doc_item_refs,

                            # Nếu không fallback, giữ raw_text.
                            # Nếu fallback, mỗi record chỉ chứa phần
                            # tương ứng để tránh text không khớp vector.
                            text=(
                                raw_text
                                if len(text_parts) == 1
                                else text_part
                            ),
                            contextualized_text=text_part,
                            embedding_text=final_embedding_text,
                            embedding_token_count=(
                                final_token_count
                            ),
                            embedding_model_max_tokens=(
                                self.settings
                                .embedding_model_max_tokens
                            ),
                            chunk_content_budget=(
                                self.runtime
                                .chunk_content_budget
                            ),
                            chunk_schema_version=(
                                self.chunk_schema_version
                            ),

                            # Metadata này hữu ích trong JSONL/debug,
                            # nhưng không nên lặp toàn bộ vào Qdrant
                            # payload nếu dung lượng lớn.
                            docling_metadata=metadata,
                        )
                    )

                    output_chunk_index += 1

        except Exception as exception:
            if isinstance(exception, RuntimeError):
                raise

            raise RuntimeError("Phát sinh lỗi khi HybridChunker tạo chunk.") from exception

        if not final_records:
            raise RuntimeError(
                "Docling không tạo được chunk có nội dung. "
                "Hãy kiểm tra file Markdown đầu ra; "
                "nếu nội dung nằm trong ảnh, cần bật OCR."
            )

        return final_records

    # --------------------------------------------------------
    # FALLBACK TOKEN SPLIT ĐÃ SỬA LỖI BỎ SÓT TOKEN
    # --------------------------------------------------------

    def _hard_split_contextualized_text(self,contextualized_text: str,) -> list[str]:
        """
        Cắt text theo token ID khi HybridChunker vẫn tạo chunk quá dài.

        Điểm quan trọng
        --------------
        Khi cửa sổ bị thu nhỏ vì decode + prefix vượt limit,
        bước nhảy tiếp theo phải dựa trên số token THỰC SỰ đã dùng.

        Nếu vẫn dùng window_size ban đầu, có thể:
        - bỏ sót một đoạn token;
        - dừng sớm và mất phần cuối;
        - tạo khoảng trống giữa hai subchunk.
        """

        tokenizer = self.runtime.huggingface_tokenizer

        token_ids = tokenizer.encode(
            contextualized_text,
            add_special_tokens=False,
        )

        if not token_ids:
            return []

        maximum_window_size = (
            self.runtime.chunk_content_budget
        )

        # Overlap tối đa 32 token hoặc 10% cửa sổ.
        configured_overlap = min(
            32,
            max(0, maximum_window_size // 10),
        )

        text_parts: list[str] = []
        start_index = 0

        while start_index < len(token_ids):
            candidate_end = min(
                start_index + maximum_window_size,
                len(token_ids),
            )

            current_token_ids = token_ids[
                start_index:candidate_end
            ]

            # Giảm cửa sổ đến khi text cuối cùng vừa limit.
            while current_token_ids:
                current_text = tokenizer.decode(
                    current_token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()

                if not current_text:
                    current_token_ids = current_token_ids[:-1]
                    continue

                final_text = (
                    self.settings.embedding_document_prefix
                    + current_text
                )

                token_count = self._count_tokens(
                    tokenizer=tokenizer,
                    text=final_text,
                    add_special_tokens=True,
                )

                if (
                    token_count
                    <= self.settings.embedding_model_max_tokens
                ):
                    break

                # Giảm theo block nhỏ để nhanh hơn giảm từng token.
                shrink_size = min(8, len(current_token_ids))
                current_token_ids = current_token_ids[
                    :-shrink_size
                ]

            if not current_token_ids:
                raise RuntimeError(
                    "Không thể tạo fallback chunk nằm trong "
                    "token limit."
                )

            current_text = tokenizer.decode(
                current_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            if not current_text:
                raise RuntimeError(
                    "Tokenizer decode tạo subchunk rỗng."
                )

            text_parts.append(current_text)

            # Số token thực tế đã được sử dụng ở subchunk này.
            consumed_tokens = len(current_token_ids)
            actual_end = start_index + consumed_tokens

            # Đã bao phủ hết tài liệu thì dừng.
            if actual_end >= len(token_ids):
                break

            # Overlap không được >= consumed_tokens,
            # nếu không start_index có thể không tiến lên.
            actual_overlap = min(
                configured_overlap,
                max(0, consumed_tokens - 1),
            )

            next_start = actual_end - actual_overlap

            if next_start <= start_index:
                raise RuntimeError(
                    "Fallback split không tiến được start_index; "
                    "ngăn vòng lặp vô hạn."
                )

            start_index = next_start

        return text_parts

    # --------------------------------------------------------
    # KIỂM TRA CHUNK TRONG RAM
    # --------------------------------------------------------

    def _validate_chunk_records(self,chunks: list[ChunkRecord],) -> ChunkValidationReport:
        """
        Kiểm tra tính toàn vẹn trước khi ghi JSONL.
        """

        if not chunks:
            raise ValueError("Danh sách chunk đang rỗng.")
        # Tính số lượng chunk có trong danh sách
        expected_indices = list(range(len(chunks)))
        actual_indices = [
            chunk.chunk_index for chunk in chunks
        ]

        # Nếu khác nhau thì có nghĩa là nó có sự ngắt quảng, không khớp
        if actual_indices != expected_indices:
            raise RuntimeError(
                "chunk_index không liên tục từ 0."
            )   
        # Xác thực point id có trùng không
        point_ids = [chunk.id for chunk in chunks]

        if len(point_ids) != len(set(point_ids)):
            raise RuntimeError(
                "Phát hiện point ID bị trùng."
            )
        # Đếm số lượng token cho toàn bộ chunk
        token_counts: list[int] = []

        for chunk in chunks:
            # Loại bỏ các khoảng trắng thừa hai bên
            if not chunk.text.strip():
                raise RuntimeError(
                    f"Chunk {chunk.chunk_index} có text rỗng."
                )
            # Loại bỏ khoảng trắng thừa cho đoạn văn bản embedding
            if not chunk.contextualized_text.strip():
                raise RuntimeError(
                    f"Chunk {chunk.chunk_index} có "
                    "contextualized_text rỗng."
                )

            # Kiểm tra từ kháo bắt đầu cho chuỗi embeeding phải luôn là: Search_document
            if not chunk.embedding_text.startswith( self.settings.embedding_document_prefix ):
                raise RuntimeError(
                    f"Chunk {chunk.chunk_index} thiếu prefix embedding document."
                )

            # Đếm số token
            actual_token_count = self._count_tokens(
                tokenizer=self.runtime.huggingface_tokenizer,
                text=chunk.embedding_text,
                add_special_tokens=True,
            )

            if actual_token_count != (
                chunk.embedding_token_count
            ):
                raise RuntimeError(
                    f"Chunk {chunk.chunk_index}: token lưu="
                    f"{chunk.embedding_token_count}, token thực="
                    f"{actual_token_count}."
                )

            if actual_token_count > (
                self.settings.embedding_model_max_tokens
            ):
                raise RuntimeError(
                    f"Chunk {chunk.chunk_index} vượt token limit."
                )

            if chunk.subchunk_count <= 0:
                raise RuntimeError(
                    f"Chunk {chunk.chunk_index} có "
                    "subchunk_count không hợp lệ."
                )

            if not (
                0
                <= chunk.subchunk_index
                < chunk.subchunk_count
            ):
                raise RuntimeError(
                    f"Chunk {chunk.chunk_index} có "
                    "subchunk_index không hợp lệ."
                )

            token_counts.append(actual_token_count)

        report = ChunkValidationReport(
            chunk_count=len(chunks),
            minimum_tokens=min(token_counts),
            maximum_tokens=max(token_counts),
            average_tokens=(
                sum(token_counts) / len(token_counts)
            ),
            chunks_with_pages=sum(
                1 for chunk in chunks if chunk.page_numbers
            ),
            chunks_without_pages=sum(
                1 for chunk in chunks if not chunk.page_numbers
            ),
            fallback_subchunks=sum(
                1
                for chunk in chunks
                if chunk.subchunk_count > 1
            ),
        )

        logger.info(
            "Chunk validation: count=%s, min=%s, max=%s, "
            "avg=%.2f, fallback=%s.",
            report.chunk_count,
            report.minimum_tokens,
            report.maximum_tokens,
            report.average_tokens,
            report.fallback_subchunks,
        )

        return report

    # --------------------------------------------------------
    # GHI VÀ ĐỌC LẠI JSONL
    # --------------------------------------------------------

    def _write_chunks_jsonl_atomic(
        self,
        chunks_jsonl_path: Path,
        chunks: list[ChunkRecord],
    ) -> None:
        """
        Ghi JSONL theo cách atomic.

        Nếu chương trình lỗi giữa chừng:
        - file chính cũ không bị biến thành file dở dang;
        - chỉ còn file .tmp để kiểm tra/xóa.
        """

        temporary_path = chunks_jsonl_path.with_suffix(
            chunks_jsonl_path.suffix + ".tmp"
        )

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
            ) as file_object:
                for chunk in chunks:
                    file_object.write(
                        json.dumps(
                            asdict(chunk),
                            ensure_ascii=False,
                        )
                    )
                    file_object.write("\n")

                # Đẩy buffer Python xuống hệ điều hành.
                file_object.flush()
                os.fsync(file_object.fileno())

            # replace là atomic trên cùng filesystem.
            temporary_path.replace(chunks_jsonl_path)

        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def _validate_written_jsonl( self, chunks_jsonl_path: Path, expected_chunks: list[ChunkRecord], ) -> None:
        """
        Đọc lại JSONL để bảo đảm:
        - mỗi dòng là JSON object hợp lệ;
        - số dòng khớp;
        - ID/chunk_index/token khớp object trong RAM.
        """

        records: list[dict[str, Any]] = []

        with chunks_jsonl_path.open(
            "r",
            encoding="utf-8",
        ) as file_object:
            for line_number, line in enumerate(
                file_object,
                start=1,
            ):
                stripped_line = line.strip()

                if not stripped_line:
                    continue

                try:
                    record = json.loads(stripped_line)
                except json.JSONDecodeError as exception:
                    raise RuntimeError(
                        f"JSONL lỗi tại dòng {line_number}: "
                        f"{exception}"
                    ) from exception

                if not isinstance(record, dict):
                    raise RuntimeError(
                        f"JSONL dòng {line_number} "
                        "không phải JSON object."
                    )

                records.append(record)

        if len(records) != len(expected_chunks):
            raise RuntimeError(
                "Số record JSONL không khớp chunks trong RAM: "
                f"{len(records)} != {len(expected_chunks)}."
            )

        for record, expected in zip(
            records,
            expected_chunks,
            strict=True,
        ):
            if record.get("id") != expected.id:
                raise RuntimeError(
                    f"JSONL chunk {expected.chunk_index}: ID sai."
                )

            if record.get("chunk_index") != (
                expected.chunk_index
            ):
                raise RuntimeError(
                    f"JSONL chunk {expected.chunk_index}: "
                    "chunk_index sai."
                )

            if record.get("embedding_token_count") != (
                expected.embedding_token_count
            ):
                raise RuntimeError(
                    f"JSONL chunk {expected.chunk_index}: "
                    "token count sai."
                )

    # --------------------------------------------------------
    # CÁC HÀM TIỆN ÍCH
    # --------------------------------------------------------

    def _validate_source_file(self, source_file: Path) -> None:
        """
        Kiểm tra file nguồn trước khi gọi Docling.
        """

        if not source_file.exists():
            raise FileNotFoundError(f"Không tìm thấy tài liệu: {source_file}")

        if not source_file.is_file():
            raise ValueError(f"Đường dẫn không phải file: {source_file}")

        try:
            file_size = source_file.stat().st_size
        except OSError as exception:
            raise RuntimeError(f"Không thể đọc metadata file: {source_file}") from exception

        if file_size <= 0:
            raise ValueError(f"Tài liệu đang rỗng: {source_file}")

        # Chỉ kiểm tra quyền đọc ở mức hệ điều hành.
        if not os.access(source_file, os.R_OK):
            raise PermissionError(f"Không có quyền đọc file: {source_file}")

    def _select_document_id(self, document_id: str | None, source_file: Path,) -> str:
        """
        Chọn document_id nghiệp vụ.

        Không bắt buộc document_id phải giống tên file.
        Ví dụ một tài liệu cập nhật nhiều lần vẫn giữ:
            document_id="fabric-warehouse-guide"
        """

        selected = (
            document_id.strip()
            if document_id is not None
            else ""
        )
        # Lấy tên file
        if not selected:
            selected = source_file.stem

        if len(selected) > 200:
            raise ValueError("document_id quá dài; tối đa 200 ký tự.")

        return selected

    def _build_output_paths( self, source_file: Path, output_directory: Path, ) -> tuple[Path, Path, Path, Path]:
        """
        Tạo toàn bộ đường dẫn đầu ra.
        """

        document_json_path = (
            output_directory
            / f"{source_file.stem}.document.json"
        )
        markdown_path = (
            output_directory
            / f"{source_file.stem}.md"
        )
        chunks_jsonl_path = (
            output_directory
            / f"{source_file.stem}.chunks.jsonl"
        )
        artifacts_directory = (
            output_directory
            / f"{source_file.stem}_artifacts"
        )

        artifacts_directory.mkdir( parents=True, exist_ok=True, )

        return (
            document_json_path,
            markdown_path,
            chunks_jsonl_path,
            artifacts_directory,
        )

    def _create_stable_point_id(self, document_id: str, source_hash: str, docling_chunk_index: int, subchunk_index: int) -> str:
        """
        Tạo UUID5 ổn định.

        Cùng đầu vào sẽ tạo cùng ID.
        Khi nội dung file đổi, source_hash đổi và ID cũng đổi.
        """

        identity_text = (
            f"{document_id}:{source_hash}:"
            f"{docling_chunk_index}:{subchunk_index}:"
            f"{self.chunk_schema_version}"
        )

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                identity_text,
            )
        )

    def _rewrite_json_with_readable_unicode(self, json_path: Path) -> None:
        """
        Ghi lại JSON với ensure_ascii=False.

        Lưu ý:
        Đây chỉ thay đổi cách hiển thị:
            "\\u1eadn" -> "ậ"
        không thay đổi ý nghĩa dữ liệu.
        """
        # Đọc dữ liệu từ tệp json gốc
        document_data = json.loads(json_path.read_text(encoding="utf-8"))
        # Tạo đường dẫn tệp tạm
        temporary_path = json_path.with_suffix(json_path.suffix + ".tmp")

        try:
            temporary_path.write_text(
                json.dumps(
                    document_data,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary_path.replace(json_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def _calculate_file_sha256( self, file_path: Path, ) -> str:
        """
        Tính SHA-256 theo block 1 MiB.
        """

        sha256_hash = hashlib.sha256()

        with file_path.open("rb") as file_object:
            while True:
                data_block = file_object.read(
                    1024 * 1024
                )

                if not data_block:
                    break

                sha256_hash.update(data_block)

        return sha256_hash.hexdigest()

    def _count_tokens(self, tokenizer: PreTrainedTokenizerBase, text: str, add_special_tokens: bool) -> int:
        """
        Đếm token bằng tokenizer embedding model.
        """

        if not isinstance(text, str):
            raise TypeError("Text cần đếm token phải là string.")

        encoded = tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=False,
            padding=False,
        )

        input_ids = encoded.get("input_ids")

        if not isinstance(input_ids, list):
            raise RuntimeError("Tokenizer không trả input_ids dạng list.")

        return len(input_ids)

    def _find_page_numbers(self, value: Any) -> set[int]:
        """
        Duyệt đệ quy metadata để tìm page_no.  
        Hiện tại chỉ có thể lấy số trang đối với tài liệu pdf, có thể chuyển đổi word sang pđf và thực hiện lại
        """

        page_numbers: set[int] = set()

        if isinstance(value, dict):
            # doc_items = value["doc_items"]
            # first_doc_item = doc_items[0]
            # prov_value = first_doc_item["prov"]
            # page_number = prov_value[0].page_no
            # print(page_number)
            for key, child_value in value.items():
                if (key == "page_no" and isinstance(child_value, int) and child_value > 0):
                    page_numbers.add(child_value)

                page_numbers.update(self._find_page_numbers(child_value))

        elif isinstance(value, list):
            for item in value:
                page_numbers.update(self._find_page_numbers(item))

        return page_numbers

    def _extract_doc_item_refs(self, metadata: dict[str, Any]) -> list[str]:
        """
        Lấy self_ref và loại trùng nhưng giữ nguyên thứ tự.
        """

        references: list[str] = []
        seen: set[str] = set()

        doc_items = metadata.get("doc_items")

        if not isinstance(doc_items, list):
            return references

        for doc_item in doc_items:
            if not isinstance(doc_item, dict):
                continue

            self_reference = doc_item.get("self_ref")

            if (
                isinstance(self_reference, str)
                and self_reference
                and self_reference not in seen
            ):
                references.append(self_reference)
                seen.add(self_reference)

        return references

    def _normalize_string_list(self, value: Any) -> list[str]:
        """
        Chuẩn hóa heading/caption thành list[str],
        bỏ phần tử rỗng và loại trùng.
        """

        if value is None:
            return []

        if isinstance(value, str):
            raw_items: Iterable[Any] = [value]
        elif isinstance(value, (list, tuple, set)):
            raw_items = value
        else:
            raw_items = [value]

        output: list[str] = []
        seen: set[str] = set()

        for item in raw_items:
            normalized_item = str(item).strip()

            if (
                normalized_item
                and normalized_item not in seen
            ):
                output.append(normalized_item)
                seen.add(normalized_item)

        return output


# ============================================================
# 3. HÀM TEST MỘT TỆP
# ============================================================

def print_chunking_summary(result: ChunkingResult, service: DocumentChunkingService, preview_count: int, show_full_text: bool) -> None:
    """
    In báo cáo dễ đọc sau khi test.
    """

    report = service._validate_chunk_records(
        result.chunks
    )

    print()
    print("=" * 88)
    print("KẾT QUẢ CHUNKING")
    print("=" * 88)
    print(f"Document ID       : {result.document_id}")
    print(f"File nguồn        : {result.source_file}")
    print(f"SHA-256           : {result.source_hash}")
    print(f"Số chunk          : {report.chunk_count}")
    print(
        f"Token min/avg/max : "
        f"{report.minimum_tokens}/"
        f"{report.average_tokens:.2f}/"
        f"{report.maximum_tokens}"
    )
    print(f"Chunk có trang    : {report.chunks_with_pages}")
    print(f"Chunk không trang : {report.chunks_without_pages}")
    print(f"Fallback subchunk : {report.fallback_subchunks}")
    print(f"Markdown          : {result.markdown_path}")
    print(f"Document JSON     : {result.document_json_path}")
    print(f"Chunks JSONL      : {result.chunks_jsonl_path}")

    preview_count = max(0, min(preview_count, len(result.chunks)))

    if preview_count == 0:
        return

    print()
    print("=" * 88)
    print(f"XEM TRƯỚC {preview_count} CHUNK")
    print("=" * 88)

    for chunk in result.chunks[:preview_count]:
        print()
        print("-" * 88)
        print(
            f"chunk_index={chunk.chunk_index} | "
            f"docling={chunk.docling_chunk_index} | "
            f"sub={chunk.subchunk_index + 1}/"
            f"{chunk.subchunk_count} | "
            f"tokens={chunk.embedding_token_count}/"
            f"{chunk.embedding_model_max_tokens}"
        )
        print(f"id             : {chunk.id}")
        print(f"headings       : {chunk.headings}")
        print(f"pages          : {chunk.page_numbers}")
        print(f"doc_item_refs  : {chunk.doc_item_refs}")

        preview_text = chunk.contextualized_text

        if not show_full_text and len(preview_text) > 1000:
            preview_text = (
                preview_text[:1000].rstrip()
                + "\n...[đã rút gọn phần xem trước]..."
            )

        print("contextualized_text:")
        print(preview_text)

def configure_logging(level_name: str) -> None:
    """
    Cấu hình log cho CLI test.
    """

    logging.basicConfig(
        level=getattr(logging, level_name),
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


def main() -> int:
    """
    Chạy test một tệp.

    Hàm này đồng bộ vì DocumentChunkingService là đồng bộ.
    """
    configure_logging("INFO")

    try:
        settings = get_settings()

        service = DocumentChunkingService(settings=settings)

        result = service.process_document(
            source_file=Path("document/farbic_warehouse_document.docx"),
            output_directory=Path("./outputs/chunk_test"),
            document_id=None,
        )

        print_chunking_summary(
            result=result,
            service=service,
            preview_count=5,
            show_full_text="store_true",
        )

        return 0

    except KeyboardInterrupt:
        print("Đã dừng bởi người dùng.", file=sys.stderr)
        return 130

    except Exception as exception:
        logger.exception(
            "Test chunking thất bại."
        )
        print(
            f"\nLỖI: {exception}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())