from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from docling_core.types.doc import ImageRefMode


# ============================================================
# CẤU HÌNH MẶC ĐỊNH
# ============================================================

# Tokenizer phải tương ứng với embedding model thực tế, có thể khác với tokenizer mặc định của Docling.
# Ở đây embedding model là nomic-embed-text-v2-moe.
DEFAULT_TOKENIZER_ID = "nomic-ai/nomic-embed-text-v2-moe"

# Theo model card của Nomic, model nhận tối đa 512 token. Có thể truy vấn lại bằng tokenizer.model_max_length, nhưng để an toàn thì hardcode.
DEFAULT_MODEL_MAX_TOKENS = 512

# Prefix dùng khi embedding tài liệu. theo model card của Nomic, prefix cho query là "search_query: ", còn prefix cho document là "search_document: ".
DEFAULT_DOCUMENT_PREFIX = "search_document: "

# Chừa một ít token dự phòng để tránh chạm sát giới hạn model.
# Khoảng dự phòng này hữu ích vì:
# - Prefix và nội dung có thể ghép token hơi khác khi tokenize riêng.
# - Một số tokenizer tự thêm special token.
# - Phiên bản tokenizer/model có thể có khác biệt nhỏ.
DEFAULT_SAFETY_MARGIN = 8

# Chỉ dùng khi bước kiểm tra cuối phát hiện chunk vẫn vượt giới hạn.
# Các mảnh con sẽ chồng nhau một ít để giảm mất ngữ cảnh ở ranh giới.
DEFAULT_FALLBACK_OVERLAP_TOKENS = 32


# ============================================================
# KIỂU DỮ LIỆU GIỮA TOKENIZER VÀ CHUNKER
# ============================================================

@dataclass(frozen=True)
class ChunkingRuntime:
    """
    Gom các thành phần liên quan đến token/chunk vào một object.

    hf_tokenizer:
        Tokenizer gốc của Hugging Face.
        Dùng để đếm CHÍNH XÁC chuỗi cuối cùng gửi vào embedding model.

    docling_tokenizer:
        Lớp bọc tokenizer để HybridChunker của Docling sử dụng.

    chunker:
        HybridChunker dùng để chia DoclingDocument.

    model_max_tokens:
        Giới hạn cứng của embedding model, ví dụ 512.

    chunk_content_budget:
        Số token tối đa dành cho phần contextualized_text.
        Giá trị này đã trừ prefix, special token và safety margin.
    """

    hf_tokenizer: PreTrainedTokenizerBase
    docling_tokenizer: HuggingFaceTokenizer
    chunker: HybridChunker
    model_max_tokens: int
    chunk_content_budget: int
    prefix_token_count: int
    special_token_count: int


@dataclass(frozen=True)
class PreparedChunkText:
    """
    Một mảnh text đã sẵn sàng để lưu hoặc gửi đi embedding.

    contextualized_text:
        Nội dung có ngữ cảnh, nhưng chưa có prefix của embedding model.

    embedding_text:
        Chuỗi cuối cùng thực sự gửi vào embedding model.

    final_token_count:
        Tổng token của embedding_text, có tính special token.
    """

    contextualized_text: str
    embedding_text: str
    final_token_count: int


# ============================================================
# HÀM TIỆN ÍCH CƠ BẢN
# ============================================================

def calculate_file_sha256(file_path: Path) -> str:
    """
    Tính SHA-256 của file nguồn.

    Hash được dùng để:
    - Nhận biết file đã thay đổi hay chưa.
    - Hỗ trợ incremental ingestion.
    - Tạo ID ổn định cho chunk.
    """

    sha256 = hashlib.sha256()

    # Đọc theo từng khối 1 MB để không nạp cả file lớn vào RAM.
    with file_path.open("rb") as file_obj:
        while True:
            block = file_obj.read(1024 * 1024)

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


def find_page_numbers(value: Any) -> set[int]:
    """
    Tìm tất cả trường page_no nằm trong metadata lồng nhau của Docling.

    PDF thường có page_no trong provenance.
    DOCX có thể không có page_no vì Word không phải định dạng trang cố định.
    """

    page_numbers: set[int] = set()

    if isinstance(value, dict):
        for key, child_value in value.items():
            if key == "page_no" and isinstance(child_value, int):
                page_numbers.add(child_value)

            page_numbers.update(find_page_numbers(child_value))

    elif isinstance(value, list):
        for item in value:
            page_numbers.update(find_page_numbers(item))

    return page_numbers


def count_tokens(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    add_special_tokens: bool,
) -> int:
    """
    Đếm token bằng đúng tokenizer của embedding model.

    add_special_tokens=True:
        Dùng khi kiểm tra chuỗi cuối cùng gửi vào model.

    add_special_tokens=False:
        Dùng khi tính riêng prefix hoặc một thành phần text.
    """

    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
        padding=False,
    )

    return len(encoded["input_ids"])


# ============================================================
# TẠO TOKENIZER VÀ HYBRID CHUNKER AN TOÀN
# ============================================================

def create_chunking_runtime(
    tokenizer_model_id: str,
    model_max_tokens: int,
    embedding_prefix: str,
    safety_margin: int,
) -> ChunkingRuntime:
    """
    Tạo tokenizer và HybridChunker với ngân sách token an toàn.

    Công thức:

        chunk_content_budget = model_max_tokens - prefix_token_count - special_token_count - safety_margin

    Ví dụ minh họa:

        model_max_tokens   = 512
        prefix             = "search_document: "
        prefix tokens      = 4
        special tokens     = 2
        safety margin      = 8

        content budget     = 512 - 4 - 2 - 8 = 498 token

    HybridChunker sẽ cố giữ contextualized_text trong 498 token.
    Sau đó chương trình vẫn kiểm tra lại toàn bộ embedding_text.
    """

    if model_max_tokens <= 0:
        raise ValueError("model_max_tokens phải lớn hơn 0.")

    if safety_margin < 0:
        raise ValueError("safety_margin không được âm.")

    print(f"Đang tải tokenizer: {tokenizer_model_id}")

    # Tokenizer này phải tương ứng với embedding model thực tế.
    hf_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_id,
        trust_remote_code=True,
    )

    # Đếm token của prefix, không cộng special token ở bước này.
    prefix_token_count = count_tokens(
        hf_tokenizer,
        embedding_prefix,
        add_special_tokens=False,
    )

    # Số special token mà tokenizer sẽ thêm cho một chuỗi đơn.
    # Đây là cách đáng tin cậy hơn việc tự đoán BOS/CLS/EOS.
    special_token_count = hf_tokenizer.num_special_tokens_to_add(pair=False)

    # Ngân sách còn lại cho contextualized_text.
    chunk_content_budget = (
        model_max_tokens
        - prefix_token_count
        - special_token_count
        - safety_margin
    )

    if chunk_content_budget <= 0:
        raise ValueError(
            "Không còn ngân sách token cho nội dung. "
            "Hãy tăng model_max_tokens, giảm safety_margin "
            "hoặc dùng prefix ngắn hơn."
        )

    print("Cấu hình token:")
    print(f"- Giới hạn model             : {model_max_tokens}")
    print(f"- Token của prefix           : {prefix_token_count}")
    print(f"- Special token              : {special_token_count}")
    print(f"- Token dự phòng             : {safety_margin}")
    print(f"- Ngân sách contextualized   : {chunk_content_budget}")

    # Bọc tokenizer để Docling dùng cho HybridChunker.
    docling_tokenizer = HuggingFaceTokenizer(
        tokenizer=hf_tokenizer,
        max_tokens=chunk_content_budget,
    )

    # HybridChunker thực hiện:
    # 1. Chia theo cấu trúc tài liệu.
    # 2. Tách tiếp chunk quá dài theo token.
    # 3. Có thể gộp các chunk nhỏ cùng heading/caption.
    chunker = HybridChunker(
        tokenizer=docling_tokenizer,
        merge_peers=True,
        repeat_table_header=True,
        omit_header_on_overflow=False,
    )

    return ChunkingRuntime(
        hf_tokenizer=hf_tokenizer,
        docling_tokenizer=docling_tokenizer,
        chunker=chunker,
        model_max_tokens=model_max_tokens,
        chunk_content_budget=chunk_content_budget,
        prefix_token_count=prefix_token_count,
        special_token_count=special_token_count,
    )


# ============================================================
# HARD-SPLIT DỰ PHÒNG THEO TOKEN
# ============================================================

def hard_split_text_by_tokens(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    embedding_prefix: str,
    model_max_tokens: int,
    safety_margin: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Chia text thành các mảnh nhỏ hơn bằng token ID.

    Hàm này chỉ là lớp bảo vệ cuối cùng khi một chunk vẫn vượt giới hạn,
    ví dụ do heading/caption quá dài hoặc cách ghép prefix làm thay đổi
    số token so với khi đếm riêng từng thành phần.

    Ưu điểm:
    - Bảo đảm không gửi chuỗi vượt model_max_tokens.

    Hạn chế:
    - Có thể cắt ở giữa câu hoặc giữa một đơn vị logic.
    - Vì vậy chỉ nên dùng làm fallback, không phải chunker chính.
    """

    if overlap_tokens < 0:
        raise ValueError("overlap_tokens không được âm.")

    prefix_tokens = count_tokens(
        tokenizer,
        embedding_prefix,
        add_special_tokens=False,
    )
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)

    # Trừ thêm safety margin để không chạm sát trần.
    max_content_tokens = (
        model_max_tokens
        - prefix_tokens
        - special_tokens
        - safety_margin
    )

    if max_content_tokens <= 0:
        raise ValueError("Không đủ token budget để hard-split nội dung.")

    if overlap_tokens >= max_content_tokens:
        raise ValueError(
            "overlap_tokens phải nhỏ hơn số token tối đa của một mảnh."
        )

    # Mã hóa phần nội dung, không thêm special token.
    token_ids: list[int] = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )["input_ids"]

    pieces: list[str] = []
    start = 0

    while start < len(token_ids):
        end = min(start + max_content_tokens, len(token_ids))

        # Decode đoạn token thành text.
        piece = tokenizer.decode(
            token_ids[start:end],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        if not piece:
            raise RuntimeError(
                "Tokenizer tạo ra một mảnh rỗng trong quá trình hard-split."
            )

        # Sau decode rồi tokenize lại, số token đôi khi có thể thay đổi nhẹ.
        # Vì vậy giảm end dần cho đến khi chuỗi cuối cùng thực sự vừa giới hạn.
        while (
            count_tokens(
                tokenizer,
                f"{embedding_prefix}{piece}",
                add_special_tokens=True,
            )
            > model_max_tokens
        ):
            end -= 1

            if end <= start:
                raise RuntimeError(
                    "Không thể tạo mảnh text nằm trong giới hạn token."
                )

            piece = tokenizer.decode(
                token_ids[start:end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

        pieces.append(piece)

        if end >= len(token_ids):
            break

        # Tạo overlap giữa hai mảnh để giảm mất ngữ cảnh ở ranh giới.
        start = end - overlap_tokens

    return pieces


def prepare_embedding_texts(
    contextualized_text: str,
    runtime: ChunkingRuntime,
    embedding_prefix: str,
    safety_margin: int,
    fallback_overlap_tokens: int,
) -> list[PreparedChunkText]:
    """
    Chuẩn bị một hoặc nhiều chuỗi an toàn để embedding.

    Trường hợp bình thường:
        Trả về một phần tử duy nhất.

    Trường hợp vẫn vượt giới hạn:
        Gọi hard_split_text_by_tokens() để tạo nhiều mảnh nhỏ.

    Mọi phần tử trả về đều được kiểm tra:
        final_token_count <= model_max_tokens
    """

    embedding_text = f"{embedding_prefix}{contextualized_text}"

    final_token_count = count_tokens(
        runtime.hf_tokenizer,
        embedding_text,
        add_special_tokens=True,
    )

    if final_token_count <= runtime.model_max_tokens:
        return [
            PreparedChunkText(
                contextualized_text=contextualized_text,
                embedding_text=embedding_text,
                final_token_count=final_token_count,
            )
        ]

    print(
        "Cảnh báo: một chunk vẫn vượt giới hạn sau HybridChunker "
        f"({final_token_count} > {runtime.model_max_tokens}). "
        "Đang hard-split dự phòng..."
    )

    pieces = hard_split_text_by_tokens(
        text=contextualized_text,
        tokenizer=runtime.hf_tokenizer,
        embedding_prefix=embedding_prefix,
        model_max_tokens=runtime.model_max_tokens,
        safety_margin=safety_margin,
        overlap_tokens=fallback_overlap_tokens,
    )

    prepared: list[PreparedChunkText] = []

    for piece in pieces:
        piece_embedding_text = f"{embedding_prefix}{piece}"
        piece_token_count = count_tokens(
            runtime.hf_tokenizer,
            piece_embedding_text,
            add_special_tokens=True,
        )

        # Đây là kiểm tra cứng cuối cùng.
        # Nếu điều kiện này sai, tuyệt đối không nên gọi embedding model.
        if piece_token_count > runtime.model_max_tokens:
            raise RuntimeError(
                "Lỗi nội bộ: hard-split vẫn tạo text vượt giới hạn token: "
                f"{piece_token_count} > {runtime.model_max_tokens}."
            )

        prepared.append(
            PreparedChunkText(
                contextualized_text=piece,
                embedding_text=piece_embedding_text,
                final_token_count=piece_token_count,
            )
        )

    return prepared


# ============================================================
# XUẤT CÁC PHIÊN BẢN TÀI LIỆU
# ============================================================

def export_document_versions(
    document: Any,
    source_file: Path,
    output_dir: Path,
) -> None:
    """
    Xuất DoclingDocument thành nhiều định dạng để kiểm tra/debug.
    """

    file_stem = source_file.stem
    artifacts_dir = output_dir / f"{file_stem}_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Markdown: ảnh được lưu ra thư mục artifacts và tham chiếu bằng đường dẫn.
    document.save_as_markdown(
        filename=output_dir / f"{file_stem}.md",
        artifacts_dir=artifacts_dir,
        image_mode=ImageRefMode.REFERENCED,
    )

    # JSON: dùng PLACEHOLDER để tránh nhúng Base64 làm file quá lớn.
    json_path = output_dir / f"{source_file.stem}.json"

    # Bước 1:
    # Để Docling tạo đúng cấu trúc JSON trước. Có thể lỗi tiếng Việt
    document.save_as_json(
        filename=json_path,
        image_mode=ImageRefMode.PLACEHOLDER,
        indent=2,
    )

    # Bước 2:
    # Đọc lại JSON. Các chuỗi \\u... tự động trở thành Unicode.
    document_dict = json.loads(
        json_path.read_text(encoding="utf-8")
    )

    # Bước 3:
    # Ghi lại, giữ nguyên ký tự tiếng Việt.
    json_path.write_text(
        json.dumps(
            document_dict,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Plain text: tiện kiểm tra nhanh phần text Docling nhận dạng được.
    plain_text = document.export_to_text()
    (output_dir / f"{file_stem}.txt").write_text(
        plain_text,
        encoding="utf-8",
    )


# ============================================================
# TẠO VÀ LƯU CHUNKS JSONL
# ============================================================

def export_chunks_to_jsonl(
    document: Any,
    source_file: Path,
    output_dir: Path,
    runtime: ChunkingRuntime,
    embedding_prefix: str,
    safety_margin: int,
    fallback_overlap_tokens: int,
) -> dict[str, Any]:
    """
    Chia tài liệu thành chunk và lưu mỗi chunk trên một dòng JSONL.

    Quy trình của từng chunk:

    1. Docling tạo chunk theo cấu trúc tài liệu.
    2. contextualize() thêm heading/caption liên quan.
    3. Thêm prefix search_document:.
    4. Đếm token trên chuỗi cuối cùng.
    5. Nếu vượt 512 token thì hard-split.
    6. Chỉ ghi chunk khi đã chắc chắn không vượt giới hạn.
    """

    output_file = output_dir / f"{source_file.stem}.chunks.jsonl"
    source_hash = calculate_file_sha256(source_file)

    # final_chunk_index tăng liên tục theo record thật sự ghi ra file.
    final_chunk_index = 0

    # Thống kê để kiểm tra chất lượng chunking.
    final_token_counts: list[int] = []
    docling_chunk_count = 0
    fallback_split_count = 0

    with output_file.open("w", encoding="utf-8") as jsonl_file:
        for docling_chunk_index, chunk in enumerate(
            runtime.chunker.chunk(dl_doc=document)
        ):
            docling_chunk_count += 1

            raw_text = chunk.text.strip()

            # Bỏ qua chunk không chứa text.
            if not raw_text:
                continue

            # contextualize() thường bổ sung heading và caption.
            # Đây là nội dung phù hợp hơn chunk.text để tạo embedding.
            contextualized_text = runtime.chunker.contextualize(
                chunk=chunk
            ).strip()

            if not contextualized_text:
                continue

            # Chuẩn bị chuỗi embedding và hard-split nếu cần.
            prepared_parts = prepare_embedding_texts(
                contextualized_text=contextualized_text,
                runtime=runtime,
                embedding_prefix=embedding_prefix,
                safety_margin=safety_margin,
                fallback_overlap_tokens=fallback_overlap_tokens,
            )

            if len(prepared_parts) > 1:
                fallback_split_count += 1

            docling_metadata = chunk.meta.export_json_dict()
            page_numbers = sorted(find_page_numbers(docling_metadata))

            for subchunk_index, prepared in enumerate(prepared_parts):
                # ID ổn định dựa trên file + chunk gốc + mảnh con.
                chunk_uuid = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        (
                            f"{source_hash}:"
                            f"{docling_chunk_index}:"
                            f"{subchunk_index}"
                        ),
                    )
                )

                chunk_record = {
                    # ID phù hợp để dùng làm Qdrant point ID.
                    "id": chunk_uuid,

                    # Chỉ số liên tục của record cuối cùng.
                    "chunk_index": final_chunk_index,

                    # Chỉ số chunk ban đầu do Docling tạo.
                    "docling_chunk_index": docling_chunk_index,

                    # Nếu chunk gốc bị hard-split, đây là vị trí mảnh con.
                    "subchunk_index": subchunk_index,
                    "subchunk_count": len(prepared_parts),

                    # Thông tin nguồn.
                    "source_file": source_file.name,
                    "source_path": str(source_file.resolve()),
                    "source_hash": source_hash,
                    "page_numbers": page_numbers,

                    # Text gốc do Docling tạo.
                    "text": raw_text,

                    # Text đã có heading/caption, có thể đã được hard-split.
                    "contextualized_text": prepared.contextualized_text,

                    # Chuỗi cuối cùng gửi vào embedding model.
                    "embedding_text": prepared.embedding_text,

                    # Số token thật của chuỗi cuối cùng.
                    "embedding_token_count": prepared.final_token_count,

                    # Giới hạn để tiện audit/debug.
                    "embedding_model_max_tokens": runtime.model_max_tokens,
                    "chunk_content_budget": runtime.chunk_content_budget,

                    # Metadata gốc của Docling.
                    "docling_metadata": docling_metadata,
                }

                jsonl_file.write(
                    json.dumps(chunk_record, ensure_ascii=False)
                )
                jsonl_file.write("\n")

                final_token_counts.append(prepared.final_token_count)

                print(
                    f"Chunk {final_chunk_index:04d} | "
                    f"Docling={docling_chunk_index:04d} | "
                    f"Part={subchunk_index + 1}/{len(prepared_parts)} | "
                    f"Token={prepared.final_token_count}/"
                    f"{runtime.model_max_tokens} | "
                    f"Chars={len(prepared.contextualized_text)}"
                )

                final_chunk_index += 1

    if final_token_counts:
        stats = {
            "output_file": str(output_file.resolve()),
            "docling_chunks": docling_chunk_count,
            "final_chunks": final_chunk_index,
            "fallback_split_chunks": fallback_split_count,
            "min_tokens": min(final_token_counts),
            "max_tokens": max(final_token_counts),
            "average_tokens": round(statistics.mean(final_token_counts), 2),
            "model_max_tokens": runtime.model_max_tokens,
            "chunk_content_budget": runtime.chunk_content_budget,
        }
    else:
        stats = {
            "output_file": str(output_file.resolve()),
            "docling_chunks": docling_chunk_count,
            "final_chunks": 0,
            "fallback_split_chunks": fallback_split_count,
            "min_tokens": 0,
            "max_tokens": 0,
            "average_tokens": 0,
            "model_max_tokens": runtime.model_max_tokens,
            "chunk_content_budget": runtime.chunk_content_budget,
        }

    return stats


# ============================================================
# KIỂM TRA LẠI FILE JSONL TRƯỚC KHI EMBEDDING
# ============================================================

def validate_jsonl_token_limits(
    jsonl_file: Path,
    tokenizer: PreTrainedTokenizerBase,
    model_max_tokens: int,
) -> None:
    """
    Đọc lại toàn bộ JSONL và kiểm tra từng embedding_text.

    Đây là lớp bảo vệ cuối cùng trước khi ingestion vào Qdrant.
    Nếu có bất kỳ record nào vượt giới hạn, hàm sẽ raise lỗi.
    """

    violations: list[str] = []

    with jsonl_file.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            embedding_text = record["embedding_text"]

            actual_count = count_tokens(
                tokenizer,
                embedding_text,
                add_special_tokens=True,
            )

            stored_count = record.get("embedding_token_count")

            if stored_count != actual_count:
                violations.append(
                    f"Dòng {line_number}: token lưu={stored_count}, "
                    f"token thực tế={actual_count}."
                )

            if actual_count > model_max_tokens:
                violations.append(
                    f"Dòng {line_number}: vượt giới hạn "
                    f"{actual_count} > {model_max_tokens}."
                )

    if violations:
        details = "\n".join(violations[:20])
        raise RuntimeError(
            "File chunks JSONL không đạt kiểm tra token:\n" + details
        )

    print("Kiểm tra JSONL thành công: không có chunk vượt giới hạn token.")


# ============================================================
# PIPELINE XỬ LÝ MỘT FILE
# ============================================================

def process_document(
    source_file: Path,
    output_dir: Path,
    tokenizer_model_id: str,
    model_max_tokens: int,
    embedding_prefix: str,
    safety_margin: int,
    fallback_overlap_tokens: int,
) -> None:
    """
    Pipeline hoàn chỉnh:

    1. Kiểm tra file đầu vào.
    2. Chuyển file thành DoclingDocument.
    3. Xuất Markdown, JSON và TXT để kiểm tra.
    4. Tạo tokenizer đúng với embedding model.
    5. Tính ngân sách token dành cho chunk.
    6. Tạo chunk bằng HybridChunker.
    7. Kiểm tra/hard-split để bảo đảm không vượt giới hạn.
    8. Lưu chunks thành JSONL.
    9. Đọc lại JSONL và kiểm tra lần cuối.
    """

    if not source_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {source_file}")

    if not source_file.is_file():
        raise ValueError(f"Đường dẫn không phải file: {source_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Đang xử lý: {source_file}")
    print("=" * 80)

    # Bước 1: chuyển tài liệu thành DoclingDocument.
    converter = DocumentConverter()
    conversion_result = converter.convert(source_file)
    document = conversion_result.document

    # Bước 2: xuất các phiên bản để quan sát kết quả trích xuất.
    print("Đang xuất Markdown, JSON và TXT...")
    export_document_versions(
        document=document,
        source_file=source_file,
        output_dir=output_dir,
    )

    # Bước 3: tạo tokenizer/chunker với budget an toàn.
    runtime = create_chunking_runtime(
        tokenizer_model_id=tokenizer_model_id,
        model_max_tokens=model_max_tokens,
        embedding_prefix=embedding_prefix,
        safety_margin=safety_margin,
    )

    # Bước 4: tạo chunks JSONL.
    print("Đang tạo chunks...")
    stats = export_chunks_to_jsonl(
        document=document,
        source_file=source_file,
        output_dir=output_dir,
        runtime=runtime,
        embedding_prefix=embedding_prefix,
        safety_margin=safety_margin,
        fallback_overlap_tokens=fallback_overlap_tokens,
    )

    # Bước 5: kiểm tra lại file vừa tạo.
    chunks_file = output_dir / f"{source_file.stem}.chunks.jsonl"
    validate_jsonl_token_limits(
        jsonl_file=chunks_file,
        tokenizer=runtime.hf_tokenizer,
        model_max_tokens=runtime.model_max_tokens,
    )

    print("\nTHỐNG KÊ CHUNK:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nĐầu ra: {output_dir.resolve()}")


# ============================================================
# THAM SỐ DÒNG LỆNH
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Ví dụ chạy:

    python docling_chunk_pipeline_safe.py \
        ./document/fabric_warehouse_document.docx \
        --output ./outputs \
        --model-max-tokens 512 \
        --safety-margin 8
    """

    parser = argparse.ArgumentParser(
        description=(
            "Chuyển tài liệu bằng Docling và tạo chunk có giới hạn token."
        )
    )

    parser.add_argument(
        "source",
        type=Path,
        help="Đường dẫn file cần xử lý.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./outputs"),
        help="Thư mục đầu ra. Mặc định: ./outputs",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER_ID,
        help="Tokenizer tương ứng với embedding model.",
    )

    parser.add_argument(
        "--model-max-tokens",
        type=int,
        default=DEFAULT_MODEL_MAX_TOKENS,
        help="Giới hạn token cứng của embedding model.",
    )

    parser.add_argument(
        "--embedding-prefix",
        type=str,
        default=DEFAULT_DOCUMENT_PREFIX,
        help="Prefix dùng cho tài liệu khi embedding.",
    )

    parser.add_argument(
        "--safety-margin",
        type=int,
        default=DEFAULT_SAFETY_MARGIN,
        help="Số token dự phòng dưới giới hạn model.",
    )

    parser.add_argument(
        "--fallback-overlap-tokens",
        type=int,
        default=DEFAULT_FALLBACK_OVERLAP_TOKENS,
        help="Overlap token khi phải hard-split dự phòng.",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    # args = parse_arguments()

    # try:
    #     process_document(
    #         source_file=args.source,
    #         output_dir=args.output,
    #         tokenizer_model_id=args.tokenizer,
    #         model_max_tokens=args.model_max_tokens,
    #         embedding_prefix=args.embedding_prefix,
    #         safety_margin=args.safety_margin,
    #         fallback_overlap_tokens=args.fallback_overlap_tokens,
    #     )

    #     return 0

    # except Exception as exc:
    #     print(
    #         f"Lỗi khi xử lý tài liệu: {exc}",
    #         file=sys.stderr,
    #     )

    #     return 1

    file_path = Path("document/farbic_warehouse_document.docx")

    try:
        process_document(
            source_file=file_path,
            output_dir=Path("./outputs"),
            tokenizer_model_id=DEFAULT_TOKENIZER_ID,
            model_max_tokens=DEFAULT_MODEL_MAX_TOKENS,
            embedding_prefix=DEFAULT_DOCUMENT_PREFIX,
            safety_margin=DEFAULT_SAFETY_MARGIN,
            fallback_overlap_tokens=DEFAULT_FALLBACK_OVERLAP_TOKENS,
        )

        return 0

    except Exception as exc:
        print(
            f"Lỗi khi xử lý tài liệu: {exc}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
