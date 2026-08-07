from __future__ import annotations

"""
Cấu hình tập trung của toàn bộ dự án.

Mọi giá trị nhạy cảm hoặc có thể thay đổi giữa môi trường phát triển,
kiểm thử và production đều được đọc từ biến môi trường hoặc file .env.

Không nên ghi trực tiếp mật khẩu SQL Server, API key hoặc địa chỉ server
vào mã nguồn.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Đại diện cho toàn bộ cấu hình chạy của ứng dụng.

    `SettingsConfigDict` cho phép tự động đọc file `.env` nằm ở thư mục
    gốc của dự án. Tên biến môi trường không phân biệt chữ hoa/chữ thường.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Thông tin chung của ứng dụng
    # ------------------------------------------------------------------
    application_name: str = "Production RAG SQL Server"
    application_environment: str = "development"
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    # Model dùng để tạo vector cho tài liệu và câu hỏi.
    embedding_model_name: str = "nomic-embed-text-v2-moe:latest"
    # Tokenizer Hugging Face phải tương ứng với embedding model.
    embedding_tokenizer_name: str = "nomic-ai/nomic-embed-text-v2-moe"
    # Kích thước vector được lưu trong Qdrant.
    embedding_vector_dimensions: int = 768
    # Giới hạn đầu vào của embedding model.
    embedding_model_max_tokens: int = 512

    # Prefix được model Nomic yêu cầu cho tài liệu và câu hỏi.
    embedding_document_prefix: str = "search_document: "
    embedding_query_prefix: str = "search_query: "

    # Chừa một khoảng an toàn cho special token và thay đổi serialization.
    chunk_token_safety_margin: int = 8

    # Kích thước batch gửi đến endpoint embedding.
    embedding_batch_size: int = 16
    embedding_batch_total_token_limit: int = 6000

    # Model Llama dùng để chọn SQL intent và sinh câu trả lời cuối cùng.
    llama_model_name: str = "llama3.1:8b"
    llama_context_window: int = 8192
    llama_max_generated_tokens: int = 1200
    llama_temperature: float = 0.1
    ollama_keep_alive: str = "30m"
    ollama_timeout_seconds: float = 6000.0
    ollama_max_retries: int = 3

    # ------------------------------------------------------------------
    # Qdrant
    # ------------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "wms_chunks_nomic_v2_768_v1"
    qdrant_upsert_batch_size: int = 128

    # Số tài liệu lấy ở bước truy xuất nhanh.
    retrieval_top_k: int = 20
    # Số tài liệu cuối cùng sau rerank được gửi cho Llama.
    rerank_top_k: int = 5

    # ------------------------------------------------------------------
    # Reranker
    # ------------------------------------------------------------------
    reranker_enabled: bool = True
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"      # "BAAI/bge-reranker-v2-m3"
    reranker_max_length: int = 512
    reranker_max_concurrency: int = 1

    reranker_apply_sigmoid: bool = True
    reranker_score_weight: float = 0.8
    dense_score_weight: float = 0.2
    reranker_duplicate_jaccard_threshold: float = 0.70
    reranker_allow_dense_fallback: bool = True


    # `auto` sẽ ưu tiên CUDA, sau đó MPS trên Mac, cuối cùng là CPU.
    reranker_device: str = "auto"
    reranker_batch_size: int = 4

    # Tỷ trọng kết hợp giữa điểm reranker và điểm dense retrieval.
    reranker_score_weight: float = 0.85
    dense_score_weight: float = 0.15

    # Cho phép trở về điểm Qdrant khi model reranker không tải được.
    reranker_allow_dense_fallback: bool = True

    # Ngưỡng dùng để loại các đoạn gần như giống hệt nhau.
    reranker_duplicate_jaccard_threshold: float = 0.92

    # ------------------------------------------------------------------
    # Context gửi cho Llama
    # ------------------------------------------------------------------
    answer_max_context_characters: int = 30000
    answer_max_characters_per_document: int = 7000

    # ------------------------------------------------------------------
    # SQL Server
    # ------------------------------------------------------------------
    # SQL bị tắt mặc định. Chỉ bật khi đã cấu hình tài khoản chỉ có quyền SELECT.
    sql_server_enabled: bool = True

    # Ví dụ dùng ODBC Driver 18:
    # DRIVER={ODBC Driver 18 for SQL Server};
    # SERVER=127.0.0.1,1433;
    # DATABASE=YourDatabase;
    # UID=rag_reader;
    # PWD=your_password;
    # Encrypt=yes;
    # TrustServerCertificate=yes;
    sql_server_odbc_connection_string: str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=172.24.48.159,25678;"
        "DATABASE=vietthien;"
        "UID=test_only;"
        "PWD=test;"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
    )

    sql_server_pool_size: int = 5
    sql_server_max_overflow: int = 5
    sql_server_command_timeout_seconds: int = 30

    # Số dòng tối đa một predefined query được phép trả về.
    sql_server_default_maximum_rows: int = 100

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Danh sách origin phân tách bằng dấu phẩy.
    cors_allowed_origins: str = "http://localhost:3000,http://localhost:8501"

    # ------------------------------------------------------------------
    # Đường dẫn dữ liệu
    # ------------------------------------------------------------------
    output_directory: str = "./outputs"

    def get_cors_allowed_origins(self) -> list[str]:
        """
        Chuyển chuỗi origin phân tách bằng dấu phẩy thành danh sách sạch.
        """

        origins: list[str] = []

        for item in self.cors_allowed_origins.split(","):
            normalized_item = item.strip()

            if normalized_item:
                origins.append(normalized_item)

        return origins


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Tạo duy nhất một object Settings trong mỗi process.

    `lru_cache` tránh đọc lại file `.env` ở mọi request.
    """

    return Settings()
