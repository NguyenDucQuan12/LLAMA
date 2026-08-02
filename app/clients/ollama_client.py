from __future__ import annotations

"""
HTTP client dùng chung để gọi Ollama.

Module này cung cấp hai chức năng:
1. Tạo embedding theo batch bằng `/api/embed`.
2. Gọi Llama bằng `/api/chat` và yêu cầu JSON có cấu trúc.
"""

import asyncio
import json
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

# Mở comment 3 dòng bên dưới mỗi khi test (Chạy trực tiếp hàm if __main__)
import os,sys
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from config import Settings, get_settings


logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client bất đồng bộ có timeout, retry và kiểm tra output.
    """
    # Các mã lỗi HTTP tạm thời có thể retry.
    retryable_status_codes: set[int] = {
        408,
        429,
        500,
        502,
        503,
        504,
    }

    def __init__(self, settings: Settings) -> None:
        # khai báo các biến cấu hình để tránh gọi settings nhiều lần.
        self.settings = settings
        # Tạo HTTP client bất đồng bộ với timeout và base_url.
        self.http_client = httpx.AsyncClient(
            base_url=settings.ollama_base_url.rstrip("/"),
            timeout=httpx.Timeout(settings.ollama_timeout_seconds),
        )

    async def close(self) -> None:
        """
        Đóng connection pool của HTTP client.
        """

        await self.http_client.aclose()

    async def embed_texts( self, texts: Sequence[str]) -> list[list[float]]:
        """
        Tạo vector cho một danh sách text.

        Lưu ý:
        - Document phải có prefix `search_document: `.
        - Query phải có prefix `search_query: `.
        - `truncate=False` để model báo lỗi nếu text quá dài thay vì cắt mất phần cuối một cách im lặng.

        Ví dụ đầu vào là các chunk:  
        ```python
        [
            "search_document: Nội dung chunk 1",
            "search_document: Nội dung chunk 2",
        ]
        ```
        Thì kết quả sẽ là:  
        ```python
        [
            [0.012, -0.08, ...],  # vector chunk 1
            [0.021, -0.03, ...],  # vector chunk 2
        ]
        ```
        """

        if not texts:
            return []

        normalized_texts: list[str] = []

        # Chuẩn hóa text: loại bỏ khoảng trắng đầu/cuối và kiểm tra rỗng.
        for index, text in enumerate(texts):
            # Kiểm tra kiểu dữ liệu.
            if not isinstance(text, str):
                raise TypeError(
                    f"Embedding input tại vị trí {index} phải là string."
                )
            # Loại bỏ khoảng trắng đầu/cuối.
            stripped_text = text.strip()
            # Sau khi loại bỏ khoảng trắng mà chuỗi không còn lý tự thì báo lỗi.
            if not stripped_text:
                raise ValueError(
                    f"Embedding input tại vị trí {index} đang rỗng."
                )

            normalized_texts.append(stripped_text)

        # Tạo request body cho Ollama theo cấu trúc API.
        request_body: dict[str, Any] = {
            "model": self.settings.embedding_model_name,
            "input": normalized_texts,
            "truncate": False,
            "keep_alive": self.settings.ollama_keep_alive,
            "dimensions": self.settings.embedding_vector_dimensions,
        }
        # Gửi request POST và retry nếu lỗi mạng hoặc lỗi server tạm thời.
        response_data = await self._post_json_with_retry(
            endpoint="/api/embed",
            request_body=request_body,
        )
        # Lấy kết quả các vector embedding từ response JSON.
        embeddings = response_data.get("embeddings")

        if not isinstance(embeddings, list):
            raise RuntimeError(
                "Ollama không trả về trường `embeddings` hợp lệ."
            )
        # Kiểm tra số lượng vector và chất lượng vector trước khi trả về.
        self._validate_embeddings(
            embeddings=embeddings,
            expected_count=len(normalized_texts),
        )

        return embeddings

    async def chat_json( self, messages: Sequence[Mapping[str, str]], json_schema: Mapping[str, Any],
                        model_name: str | None = None, temperature: float | None = None, ) -> dict[str, Any]:
        """
        Gọi `/api/chat` và bắt model trả JSON theo schema.

        Structured output giúp tầng ứng dụng không phải cố gắng tách JSON
        từ một đoạn văn tự do.
        """
        # Lấy tên model để gọi
        selected_model_name = (
            model_name
            if model_name is not None
            else self.settings.llama_model_name
        )

        selected_temperature = (
            temperature
            if temperature is not None
            else self.settings.llama_temperature
        )
        # Tạo request body cho Ollama theo cấu trúc API.
        # Cấu trúc của request như sau:
        # {
        #     "model": "llama2",
        #     "messages": [ {"role": "user", "content": "Hello"} ],
        #     "stream": False,
        #     "format": { "type": "object", "properties": { ... } },
        #     "keep_alive": True,
        #     "options": {
        #         "temperature": 0.7,
        #         "num_ctx": 2048,
        #         "num_predict": 512
        #     }
        # }
        # Format là JSON schema để yêu cầu model trả về JSON có cấu trúc. Ví dụ:
        # {
        #     "type": "object",
        #     "properties": {
        #         "summary": {"type": "string"},
        #         "keywords": {"type": "array", "items": {"type": "string"}}
        #     },
        #     "required": ["summary"]
        # }
        # sẽ yêu cầu model trả về JSON object có trường `summary` là string và `keywords` là array of string.
        request_body: dict[str, Any] = {
            "model": selected_model_name,
            "messages": [dict(message) for message in messages],
            "stream": False,
            "format": dict(json_schema),
            "keep_alive": self.settings.ollama_keep_alive,
            "options": {
                "temperature": selected_temperature,
                "num_ctx": self.settings.llama_context_window,
                "num_predict": self.settings.llama_max_generated_tokens,
            },
        }
        # Gửi request POST và retry nếu lỗi mạng hoặc lỗi server tạm thời.
        response_data = await self._post_json_with_retry(
            endpoint="/api/chat",
            request_body=request_body,
        )

        message = response_data.get("message")

        if not isinstance(message, Mapping):
            raise RuntimeError(
                "Ollama không trả về trường `message` hợp lệ."
            )

        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Llama trả về nội dung rỗng.")

        # Tiến hành parse JSON từ content. Nếu không parse được thì báo lỗi.
        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError as exception:
            raise RuntimeError(
                "Llama không trả về JSON hợp lệ theo structured output."
            ) from exception

        if not isinstance(parsed_content, dict):
            raise RuntimeError("Structured output phải là một JSON object.")

        return parsed_content

    async def _post_json_with_retry(self, endpoint: str, request_body: Mapping[str, Any]) -> dict[str, Any]:
        """
        Gửi POST request và retry lỗi mạng hoặc lỗi server tạm thời.

        Các lỗi cấu hình như HTTP 400 hoặc 404 không được retry vì retry
        không thể sửa được model sai tên hoặc request sai cấu trúc.
        """

        last_exception: Exception | None = None

        for attempt_number in range( self.settings.ollama_max_retries + 1):
            try:
                # Gọi api
                response = await self.http_client.post(
                    endpoint,
                    json=dict(request_body),
                )
                
                # Nếu Ollama trả lỗi tạm thời thì raise HTTPStatusError để retry.
                if response.status_code in self.retryable_status_codes:
                    raise httpx.HTTPStatusError(
                        message=(
                            "Ollama trả lỗi tạm thời: "
                            f"HTTP {response.status_code}"
                        ),
                        request=response.request,
                        response=response,
                    )
                # Nếu Ollama trả lỗi cấu hình thì raise RuntimeError để không retry.
                response.raise_for_status()

                response_json = response.json()

                if not isinstance(response_json, dict):
                    raise RuntimeError(
                        "Ollama response không phải JSON object."
                    )

                return response_json

            except ( httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.HTTPStatusError, ) as exception:
                last_exception = exception
                # Nếu Ollama trả lỗi HTTP không retryable thì raise RuntimeError để không retry.
                if isinstance(exception, httpx.HTTPStatusError):
                    status_code = exception.response.status_code

                    if status_code not in self.retryable_status_codes:
                        response_body = exception.response.text[:2000]

                        raise RuntimeError(
                            "Không thể gọi Ollama do lỗi request hoặc cấu hình. "
                            f"HTTP {status_code}: {response_body}"
                        ) from exception
                # Nếu vượt quá số lần retry thì break vòng lặp và raise RuntimeError.
                if attempt_number >= self.settings.ollama_max_retries:
                    break
                
                # Tạo một khoảng thời gian delay theo exponential backoff để tránh spam request liên tục.
                delay_seconds = min(2 ** attempt_number, 20)

                logger.warning(
                    "Ollama tạm thời không sẵn sàng. Thử lại sau %s giây. "
                    "Lần thử %s/%s.",
                    delay_seconds,
                    attempt_number + 1,
                    self.settings.ollama_max_retries,
                )

                await asyncio.sleep(delay_seconds)

        raise RuntimeError(f"Không thể kết nối Ollama sau {self.settings.ollama_max_retries} lần retry.") from last_exception

    def _validate_embeddings(
        self,
        embeddings: Any,
        expected_count: int,
    ) -> None:
        """
        Kiểm tra vector sau khi nhận được từ model embedding, và trước khi đưa các vector này vào Qdrant.

        Các kiểm tra gồm:
        - Số vector phải bằng số input.
        - Mỗi vector phải đúng dimension.
        - Không có NaN hoặc Infinity.
        - Norm phải gần 1 vì Ollama trả embedding đã L2-normalized.
        """
        # Kiểm tra số lượng vector.
        if len(embeddings) != expected_count:
            raise RuntimeError(
                f"Số vector không khớp số embedding input: {len(embeddings)} != {expected_count}."
            )

        # Kiểm tra từng vector.
        for vector_index, vector in enumerate(embeddings):
            # Nếu vector không phải list thì raise TypeError. Vì Ollama trả về embedding là list of float.
            if not isinstance(vector, list):
                raise TypeError(
                    f"Vector tại vị trí {vector_index} không phải list."
                )

            # Kiểm tra dimension của vector. Mỗi collection trong Qdrant yêu cầu dimension cố định. Nếu vector không đúng dimension thì raise RuntimeError.
            if len(vector) != self.settings.embedding_vector_dimensions:
                raise RuntimeError(
                    f"Vector {vector_index} có dimension {len(vector)}, nhưng collection yêu cầu {self.settings.embedding_vector_dimensions}."
                )

            # Khai báo một list mới để chứa các giá trị float đã chuẩn hóa. Nếu vector có NaN hoặc Infinity thì raise RuntimeError.
            numeric_vector: list[float] = []

            for value in vector:
                # Kiểm tra từng giá trị trong vector. Nếu giá trị không phải số thì raise TypeError. Nếu giá trị là NaN hoặc Infinity thì raise RuntimeError.
                if not isinstance(value, (int, float)):
                    raise TypeError(
                        f"Vector {vector_index} chứa giá trị không phải số."
                    )
                # Ép kiểu về float
                numeric_value = float(value)
                # Kiểm tra NaN hoặc Infinity. Nếu có thì raise RuntimeError.
                if not math.isfinite(numeric_value):
                    raise RuntimeError(
                        f"Vector {vector_index} chứa NaN hoặc Infinity."
                    )

                numeric_vector.append(numeric_value)
            # Tính norm của vector. Ollama trả embedding đã L2-normalized, nên norm phải gần 1. Nếu norm không trong khoảng [0.97, 1.03] thì raise RuntimeError.
            vector_norm = math.sqrt(
                sum(value * value for value in numeric_vector)
            )

            if not 0.97 <= vector_norm <= 1.03:
                raise RuntimeError(
                    f"Vector {vector_index} có norm bất thường: "
                    f"{vector_norm:.6f}."
                )

if __name__ == "__main__":


    """
    Đối với format trong chat api:  
    Nó là trường trong API Ollama dùng để yêu cầu model trả về structured output.  
    Model llama sẽ trả về trường `message` chứa nội dung câu trả lời của model.  
    - Nếu ta không sử dụng format, model sẽ trả về message.content là chuỗi văn bản tự do. Không chắc nó là Json
    Request:
    {
    "model": "llama3.1:8b",
    "messages": [
        {
        "role": "user",
        "content": "Tóm tắt tài liệu kho hàng."
        }
    ],
    "stream": false
    }
    thì response từ model:
    {
    "message": {
        "role": "assistant",
        "content": "Tài liệu mô tả quy trình nhập và xuất kho..."
    }
    }
    - Nếu sử dụng `format: "json"`  
    Request:
    {
        "model": "llama3.1:8b",
        "messages": [
            {
            "role": "user",
            "content": "Tóm tắt tài liệu kho hàng."
            }
        ],
        "stream": false,
        "format": "json"
    }
    thì model được yêu cầu trả json như sau
    {
        "summary": "Tài liệu mô tả quy trình nhập kho."
    }
    hoặc
    {
        "result": "Tài liệu mô tả quy trình nhập kho.",
        "topic": "warehouse"
    }
    Cả hai cách trả trên đều có thể đúng vì ta chưa quy định cụ thể các trường mà model sẽ trả về, chỉ quy định trả về phải bắt buộc là json
    - Nếu sử dụng format với json schema
    Request:
    {
        "model": "llama3.1:8b",
        "messages": [
            {
            "role": "user",
            "content": "Tóm tắt tài liệu kho hàng."
            }
        ],
        "stream": false,
        "format": {
            "type": "object",
            "properties": {
            "summary": {
                "type": "string"
            },
            "keywords": {
                "type": "array",
                "items": {
                "type": "string"
                }
            }
            },
            "required": [
            "summary",
            "keywords"
            ],
            "additionalProperties": false
        }
    }
    Khi đó model sẽ trả về json và các trường được yêu cầu:
    {
        "summary": "Tài liệu mô tả quy trình nhập và xuất kho.",
        "keywords": [
            "nhập kho",
            "xuất kho",
            "AGV"
        ]
    }
    Trong đó với format như trên ta có:
    + type: object    thì kết quả trả về phải là 1 object
    + properties là khai báo các trường được yêu cầu
    + summary.type: string là trường summary phải trả về chuỗi
    + keywords.type: array    thì trường keywords phải là danh sách
    + required là danh sách các trường bắt buộc phải có
    + additionalProperties: false có nghĩa là không cho phép model tạo thêm các trường khác ngoài các trường đã khai báo sẵn


    Tuy nhiên Ollama vẫn sẽ trả như sau:  
    {
        "model": "llama3.1:8b",
        "created_at": "...",
        "message": {
            "role": "assistant",
            "content": "{\"summary\":\"Tài liệu mô tả quy trình kho hàng.\"}"
        },
        "done": true,
        "total_duration": 123456789,
        "prompt_eval_count": 40,
        "eval_count": 20
    }
    Trong đó 
    message.content là CHUỖI CHỨA JSON, do đó ta cần xử lý theo 2 bước
    Bước 1: lấy chuỗi từ kết quả trả về của llama
    message = response_data.get("message")
    content = message.get("content")
    Bước 2: parse json từ chuỗi trên
    parsed_content = json.loads(content)
    Để cuối cùng cho ra kết quả là json theo đúng ý ta
    """
    # Mẫu format
    rag_answer_schema = {
        "type": "object",

        "properties": {
            "answer": {
                "type": "string",
                "description": (
                    "Câu trả lời dựa trên tài liệu."
                ),
            },

            "insufficient_context": {
                "type": "boolean",
                "description": (
                    "True nếu tài liệu không đủ thông tin."
                ),
            },

            "citations": {
                "type": "array",

                "items": {
                    "type": "object",

                    "properties": {
                        "source_label": {
                            "type": "string",
                        },

                        "evidence": {
                            "type": "string",
                        },
                    },

                    "required": [
                        "source_label",
                        "evidence",
                    ],

                    "additionalProperties": False,
                },
            },
        },

        "required": [
            "answer",
            "insufficient_context",
            "citations",
        ],

        "additionalProperties": False,
    }

    # Kết quả như sau:
    """
    {
        "answer": "Nhân viên cần chọn vị trí và nhấn nút gọi robot.",
        "insufficient_context": false,
        "citations": [
            {
            "source_label": "D1",
            "evidence": "Tài liệu D1 mô tả bước chọn vị trí và gọi robot."
            }
        ]
    }
    """

    async def test_embed(
        client: OllamaClient,
    ) -> None:
        """
        Kiểm tra chức năng embedding.
        """

        texts = [
            (
                "search_document: "
                "Đây là một tài liệu về kho hàng."
            ),
            (
                "search_query: "
                "Tìm các quy trình nhập kho."
            ),
        ]

        embeddings = await client.embed_texts(texts)

        print("Số vector:", len(embeddings))

        for index, vector in enumerate(embeddings):
            print(
                f"Vector {index}: "
                f"{len(vector)} chiều"
            )


    async def test_chat(
        client: OllamaClient,
    ) -> None:
        """
        Kiểm tra chức năng chat structured output.
        """

        messages = [
            {
                "role": "user",
                "content": (
                    "Hãy tóm tắt nội dung sau: "
                    "Đây là một tài liệu về kho hàng."
                ),
            }
        ]

        json_schema = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                }
            },
            "required": [
                "summary",
            ],

            # Không cho phép model tự thêm trường khác.
            "additionalProperties": False,
        }

        response = await client.chat_json(
            messages=messages,
            json_schema=json_schema,
        )

        print(
            "Chat JSON Response:",
            json.dumps(
                response,
                ensure_ascii=False,
                indent=2,
            ),
        )


    async def main() -> None:
        """
        Chỉ có một event loop duy nhất.

        HTTP client được tạo, sử dụng và đóng
        trong cùng event loop này.
        """

        settings = get_settings()

        client = OllamaClient(settings)

        try:
            # Có thể chạy cả hai bài kiểm tra.
            await test_embed(client)
            await test_chat(client)

        finally:
            # Luôn đóng HTTP connection pool,
            # kể cả khi embed hoặc chat phát sinh lỗi.
            await client.close()


    # Chỉ gọi asyncio.run đúng một lần.
    asyncio.run(main())