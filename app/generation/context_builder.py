from __future__ import annotations

"""
Biến top 5 document và kết quả SQL thành context có nhãn nguồn rõ ràng.
"""

import json
from dataclasses import dataclass
from typing import Any

# Mở comment 3 dòng bên dưới mỗi khi test (Chạy trực tiếp hàm if __main__)
import os,sys
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from config import Settings, get_settings
from retrieval.models import RerankedChunk
from schemas import SqlExecutionResponse


@dataclass(frozen=True)
class GenerationSource:
    """
    Cấu trúc một nguồn được gửi cho Llama.
    """

    source_label: str
    source_type: str
    content: str
    source_file: str | None
    chunk_index: int | None
    page_numbers: list[int]
    headings: list[str]


class AnswerContextBuilder:
    """
    Xây dựng context có giới hạn kích thước.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build_sources(self, document_chunks: list[RerankedChunk], sql_result: SqlExecutionResponse | None,) -> list[GenerationSource]:
        """
        Tạo danh sách nguồn D1... và S1...  
        D: Document  
        S: SQL  
        """

        # Khởi tạo list sources hợp lệ, và đếm tổng ký tự trong sources
        sources: list[GenerationSource] = []
        used_characters = 0

        # Duyệt qua các document chunk
        for document_index, chunk in enumerate(document_chunks, start=1):
            # Lấy contextualized và loại bỏ khoảng trắng thừa
            content = chunk.contextualized_text.strip()

            if not content:
                continue
            
            # Giới hạn độ dài cho một nguồn, và tính toán độ dài ký tự còn lại có thể sử dụng
            content = self._limit_single_source(content)
            remaining_characters = self.settings.answer_max_context_characters - used_characters

            # Nếu độ dài cho tổng đã hết thì không xử lý nữa
            if remaining_characters <= 0:
                break
            # Nếu sau khi giới hạn nhưng vẫn nhiều ký tự hơn số ký tự còn lại cho phép
            if len(content) > remaining_characters:
                # Nếu nội dung nhỏ hơn 300 ký tự thì không cần thêm vào, tránh nội dung quá ngắn, vô nghĩa, ...
                if remaining_characters < 300:
                    break
                
                # Tính toán số ký tự của chuỗi thêm vào 
                suffix = "\n[Phần cuối đã được cắt do giới hạn context.]"
                allowed_content_length = max(0,remaining_characters - len(suffix),)
                content = (content[:allowed_content_length].rstrip()+ suffix)

            # Thêm source này vào danh sách, D ký hiệu cho Document, S ký hiệu cho SQL
            sources.append(
                GenerationSource(
                    source_label=f"D{document_index}",   # Nếu D1 rỗng thì nó nhảy luôn chunk đầu tiên được tạo là D2
                    source_type="document",              # Muốn nhất quán thì sử dụng source_label = f"D{len(sources) + 1}"
                    content=content,
                    source_file=chunk.source_file,
                    chunk_index=chunk.chunk_index,
                    page_numbers=chunk.page_numbers,
                    headings=chunk.headings,
                )
            )
            # Cộng số lượng ký tự vào tổng số lượng ký tự đã sử dụng
            used_characters += len(content)
            
        """
        Hiện tại chỉ xử lý 1 SQL result, nếu có nhiều SQL result thì cần phải thay đổi cách đánh nhãn S1, S2, ...
        Và cần phải thay đổi cách tính toán ký tự còn lại, vì SQL result có thể chiếm nhiều ký tự hơn Document.
        """
        # Nếu model có sử dụng truy vấn SQL thì xử lý nó
        if (sql_result is not None   # SQL có kết quả
            and sql_result.executed  # Nó đã được thực thi
            # and sql_result.rows    # Đôi khi không có dữ liệu vẫn đúng, ví dụ hôm nay ko có nhiệm vụ nên sẽ ko có dữ liệu
            ):
            # Xử lý kết quả trả về từ SQL Server
            sql_content = self._format_sql_result(sql_result)
            sql_content = self._limit_single_source(sql_content)
            # Tính số lượng ký tự còn lại cho SQL, nếu Document đã chiếm hết không gian thì SQL có thể bị bỏ
            remaining_characters = self.settings.answer_max_context_characters - used_characters

            if remaining_characters >= 300:
                if len(sql_content) > remaining_characters:
                    # Tính toán số ký tự của chuỗi thêm vào 
                    suffix = "\n[Phần cuối của kết quả SQL đã được cắt.]"
                    allowed_content_length = max(0,remaining_characters - len(suffix),)
                    sql_content = (sql_content[:allowed_content_length].rstrip()+ suffix)

                sources.append(
                    GenerationSource(
                        source_label="S1",
                        source_type="sql",
                        content=sql_content,
                        source_file=None,
                        chunk_index=None,
                        page_numbers=[],
                        headings=[
                            sql_result.query_description
                            or sql_result.query_key
                            or "SQL result"
                        ],
                    )
                )

        return sources

    def build_context_text(self,sources: list[GenerationSource],) -> str:
        """
        Serialize nguồn thành các thẻ XML-like dễ phân biệt.  
        Nội dung này sẽ được gửi tới LLAMA  
        Ví dụ đầu vào:  
        [
            GenerationSource(source_label="D1", ...),
            GenerationSource(source_label="S1", ...),
        ]
        Thì đầu ra sẽ là:  
        <source label="D1" type="document">
        ...
        </source>

        <source label="S1" type="sql">
        ...
        </source>
        """
        # Khởi tạo danh sách blocks, mỗi nguồn sẽ được tạo thành 1 block
        blocks: list[str] = []

        for source in sources:
            # Tạo metadata để nhận biết nguồn này từ đâu
            metadata = {
                "source_label": source.source_label,
                "source_type": source.source_type,
                "source_file": source.source_file,
                "chunk_index": source.chunk_index,
                "page_numbers": source.page_numbers,
                "headings": source.headings,
            }
            # Thêm dữ liệu vào blocks, tương ứng với 1 block
            # <source label="D1" type="document">
            # <metadata>
            # {
            # "source_label": "D1"
            # }
            # </metadata>
            # <content>
            # Nội dung tài liệu
            # </content>
            # </source>
            # Giúp LLama dễ phân biệt các nguồn, khi viết prompt thì bảo AI sử dụng dữ liệu trong thẻ source
            blocks.append(
                f'<source label="{source.source_label}" '
                f'type="{source.source_type}">\n'
                "<metadata>\n"
                + json.dumps(
                    metadata,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n</metadata>\n"
                "<content>\n"
                + source.content
                + "\n</content>\n"
                "</source>"
            )
        # Nối chuỗi và trả về một chuỗi chứa toàn bộ source
        return "\n\n".join(blocks)

    def _format_sql_result( self, sql_result: SqlExecutionResponse, ) -> str:
        """
        Chuyển kết quả SQL thành JSON dễ đọc cho Llama.
        """

        payload: dict[str, Any] = {
            "query_key": sql_result.query_key,
            "description": sql_result.query_description,
            "parameters": sql_result.parameters,
            "row_count": sql_result.row_count,
            "rows": sql_result.rows,
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )

    def _limit_single_source(self, content: str) -> str:
        """
        Không để một nguồn chiếm toàn bộ context.
        """
        # Độ dài tối đa cho một nguồn, ví dụ 3000 ký tự, nếu vượt quá thì cắt bớt và thêm thông báo
        maximum_characters = self.settings.answer_max_characters_per_document

        if len(content) <= maximum_characters:
            return content
        
        suffix = "\n[Nguồn đã được cắt do giới hạn.]"
        # Tính toán số ký tự còn lại cho phép, nếu không còn thì trả về chuỗi rỗng
        allowed_length = max(0,maximum_characters - len(suffix),)

        # Trả về chuỗi được cắt bớt tại allowed_length và thêm suffix thông báo
        return (content[:allowed_length].rstrip()+ suffix)
    
if __name__ == "__main__":
    document_chunks = [
        RerankedChunk(
            point_id = "123iobsk482ndls",
            dense_score = 0.8,
            document_id = "239haksdh",
            source_hash = "39889sakjfn",
            doc_item_refs = ["bbbb"],
            text = "Nhất nút gọi robot",
            contextualized_text=(
                "1.3 Gọi Robot\n"
                "Nhấn nút Call Robot."
            ),
            payload = {},
            reranker_score = 1,
            final_score = 2,
            source_file="warehouse.docx",
            chunk_index=10,
            page_numbers=[],
            headings=["Nhận hàng", "Gọi Robot"],
        ),
        RerankedChunk(
            point_id = "yhgff3453453",
            dense_score = 0.8,
            document_id = "239jhgfhaksdh",
            source_hash = "t466eh7db53gfd",
            doc_item_refs = ["aaa"],
            text = "Kiểm tra robot",
            contextualized_text=(
                "8 Các lỗi\n"
                "Kiểm tra kết nối robot."
            ),
            payload = {},
            reranker_score = 1,
            final_score = 2,
            source_file="warehouse.docx",
            chunk_index=21,
            page_numbers=[],
            headings=["Các lỗi"],
        ),
    ]

    sql_result = SqlExecutionResponse(
        executed=True,
        query_key="get_robot_status",
        query_description="Trạng thái robot",
        parameters={"robot_id": "AGV-01"},
        row_count=1,
        rows=[
            {
                "robot_id": "AGV-01",
                "status": "IDLE",
            }
        ],
    )

    settings = get_settings()
    builder = AnswerContextBuilder(settings)

    sources = builder.build_sources(
        document_chunks=document_chunks,
        sql_result=sql_result,
    )

    context_text = builder.build_context_text(sources)

    print(context_text)