Ta sử dụng `Docling` để tạo chunk cho tệp văn bản, pdf, excel, ... mà ta cần embedding. Đây là tất cả các tài liệu chứa câu trả lời cho những câu hỏi của người dùng.  

Cấu trúc của 1 chunk trong `Docling`. 
```bash
chunk
├── text
└── meta
    ├── headings
    ├── captions
    ├── doc_items
    └── origin
```
Trong đó:  
- Text: Là nội dung chính của chunk
- Headings: Là các tiêu đề cha của chunk  
- Captions: Là chú thích đây là bảng hoặc hình ảnh  
- Doc_items: Là các phần tử gốc tạo ra chunk  
- Origin: Là thông tin các tài liệu nguồn   

# Token
Token là đơn vị văn bản mà model thực sự xử lý. Một token không nhất thiết là:  

- Một từ.
- Một ký tự.
- Một âm tiết.
- Một từ tiếng Việt hoàn chỉnh.

Token có thể là:  
```bash
Một từ:
"robot"

Một phần của từ:
"submit" → "sub" + "mit"

Dấu câu:
","
"."
":"

Khoảng trắng đi kèm từ:
"▁robot"

Một chuỗi kỹ thuật:
"usp_wms_custom_in_submit"
```
Các tokenizer hiện đại thường chia văn bản thành các subword, tức là đơn vị nằm giữa từ và ký tự. Từ phổ biến có thể được giữ nguyên, còn từ lạ hoặc từ kỹ thuật có thể bị chia thành nhiều phần nhỏ. Ví dụ:  
```bash
Từ phổ biến:
"warehouse"
→ ["warehouse"]

Từ ít phổ biến:
"customization"
→ ["custom", "ization"]

Tên stored procedure:
"usp_wms_custom_in_submit"
→ ["usp", "_wms", "_custom", "_in", "_submit"]
```

# Tokenizer
Tokenizer là bộ tiền xử lý chuyển văn bản thành đầu vào mà model hiểu được. Tokenizer thường thực hiện các bước:  
```bash
Văn bản gốc
    │
    ├── Chuẩn hóa ký tự
    ├── Phân tách thành token/subword
    ├── Chuyển token thành ID
    ├── Thêm special token nếu cần
    ├── Cắt bớt nếu quá dài
    └── Tạo attention mask
```

`Hugging Face` mô tả `tokenizer` là thành phần chuẩn bị đầu vào cho model, bao gồm phân tách văn bản, ánh xạ token sang ID và quản lý các special token.  

Model không trực tiếp đưa các đoạn văn bản gốc vào model embedding, vì chúng không xử lý trực tiếp các chuỗi đó mà sẽ thông qua các con số, gọi là id.  

Ví dụ với đoạn văn: `"Robot AGV không chạy"`. Trước tiên, tokenizer biến nó thành các mảnh và ID:  
```bash
Văn bản:
"Robot AGV không chạy"

Token, minh họa:
["▁Robot", "▁AGV", "▁không", "▁chạy"]

Token ID, minh họa:
[18372, 42641, 923, 8172]
```
Với mỗi tokenize của từng model sẽ cho ra một kết quả khác nhau, vì vậy ta cần sử dụng các tokenize tương ứng với model đó.  
Sau khi có các id này thì Các model embedding mới chuyển đổi từng id thành các vecto tương ứng, ví dụ với id 18372.  
```bash
Token ID 18372
    ↓
[0.021, -0.138, 0.512, ...]
```
Như thế cả câu sẽ được chuyển thành vectoc sau khi qua mô hình embedding như sau:  
```bash
"Robot AGV không chạy"
    ↓
Tokenizer
    ↓
[18372, 42641, 923, 8172]
    ↓
Embedding model
    ↓
[0.012, -0.084, 0.156, ..., 0.043]
```

Mỗi tokenizer tương ứng với từng model embedding của nó, không được sử dụng tokenizer của model khác để xử lý cho model này.  
Ví dụ với model embedding `nomic-embed-text-v2-moe` thì ta có tokenizer tương ứng ` nomic-ai/nomic-embed-text-v2-moe`  
Mỗi model có cách thức chuẩn hoá văn bản, cách chia token khác nhau nên ta không được phép sử dụng lẫn lộn.  

# Prefix
Model `nomic-embed-text-v2-moe` yêu cầu 2 prefix.  

Đối với các chunk tài liệu, model yêu cầu thêm:  
```bash
search_document:
```
Đối với câu hỏi của người dùng, model yêu cầu thêm:
```bash
search_query:
```

Ví dụ:  
```bash
question = "Làm thế nào để submit Return In?"

embedding_input = (
    "search_query: "
    + question
)
```
thì embedding_input ta nhận được sẽ như sau:  
```bash
search_query: Làm thế nào để submit Return In?
```
Điều này để cho model biết được rằng vai trog của từng câu. Nếu câu hỏi đến từ người dùng thì sẽ có từ kháo `search_query`, còn các đoạn tài liệu dùng để trả lời câu hỏi sẽ bắt đầu từ `search_document`.  


# Tạo chunk
`Docling` sẽ tạo chunk như sau:  
```bash
DOCX/PDF
   │
   ▼
DocumentConverter
   │
   ▼
DoclingDocument
   │
   ▼
HybridChunker
   │
   ├── Chia theo heading, đoạn, bảng
   ├── Chia chunk quá dài
   └── Gộp chunk nhỏ cùng cấu trúc
   │
   ▼
chunk.text
   │
   ▼
chunker.contextualize(chunk)
   │
   ▼
Thêm "search_document: "
   │
   ▼
Đếm token toàn bộ chuỗi
   │
   ├── <= 512 → chấp nhận
   │
   └── > 512  → hard-split dự phòng
   │
   ▼
Kiểm tra lại lần cuối
   │
   ▼
Lưu JSONL
   │
   ▼
Embedding
```
Ví dụ 1 chunk:  
```bash
{
  "id": "0a678c4f-8049-50a8-aba7-bf6a506d9979",
  "chunk_index": 20,
  "docling_chunk_index": 19,
  "subchunk_index": 0,
  "subchunk_count": 1,
  "source_file": "Fabric_Inventory_Manual.pdf",
  "source_path": "/Users/ducquan/Project/Test/document/Fabric_Inventory_Manual.pdf",
  "source_hash": "9b38ea477bd867183f004970b7f31c980fea20697a5fbdd6fbc9ba45959c6351",
  "page_numbers": [34],
  "text": "- Kiểm tra tồn kho theo CUỘN (chi tiết )\n- Nút 'Xuất ra Excel': xuất dữ liệu ra theo file excel Các thông tin chính:\n- -Số cuộn và số lượng đến theo packing list\n- -Số cuộn và số lượng nhập kho\n- -Số cuộn và số lượng xuất kho\n- -Số cuộn và số lượng tồn kho",
  "contextualized_text": "21. BẢNG CHI TIẾT HÀNG ĐẾN XUẤT NHẬP TỒN\n- Kiểm tra tồn kho theo CUỘN (chi tiết )\n- Nút 'Xuất ra Excel': xuất dữ liệu ra theo file excel Các thông tin chính:\n- -Số cuộn và số lượng đến theo packing list\n- -Số cuộn và số lượng nhập kho\n- -Số cuộn và số lượng xuất kho\n- -Số cuộn và số lượng tồn kho",
  "embedding_text": "search_document: 21. BẢNG CHI TIẾT HÀNG ĐẾN XUẤT NHẬP TỒN\n- Kiểm tra tồn kho theo CUỘN (chi tiết )\n- Nút 'Xuất ra Excel': xuất dữ liệu ra theo file excel Các thông tin chính:\n- -Số cuộn và số lượng đến theo packing list\n- -Số cuộn và số lượng nhập kho\n- -Số cuộn và số lượng xuất kho\n- -Số cuộn và số lượng tồn kho",
  "embedding_token_count": 116,
  "embedding_model_max_tokens": 512,
  "chunk_content_budget": 498,
  "docling_metadata": {
    "schema_name": "docling_core.transforms.chunker.DocMeta",
    "version": "1.0.0",
    "doc_items": [
      {
        "self_ref": "#/texts/2952",
        "parent": {
          "$ref": "#/groups/23"
        },
        "children": [],
        "content_layer": "body",
        "label": "list_item",
        "prov": [
          {
            "page_no": 34,
            "bbox": {
              "l": 713.35,
              "t": 484.72,
              "r": 926.0100000000002,
              "b": 445.11400000000003,
              "coord_origin": "BOTTOMLEFT"
            },
            "charspan": [0, 38]
          }
        ]
      },
      {
        "self_ref": "#/texts/2953",
        "parent": {
          "$ref": "#/groups/23"
        },
        "children": [],
        "content_layer": "body",
        "label": "list_item",
        "prov": [
          {
            "page_no": 34,
            "bbox": {
              "l": 713.35,
              "t": 441.49,
              "r": 926.8140000000001,
              "b": 380.27,
              "coord_origin": "BOTTOMLEFT"
            },
            "charspan": [0, 75]
          }
        ]
      },
      {
        "self_ref": "#/texts/2954",
        "parent": {
          "$ref": "#/groups/23"
        },
        "children": [],
        "content_layer": "body",
        "label": "list_item",
        "prov": [
          {
            "page_no": 34,
            "bbox": {
              "l": 713.35,
              "t": 376.67,
              "r": 902.1300000000001,
              "b": 337.04,
              "coord_origin": "BOTTOMLEFT"
            },
            "charspan": [0, 42]
          }
        ]
      },
      {
        "self_ref": "#/texts/2955",
        "parent": {
          "$ref": "#/groups/23"
        },
        "children": [],
        "content_layer": "body",
        "label": "list_item",
        "prov": [
          {
            "page_no": 34,
            "bbox": {
              "l": 713.35,
              "t": 333.44,
              "r": 910.7520000000001,
              "b": 293.83399999999995,
              "coord_origin": "BOTTOMLEFT"
            },
            "charspan": [0, 29]
          }
        ]
      },
      {
        "self_ref": "#/texts/2956",
        "parent": {
          "$ref": "#/groups/23"
        },
        "children": [],
        "content_layer": "body",
        "label": "list_item",
        "prov": [
          {
            "page_no": 34,
            "bbox": {
              "l": 713.35,
              "t": 290.22,
              "r": 905.1899999999999,
              "b": 250.62,
              "coord_origin": "BOTTOMLEFT"
            },
            "charspan": [0, 29]
          }
        ]
      },
      {
        "self_ref": "#/texts/2957",
        "parent": {
          "$ref": "#/groups/23"
        },
        "children": [],
        "content_layer": "body",
        "label": "list_item",
        "prov": [
          {
            "page_no": 34,
            "bbox": {
              "l": 713.35,
              "t": 247.0,
              "r": 898.7460000000001,
              "b": 207.39999999999998,
              "coord_origin": "BOTTOMLEFT"
            },
            "charspan": [0, 28]
          }
        ]
      }
    ],
    "headings": [
      "21. BẢNG CHI TIẾT HÀNG ĐẾN XUẤT NHẬP TỒN"
    ],
    "origin": {
      "mimetype": "application/pdf",
      "binary_hash": 18143237381864317777,
      "filename": "Fabric_Inventory_Manual.pdf"
    }
  }
}
```
Ta có:  
- id: Mã UUID ta tự tạo làm Qdrant point ID. 
- chunk_index: Vị trí của chunk này sau khi tạo ra
- docling_chunk_index: Vị trí chunk gốc do Docling tạo ra
- sub_chunk_index: Nếu chunk quá dài, nó sẽ bị chia ra thành nhiều chunk, và tham số này sẽ là vị trí của chunk bị chia ra
- sub_chunk_count: Là tổng chunk bị chia ra từ chunk gốc, 1 sẽ là không bị chia ra
- text: Nội dung văn bản gốc, nó chưa chắc đã chứa các heading, ....
- contextualized_text: Nội dung văn abnr bổ sung ngữ cảnh cho text, đây sẽ là nội dung chính mà ta chuyển nó thành vecto để lưu trữ cho việc truy xuất
- embedding_text: Nội dung văn bản đã thêm prefix để sẵn sàng cho embedding
- embedding_token_count: Số token chính xác của emmebedding_text, có tính cả special token
- metadata: Chứa các thông tin xác định nguônf gốc của text