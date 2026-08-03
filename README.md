Xây dựng kiến trúc Hybrid Search
Cấu trúc này dùng để xử lý hai loại câu hỏi như sau:  
Lọai câu hỏi 1:  
Các câu hỏi có chứa mã kỹ thuật
```bash
return_in_muji lỗi ở đâu?
usp_wms_custom_in_submit thiếu điều kiện gì?
Pallet F3-29 bị trả sai vị trí vì sao?
```
Khi dữ liệu có nhiều mã kỹ thuật như `return_in_muji`, `J3-5`, `F3-29`, `usp_wms_custom_agv_api` thì nó sẽ tìm kiếm theo những từ khoá này theo `BM25 Keyword Search` sẽ cho ra kết quả tốt hơn.  
Các câu hỏi tự nhiên :  
```bash
Khi hàng trả về kho thì xử lý như thế nào?
Tại sao robot không chạy dù đang rảnh?
Quy trình nhập hàng gồm những bước nào?
```
Khi người dùng hỏi những câu hỏi tự nhiên như trên, vif nó không chứa các từ khoá đúng như trong tài liệu thì ta sử dụng `vector keyword` sẽ mang lại kết quả tốt hơn.  


Ta có kiến trúc tổng thể như sau:  
```bash
User Question
    ↓
Normalize Query
    ↓
BM25 Search        Qdrant Vector Search
    ↓                    ↓
Top 30             Top 30
    ↓                    ↓
        RRF Fusion
             ↓
        Top 5–8 chunks
             ↓
        Build Context
             ↓
        LLM Answer
             ↓
        Trả lời + nguồn
```
Ví dụ như sau:  
```bash
Câu hỏi:
"Tại sao return_in_muji không submit được?"

BM25 sẽ tìm tốt:
- return_in_muji
- submit
- stockType
- usp_wms_custom_in_submit

Vector sẽ tìm tốt:
- lỗi trả hàng
- quy trình Return In
- điều kiện submit dữ liệu

RRF sẽ gộp 2 danh sách lại.
LLM chỉ trả lời dựa trên context đã gộp.
```

# OLLAMA
Để sử dụng những AI, ta cần một nơi quản lý nó, gọi là OLLAMA


Tải mô hình `llama 3.1b`:  
```bash
ollama pull llama3.1:8b
```
Tải mô hình `embedding`:  
```bash
ollama pull nomic-embed-text-v2-moe
```

Sau đó chạy mô hình:  
```bash
ollama run llama3.1:8b
```


# Chạy code

Cài đặt môi trường ảo trên `MacOS` hoặc `Windows`:  
```python
# Tạo môi trường Python riêng cho dự án
# MacOS
python3 -m venv .llama_venv --prompt="venv llama"
# Windows
python -m venv .llama_venv --prompt="venv llama"
```
Sau đó kích hoạt môi trường ảo  
```python
# Kích hoạt môi trường ảo trên MacOS
source .venv/bin/activate
# Kích hoạt môi trường ảo trên Win
.llama_venv\Scripts\activate
```
Tiến hành cài đặt các thư viện cần thiết  
```python
# Trên MacOS
python3 -m pip install -r requirements.txt
# Trên Win
python -m pip install -r requirements.txt
```

Cài đa

```python
python3 chunk/embedding_chunk.py ingest --chunks "./outputs/farbic_warehouse_document.chunks.jsonl" --tenant-id "wms" --remove-old-versions
```
thiết lập tên người dùng Git toàn cục.
git config --global user.name "{Tên_Của_Bạn}"
thiết lập email người dùng Git toàn cục.
git config --global user.email "{email_của_bạn@example.com}"
sửa lại commit gần nhất,
đặt lại tác giả của commit theo cấu hình vừa thiết lập,
giữ nguyên nội dung thông điệp commit.
git commit --amend --reset-author --no-edit