# FastAPI + Redis Chat History

## Cài dependency

```bash
pip install "redis>=5"
```

## Chép file

- `app/main.py`
- `app/schemas_conversation.py`
- `app/services/conversation_history_service.py`
- `app/services/conversation_question_rewriter.py`
- `app/services/conversation_aware_qa_service.py`

Bổ sung các field trong `SETTINGS_PATCH.py` vào `Settings`.

## Khởi động Redis

```bash
docker run --name rag-redis -p 6379:6379 -d redis:7-alpine
```

## Chạy API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## Tạo conversation

```bash
curl -X POST http://127.0.0.1:8000/v1/conversations \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"viva_factory"}'
```

## Câu đầu

```bash
curl -X POST http://127.0.0.1:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id":"viva_factory",
    "question":"Robot AGV-01 bị treo nhiệm vụ là gì?",
    "mode":"documents"
  }'
```

Lấy `conversation_id` trong response.

## Câu tiếp theo

```bash
curl -X POST http://127.0.0.1:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id":"viva_factory",
    "conversation_id":"UUID_TU_RESPONSE_TRUOC",
    "question":"Cách xử lý thế nào?",
    "mode":"documents"
  }'
```

Server sẽ viết lại câu sau thành câu độc lập dựa trên lịch sử.

## Xem lịch sử

```bash
curl "http://127.0.0.1:8000/v1/conversations/UUID?tenant_id=viva_factory"
```

## Xóa lịch sử

```bash
curl -X DELETE \
  "http://127.0.0.1:8000/v1/conversations/UUID?tenant_id=viva_factory"
```

## Cuộc trò chuyện dài

Khi số message vượt `chat_history_compact_trigger_messages`, service gọi Llama
để gộp các lượt cũ thành rolling summary, giữ nguyên
`chat_history_keep_recent_messages` lượt gần nhất. Summary và các lượt gần nhất
được dùng để viết lại câu hỏi follow-up.

Ví dụ:

1. `Robot AGV-01 bị treo nhiệm vụ là gì?`
2. `Cách xử lý thế nào?`

Câu thứ hai có thể được viết lại thành:

`Cách xử lý khi robot AGV-01 bị treo nhiệm vụ là gì?`

Dense retrieval chỉ embedding câu hỏi độc lập này, không embedding toàn bộ lịch sử.
