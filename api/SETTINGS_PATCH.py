# Bổ sung vào app/config.py / Settings.
# Giá trị dưới đây là gợi ý khởi đầu.

redis_url: str = "redis://localhost:6379/0"
redis_connect_timeout_seconds: float = 3.0
redis_command_timeout_seconds: float = 5.0

chat_history_key_prefix: str = "rag:conversation"
chat_history_ttl_seconds: int = 604800            # 7 ngày
chat_history_max_stored_messages: int = 100
chat_history_context_messages: int = 12
chat_history_max_message_characters: int = 12000
chat_history_lock_timeout_seconds: int = 180
chat_history_lock_wait_seconds: float = 2.0

conversation_rewrite_timeout_seconds: float = 30.0
conversation_context_max_characters: int = 12000

api_max_concurrent_questions: int = 2
api_queue_wait_seconds: float = 0.1
api_expose_internal_errors: bool = False
cors_allow_credentials: bool = True

chat_history_compact_trigger_messages: int = 24
chat_history_keep_recent_messages: int = 10
chat_history_compact_timeout_seconds: float = 45.0
chat_history_max_summary_characters: int = 6000
