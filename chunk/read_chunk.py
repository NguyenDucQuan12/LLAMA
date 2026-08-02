import json
from pathlib import Path


chunks_file = Path(
    "./outputs/"
    "fabric_warehouse_document.chunks.jsonl"
)


with chunks_file.open(
    "r",
    encoding="utf-8",
) as file_obj:

    for line_number, line in enumerate(
        file_obj,
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        record = json.loads(line)

        print("=" * 80)
        print("Chunk:", record["chunk_index"])
        print(
            "Token:",
            record["embedding_token_count"],
        )

        # Chuỗi này đã có prefix search_document:.
        text_to_embed = record["embedding_text"]

        print(text_to_embed)