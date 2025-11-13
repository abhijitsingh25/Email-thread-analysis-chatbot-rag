# ingest.py
import pandas as pd
import json, os
from pathlib import Path
import nltk
nltk.download("punkt", quiet=True)

def chunk_text(text, max_words=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
        i += max_words - overlap
    return chunks

def ingest_csv(csv_path, out_jsonl="backend/store/chunks.jsonl"):
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    count = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            text = str(row.get("body", ""))
            chunks = chunk_text(text)
            for i, ch in enumerate(chunks):
                record = {
                    "message_id": row.get("message_id"),
                    "thread_id": row.get("thread_id"),
                    "subject": row.get("subject"),
                    "date": row.get("date"),
                    "chunk_id": f"{row.get('message_id')}_{i}",
                    "text": ch,
                    "source": "email",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    print(f"✅ Saved {count} text chunks to {out_jsonl}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Path to sliced CSV")
    p.add_argument("--out", default="backend/store/chunks.jsonl", help="Output JSONL path")
    args = p.parse_args()
    ingest_csv(args.src, args.out)

#uv run ingest.py --src data/sliced_emails_labeled.csv --out backend/store/chunks.jsonl