# indexer.py
import json, argparse
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import pickle

def build_bm25(chunks_jsonl, out_index="bm25_index.pkl", out_docs="docs.pkl"):
    docs = []
    metas = []
    tokenized = []
    for line in open(chunks_jsonl, encoding="utf-8"):
        j = json.loads(line)
        text = j.get("text","")
        tokens = word_tokenize(text.lower())
        docs.append(tokens)
        metas.append(j)
        tokenized.append(tokens)
    bm25 = BM25Okapi(tokenized)
    with open(out_index, "wb") as f:
        pickle.dump(bm25, f)
    with open(out_docs, "wb") as f:
        pickle.dump(metas, f)
    print("Saved index and docs")

if __name__ == "__main__":
    import nltk
    nltk.download('punkt', quiet=True)
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", default="chunks.jsonl")
    args = p.parse_args()
    build_bm25(args.chunks)
