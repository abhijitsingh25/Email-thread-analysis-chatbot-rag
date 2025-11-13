# indexer.py (Hybrid Index)
import json, pickle, numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from nltk.tokenize import word_tokenize
from pathlib import Path
import nltk

nltk.download("punkt", quiet=True)

def build_hybrid_index(chunks_file="backend/store/chunks.jsonl"):
    texts, tokens, metas = [], [], []

    for line in open(chunks_file, encoding="utf-8"):
        j = json.loads(line)
        texts.append(j["text"])
        tokens.append(word_tokenize(j["text"].lower()))
        metas.append(j)

    print(f"📘 Loaded {len(texts)} chunks")

    # BM25 index
    bm25 = BM25Okapi(tokens)

    # Semantic embeddings
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(emb)

    # FAISS index
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    Path("backend/store").mkdir(parents=True, exist_ok=True)
    with open("backend/store/bm25_index.pkl", "wb") as f: pickle.dump(bm25, f)
    with open("backend/store/docs.pkl", "wb") as f: pickle.dump(metas, f)
    np.save("backend/store/embeddings.npy", emb)
    faiss.write_index(index, "backend/store/faiss_index.index")

    print(" Hybrid index saved successfully.")

if __name__ == "__main__":
    build_hybrid_index()