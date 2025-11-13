# # backend/main.py (Hybrid + Cross-Encoder + Summarizer)
# from fastapi import FastAPI
# from pydantic import BaseModel
# import pickle, uuid, numpy as np, faiss
# from sentence_transformers import SentenceTransformer, CrossEncoder
# from rank_bm25 import BM25Okapi
# from transformers import pipeline
# from nltk.tokenize import word_tokenize
# from pathlib import Path
# import nltk

# nltk.download("punkt", quiet=True)

# app = FastAPI(title="Hybrid RAG Backend with Cross-Encoder")

# STORE = Path(__file__).resolve().parent / "store"

# # --- Load indexes ---
# with open(STORE / "bm25_index.pkl", "rb") as f: BM25 = pickle.load(f)
# with open(STORE / "docs.pkl", "rb") as f: DOCS = pickle.load(f)
# embeddings = np.load(STORE / "embeddings.npy")
# index = faiss.read_index(str(STORE / "faiss_index.index"))

# # --- Models ---
# retriever_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# summarizer = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")

# SESSIONS = {}

# class StartSession(BaseModel):
#     thread_id: str

# class AskQuery(BaseModel):
#     session_id: str
#     question: str


# @app.post("/start_session")
# def start_session(req: StartSession):
#     sid = str(uuid.uuid4())
#     SESSIONS[sid] = {"thread_id": req.thread_id}
#     return {"session_id": sid, "thread_id": req.thread_id}


# def hybrid_search(query, thread_id, topk=10, alpha=0.6):
#     """Combine BM25 (lexical) + FAISS (semantic)."""
#     tokens = word_tokenize(query.lower())
#     bm25_scores = BM25.get_scores(tokens)

#     q_emb = retriever_model.encode([query], convert_to_numpy=True)
#     faiss.normalize_L2(q_emb)
#     D, I = index.search(q_emb, len(DOCS))
#     vector_scores = np.zeros(len(DOCS))
#     for rank, idx in enumerate(I[0]):
#         vector_scores[idx] = D[0][rank]

#     # Normalize & combine
#     bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)
#     vec_norm = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min() + 1e-9)
#     hybrid = alpha * bm25_norm + (1 - alpha) * vec_norm

#     ranked = np.argsort(hybrid)[::-1]
#     results = []
#     for i in ranked:
#         doc = DOCS[i]
#         if doc["thread_id"] == thread_id:
#             results.append((DOCS[i], hybrid[i]))
#         if len(results) >= topk:
#             break
#     return results


# @app.post("/ask")
# def ask(req: AskQuery):
#     if req.session_id not in SESSIONS:
#         return {"error": "Invalid session"}

#     thread_id = SESSIONS[req.session_id]["thread_id"]

#     # --- Step 1: Hybrid retrieval ---
#     hybrid_results = hybrid_search(req.question, thread_id, topk=20, alpha=0.6)

#     # --- Step 2: Cross-Encoder re-ranking ---
#     pairs = [(req.question, doc["text"]) for doc, _ in hybrid_results]
#     ce_scores = cross_encoder.predict(pairs)
#     reranked = sorted(zip(hybrid_results, ce_scores), key=lambda x: x[1], reverse=True)

#     # --- Step 3: Select top contexts ---
#     contexts, citations = [], []
#     for (doc, _), ce_score in reranked[:3]:
#         contexts.append(doc["text"].strip())
#         citations.append(doc["message_id"])

#     # --- Step 4: Summarization ---
#     context_text = "\n\n".join(contexts)[:1500]
#     prompt = f"Question: {req.question}\nContext: {context_text}\nAnswer concisely and factually:"
#     gen = summarizer(prompt, max_new_tokens=64, do_sample=False)[0]["generated_text"].strip()

#     return {
#         "answer": gen,
#         "citations": citations,
#         "retrieved": len(contexts),
#         "trace_id": str(uuid.uuid4())
#     }


# @app.get("/")
# def home():
#     return {"status": "Hybrid + Cross-Encoder RAG backend running"}
# backend/main.py (Hybrid + Cross-Encoder + Summarizer)
from fastapi import FastAPI
from pydantic import BaseModel
import pickle, uuid, numpy as np, faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import pipeline
from nltk.tokenize import word_tokenize
from pathlib import Path
import nltk, json   # 🔵 Added json

nltk.download("punkt", quiet=True)

app = FastAPI(title="Hybrid RAG Backend with Cross-Encoder")

STORE = Path(__file__).resolve().parent / "store"

# Load thread label -> real thread ID map
with open(Path(__file__).resolve().parent.parent / "data" / "thread_map.json") as f:
    original_to_label = json.load(f)

# Reverse it: T-0001 → real thread_id
label_to_original = {v: k for k, v in original_to_label.items()}

# --- Load indexes ---
with open(STORE / "bm25_index.pkl", "rb") as f: BM25 = pickle.load(f)
with open(STORE / "docs.pkl", "rb") as f: DOCS = pickle.load(f)
embeddings = np.load(STORE / "embeddings.npy")
index = faiss.read_index(str(STORE / "faiss_index.index"))

# --- Models ---
retriever_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
summarizer = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")

SESSIONS = {}

# UPDATE: Accept thread_label instead of raw thread_id
class StartSession(BaseModel):
    thread_label: str

class AskQuery(BaseModel):
    session_id: str
    question: str


@app.post("/start_session")
def start_session(req: StartSession):
    # 🔵 Convert label → real thread_id
    if req.thread_label not in label_to_original:
        return {"error": f"Unknown thread label: {req.thread_label}"}

    real_thread_id = label_to_original[req.thread_label]

    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"thread_id": real_thread_id}

    return {
        "session_id": sid,
        "thread_label": req.thread_label,
        "thread_id": real_thread_id   # for debugging
    }


def hybrid_search(query, thread_id, topk=10, alpha=0.6):
    """Combine BM25 (lexical) + FAISS (semantic)."""
    tokens = word_tokenize(query.lower())
    bm25_scores = BM25.get_scores(tokens)

    q_emb = retriever_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, len(DOCS))
    vector_scores = np.zeros(len(DOCS))
    for rank, idx in enumerate(I[0]):
        vector_scores[idx] = D[0][rank]

    # Normalize & combine
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)
    vec_norm = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min() + 1e-9)
    hybrid = alpha * bm25_norm + (1 - alpha) * vec_norm

    ranked = np.argsort(hybrid)[::-1]
    results = []
    for i in ranked:
        doc = DOCS[i]
        if doc["thread_id"] == thread_id:
            results.append((DOCS[i], hybrid[i]))
        if len(results) >= topk:
            break
    return results


@app.post("/ask")
def ask(req: AskQuery):
    if req.session_id not in SESSIONS:
        return {"error": "Invalid session"}

    thread_id = SESSIONS[req.session_id]["thread_id"]

    # --- Step 1: Hybrid retrieval ---
    hybrid_results = hybrid_search(req.question, thread_id, topk=20, alpha=0.6)

    # --- Step 2: Cross-Encoder re-ranking ---
    pairs = [(req.question, doc["text"]) for doc, _ in hybrid_results]
    ce_scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(hybrid_results, ce_scores), key=lambda x: x[1], reverse=True)

    # --- Step 3: Select top contexts ---
    contexts, citations = [], []
    for (doc, _), ce_score in reranked[:3]:
        contexts.append(doc["text"].strip())
        citations.append(doc["message_id"])

    # --- Step 4: Summarization ---
    context_text = "\n\n".join(contexts)[:1500]
    prompt = f"Question: {req.question}\nContext: {context_text}\nAnswer concisely and factually:"
    gen = summarizer(prompt, max_new_tokens=64, do_sample=False)[0]["generated_text"].strip()

    return {
        "answer": gen,
        "citations": citations,
        "retrieved": len(contexts),
        "trace_id": str(uuid.uuid4())
    }


@app.get("/")
def home():
    return {"status": "Hybrid + Cross-Encoder RAG backend running"}
