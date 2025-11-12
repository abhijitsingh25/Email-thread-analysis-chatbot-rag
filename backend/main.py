# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import uuid, time, json, os, pickle
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import sqlite3
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize

app = FastAPI()
BASE = Path(__file__).resolve().parent
INDEX_PATH = BASE / "store" / "bm25_index.pkl"
DOCS_PATH = BASE / "store" / "docs.pkl"
CHUNKS_PATH = BASE / "store" / "chunks.jsonl"
TRACE_DIR = Path("../runs")
TRACE_DIR.mkdir(parents=True, exist_ok=True)

# sessions: session_id -> {thread_id, memory, recent_turns}
SESSIONS = {}

# load index
import pickle
with open(INDEX_PATH, "rb") as f:
    BM25 = pickle.load(f)
with open(DOCS_PATH, "rb") as f:
    DOCS = pickle.load(f)

class StartReq(BaseModel):
    thread_id: str

class AskReq(BaseModel):
    session_id: str
    text: str
    search_outside_thread: Optional[bool] = False

def rewrite_with_memory(session, text):
    # very simple rule-based rewrite: replace "that doc" => last filename in memory
    mem = session.get("memory", {})
    if "last_filename" in mem and "that document" in text.lower() or "that doc" in text.lower():
        text = text.replace("that document", mem["last_filename"])
        text = text.replace("that doc", mem["last_filename"])
    return text

def retrieve_for_thread(thread_id, query, topk=8):
    tokens = word_tokenize(query.lower())
    scores = BM25.get_scores(tokens)
    ranked_idxs = scores.argsort()[::-1][:topk]
    results = []
    for idx in ranked_idxs:
        meta = DOCS[idx]
        if meta.get("thread_id") == thread_id:
            results.append({"meta": meta, "score": float(scores[idx])})
    # if not enough, return topk anyway
    if len(results) < topk:
        for idx in ranked_idxs:
            meta = DOCS[idx]
            if meta.get("thread_id") != thread_id:
                results.append({"meta": meta, "score": float(scores[idx])})
            if len(results) >= topk: break
    return results[:topk]

def assemble_answer(retrieved):
    # naive: find sentences that mention numbers/dates/names and cite them
    used = []
    answer = []
    citations = []
    for r in retrieved:
        m = r["meta"]
        text = m.get("text","")
        snippet = text.strip().split("\n")[0][:400]
        used.append(snippet)
        # citation format
        if m.get("source") == "email":
            cit = f"[msg: {m.get('message_id')}]"
        else:
            cit = f"[msg: {m.get('message_id')}, page: {m.get('page_no')}]"
        answer.append(f"{snippet} {cit}")
        citations.append(cit)
    final = "\n\n".join(answer[:3])  # keep short
    return final, citations

@app.post("/start_session")
def start_session(req: StartReq):
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {"thread_id": req.thread_id, "memory": {}, "recent_turns": []}
    return {"session_id": session_id, "thread_id": req.thread_id}

@app.post("/ask")
def ask(req: AskReq):
    session = SESSIONS.get(req.session_id)
    if not session:
        return {"error":"invalid session"}
    rewrite = rewrite_with_memory(session, req.text)
    retrieved = retrieve_for_thread(session["thread_id"], rewrite, topk=8)
    answer, citations = assemble_answer(retrieved)
    # update memory:
    session["recent_turns"].append({"user": req.text, "rewrite": rewrite, "time": time.time()})
    # log trace
    trace = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": req.session_id,
        "thread_id": session["thread_id"],
        "user": req.text,
        "rewrite": rewrite,
        "retrieved": [{"message_id": r["meta"]["message_id"], "score": r["score"]} for r in retrieved],
        "answer": answer,
        "citations": citations
    }
    trace_file = TRACE_DIR / f"trace_{int(time.time())}.jsonl"
    with open(trace_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace) + "\n")
    return {"answer": answer, "citations": citations, "rewrite": rewrite, "retrieved": retrieved, "trace_id": str(uuid.uuid4())}

@app.post("/reset_session")
def reset_session():
    SESSIONS.clear()
    return {"ok": True}
