# ingest.py
import os, json, argparse, hashlib
from mailparser import parse_from_file
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
from pathlib import Path
import sqlite3
import nltk, math
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize, sent_tokenize

def sha1(x): return hashlib.sha1(x.encode('utf-8')).hexdigest()

def extract_pdf_text_with_pages(path):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append((i+1, text))
    return pages

def chunk_text(text, max_tokens=300, overlap=50):
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+max_tokens])
        out.append(chunk)
        i += max_tokens - overlap
    return out

def write_chunk(out_f, chunk_meta):
    out_f.write(json.dumps(chunk_meta, ensure_ascii=False) + "\n")

def parse_eml_folder(src_folder, out_jsonl, metadata_db):
    conn = sqlite3.connect(metadata_db)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS messages (
                    doc_id TEXT PRIMARY KEY, message_id TEXT, thread_id TEXT,
                    date TEXT, from_addr TEXT, to_addr TEXT, subject TEXT
                   )""")
    conn.commit()

    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for root, _, files in os.walk(src_folder):
            for fn in files:
                if not fn.lower().endswith(".eml"): continue
                path = os.path.join(root, fn)
                mail = parse_from_file(path)
                message_id = mail.message_id or sha1(mail.headers.get("Subject","") + str(mail.date))
                thread_id = mail.headers.get("In-Reply-To") or mail.headers.get("References") or sha1(mail.subject or "")
                plain = mail.text_plain[0] if mail.text_plain else mail.body or ""
                doc_id = sha1(message_id + plain[:500])
                # message chunk
                chunk_meta = {
                    "doc_id": doc_id,
                    "thread_id": thread_id,
                    "message_id": message_id,
                    "page_no": None,
                    "chunk_id": doc_id + "_m",
                    "text": plain,
                    "source": "email",
                    "subject": mail.subject,
                    "from": mail.from_[0][1] if mail.from_ else "",
                    "to": ", ".join([t[1] for t in mail.to]) if mail.to else "",
                }
                write_chunk(out_f, chunk_meta)
                cur.execute("INSERT OR REPLACE INTO messages(doc_id,message_id,thread_id,date,from_addr,to_addr,subject) VALUES(?,?,?,?,?,?,?)",
                            (chunk_meta["doc_id"], message_id, thread_id, str(mail.date), chunk_meta["from"], chunk_meta["to"], mail.subject))
                conn.commit()

                # attachments
                for att in mail.attachments or []:
                    att_fname = att.get("filename") or "attachment.bin"
                    att_data = att.get("payload") or b""
                    tmp_path = "/tmp/" + att_fname
                    with open(tmp_path, "wb") as f:
                        f.write(att_data)
                    # only process pdf/docx/txt/html
                    if tmp_path.lower().endswith(".pdf"):
                        pages = extract_pdf_text_with_pages(tmp_path)
                        for pnum, text in pages:
                            chunks = chunk_text(text, max_tokens=300, overlap=50)
                            for i,ch in enumerate(chunks):
                                chunk_meta = {
                                    "doc_id": sha1(message_id + att_fname + str(pnum) + str(i)),
                                    "thread_id": thread_id,
                                    "message_id": message_id,
                                    "page_no": pnum,
                                    "chunk_id": sha1(message_id + att_fname + str(pnum) + str(i)),
                                    "text": ch,
                                    "source": "attachment",
                                    "filename": att_fname
                                }
                                write_chunk(out_f, chunk_meta)
                    else:
                        # try plain text extraction
                        try:
                            text = extract_text(tmp_path)
                        except Exception:
                            text = ""
                        if not text:
                            continue
                        chunks = chunk_text(text, max_tokens=300, overlap=50)
                        for i,ch in enumerate(chunks):
                            chunk_meta = { ... }  # same pattern as above
                            write_chunk(out_f, chunk_meta)

    conn.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", default="chunks.jsonl")
    p.add_argument("--db", default="metadata.sqlite")
    args = p.parse_args()
    parse_eml_folder(args.src, args.out, args.db)
