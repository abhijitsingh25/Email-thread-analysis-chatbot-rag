# Email and Attachment RAG System
This project implements a Retrieval-Augmented Generation (RAG) system for querying the Enron Email Dataset. It uses hybrid retrieval (BM25 + FAISS semantic search), cross-encoder reranking, and a T5-based answer generation model. The frontend is built with Streamlit and the backend is built using FastAPI. The entire pipeline is Dockerized for reproducible deployment.

# Docker Image Repository
Docker Hub Repository:
https://hub.docker.com/r/abhijitdevs/email-rag-app

Pull Command:
```bash
docker pull abhijitdevs/email-rag-app:latest
```

# Demo Video
https://drive.google.com/file/d/1FtMLjnP5VQ-XQtC50DS9Ttds3jzV5A2L/view

# Overview
This system enables question answering over email threads. Users can select a thread, ask a question, and receive an answer grounded in the actual email content. The system retrieves relevant chunks, reranks them for maximum relevance, and synthesizes a final answer.

# Architecture
The complete workflow consists of four major components:

1. Preprocessing  
2. Ingestion  
3. Backend (FastAPI)  
4. Frontend (Streamlit)

The retrieval workflow is:
Raw Emails → Preprocessed Metadata → Chunking → BM25 and FAISS Indexes → Hybrid Retrieval → Cross-Encoder Reranking → T5 Answer Generation → Streamlit Interface

# Preprocessing Pipeline
The preprocessing script converts raw RFC822 emails into a structured dataset. Main tasks include:

- Parsing raw Enron email text
- Extracting metadata such as message_id, subject, date, and body
- Building thread identifiers using SHA-1 hashing
- Grouping messages into threads
- Filtering a meaningful date range
- Selecting the top threads based on activity
- Limiting total data size to comply with assignment requirements
- Adding human-readable labels like T-0001, T-0002
- Saving:
  - sliced_emails_labeled.csv
  - thread_map.json

This step ensures the dataset is clean, compact, and optimized for retrieval.

# Ingestion Pipeline
The ingestion script prepares the email dataset for retrieval by:

- Chunking long email bodies into small overlapping segments
- Storing each chunk as a separate indexed unit
- Generating a chunks.jsonl file
- Creating all offline retrieval indexes:
  - docs.pkl
  - bm25_index.pkl
  - embeddings.npy
  - faiss_index.index

Chunking is essential for improving retrieval quality and ensuring accurate matches for user queries.

# Backend (FastAPI)
The backend handles:

- Loading BM25, FAISS, embeddings, and documents
- Processing user queries
- Hybrid retrieval using:
  - BM25 lexical matching
  - FAISS semantic vector search
- Cross-encoder reranking using MiniLM
- Answer generation with FLAN-T5
- Returning structured responses containing:
  - Final answer
  - Citations (message IDs)
  - Number of retrieved chunks
  - Trace ID for debugging

Thread-level isolation ensures that answers come only from the selected email thread.

# Frontend (Streamlit)
The Streamlit application provides:

- Dropdown for selecting thread labels (T-XXXX)
- Button to start a session linked to a thread
- Text area for entering questions
- Display of final answer
- List of citations for transparency

The UI is simple, responsive, and optimized for interactive usage.

# Docker Usage
Build the image:
``` bash
docker build -t email-rag-app .
```
Pull the published image:
``` bash
docker pull abhijitdevs/email-rag-app:latest
```
Run the full application:
``` bash
docker run -p 8000:8000 -p 8501:8501 abhijitdevs/email-rag-app:latest
```
Endpoints:
``` bash
Streamlit UI: http://localhost:8501

FastAPI backend: http://localhost:8000
```
# Technologies Used

## Retrieval
- BM25 (rank_bm25)
- MiniLM Embeddings (SentenceTransformers)
- FAISS Vector Store

## Reranking
- Cross-Encoder MiniLM (MS MARCO model)

## Answer Generation
- FLAN-T5 Base

## Backend
- FastAPI
- Uvicorn

## Frontend
- Streamlit

## Preprocessing
- Pandas
- Python email parser
- SHA-1 thread hashing

## Containerization
- Docker
- Docker Hub

# Author
Abhijit Singh  
AI Engineer — NLP, RAG Systems, and Generative AI Applications
