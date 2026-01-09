# Local RAG Prototype (Interview Demo)

This project is a fully local Retrieval-Augmented Generation (RAG) prototype built for an interview demonstration.

## Features
- Local document ingestion (.txt)
- Text chunking with overlap
- Embeddings using an open-source model
- Local vector database (FAISS)
- Local LLM inference (Ollama)
- Grounded question answering with citations

## Tech Stack
- Python
- sentence-transformers
- FAISS
- Ollama (local LLM)

## How It Works
1. Documents are ingested and chunked
2. Chunks are embedded and stored in a local vector index
3. User queries retrieve top-k relevant chunks
4. A local LLM generates answers grounded in retrieved content

## Notes
- No external APIs are used
- All inference runs locally
- Demo uses fake weather facts for Vancouver
