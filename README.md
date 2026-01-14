# Local RAG System Prototype (Fully Offline)

This repository contains a **fully local Retrieval-Augmented Generation (RAG) prototype** built in Python. The system ingests text documents, generates embeddings using an open-source model, stores them in a local vector database, retrieves relevant passages for a user query, and generates grounded answers using a local LLM (via Ollama).

This project was built as a technical prototype for an **AI Knowledge Base / RAG System interview** and is designed to be simple, explainable, and runnable end-to-end on a local machine with no external API calls.

---

## 🧠 What This Project Demonstrates

* End-to-end RAG pipeline (ingestion → embeddings → retrieval → generation)
* Use of **open-source embeddings** (SentenceTransformers)
* Use of a **local vector database** (FAISS)
* Use of a **local LLM** (Ollama – no cloud APIs)
* Grounded answers with **source citations**
* Simple, clear Python architecture suitable for extension

---

## 📂 Project Structure

```
rag_systems_prototype/
├── data/
│   └── vancouver_weather.txt   # Fake knowledge base (TXT)
├── ingest.py                   # Document loading & chunking
├── embed_store.py              # Embedding generation & vector indexing
├── query.py                    # Retrieval-only querying (no LLM)
├── rag.py                      # Full RAG pipeline (retrieve + generate)
├── vector.index                # FAISS vector index (generated)
├── chunks.pkl                  # Stored text chunks (generated)
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 📄 Data

The `data/vancouver_weather.txt` file contains **fake, intentionally unrealistic facts** about Vancouver weather.

This is done to ensure:

* The LLM cannot rely on pretrained knowledge
* All answers must come from retrieved documents
* Hallucinations are easy to detect

---

## ⚙️ Setup Instructions

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama

Download and install Ollama from:

[https://ollama.com/download](https://ollama.com/download)

Then pull a small local model (example):

```bash
ollama pull llama3.2:1b
```

---

## 🚀 How to Run the Pipeline

### Step 1: Ingest & embed documents

```bash
python embed_store.py
```

This will:

* Load the document
* Chunk the text
* Generate embeddings
* Store them in a FAISS index

### Step 2: (Optional) Test retrieval only

```bash
python query.py
```

This allows you to verify that vector search works before adding generation.

### Step 3: Run the full RAG system

```bash
python rag.py
```

You can now ask multiple questions interactively:

```text
Ask a question: When does neon snowfall occur in Vancouver?
```

Type `exit` or `quit` to stop the program.

---

## 🧩 How the RAG Pipeline Works

1. **Ingestion** (`ingest.py`)

   * Loads TXT document
   * Splits text into overlapping chunks

2. **Embedding & Indexing** (`embed_store.py`)

   * Generates embeddings using `all-MiniLM-L6-v2`
   * Stores vectors in a FAISS index

3. **Retrieval** (`query.py` / `rag.py`)

   * Embeds user query
   * Performs top-k similarity search

4. **Generation** (`rag.py`)

   * Builds a grounded prompt
   * Calls a local LLM via Ollama
   * Returns answers with source citations

---

## 🛡️ Hallucination Control

The system reduces hallucinations by:

* Using fake data (no real-world knowledge leakage)
* Enforcing strict prompt rules
* Requiring answers to be grounded in retrieved context
* Returning "I don't know based on the provided documents" when needed

---

## 🔮 Future Improvements

* PDF ingestion and OCR support
* Metadata-aware chunking
* Reranking (cross-encoder)
* Hybrid retrieval (BM25 + vectors)
* Web or API interface (FastAPI)
* Incremental re-indexing

---

## 👩‍💻 Author

**Prakriti Dhital**
BSIT Student | Python & AI Systems Developer

This project was built as a learning-focused prototype and interview demonstration of Retrieval-Augmented Generation systems.
