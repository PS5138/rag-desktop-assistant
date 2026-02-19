# RAG Desktop Assistant

A personal RAG (Retrieval-Augmented Generation) desktop application that indexes your local documents into a vector database and lets you ask natural language questions about them — with answers grounded in your own files.

Built with LangChain, ChromaDB, OpenAI, FastAPI, and a Tauri + React desktop frontend.

## Motivation

I wanted a way to quickly search and query across all my personal documents — PDFs, Word docs, code files, markdown notes — using natural language instead of manually digging through folders. Rather than relying on a cloud service, I built this as a fully local tool where the vector database lives on my machine and I control exactly which directories get indexed.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────┐     ┌───────────────┐
│   Documents     │────▶│  backend.py  │────▶│  ChromaDB     │
│  (.pdf, .docx,  │     │  (indexer)   │     │  Vector Store  │
│   .py, .md ...) │     └──────────────┘     └───────┬───────┘
│                 │                                   │
│                 │     ┌──────────────┐              │
│                 │────▶│  watcher.py  │──────────────┘
│                 │     │  (live sync) │        (updates on file change)
└─────────────────┘     └──────────────┘
                                                      │
                        ┌──────────────┐              │
                        │  query.py    │◀─────────────┘
                        │  (terminal)  │        (retrieves chunks)
                        └──────────────┘
                                                      │
┌─────────────────┐     ┌──────────────┐              │
│  React + Tauri  │◀───▶│  server.py   │◀─────────────┘
│  Desktop App    │     │  (FastAPI)   │
└─────────────────┘     └──────────────┘
```

## How It Works

### 1. Document Indexing (`backend.py`)

The indexer recursively scans target directories and loads documents using LangChain's document loaders (each file type has a specialised loader — `UnstructuredPDFLoader` for PDFs, `PythonLoader` for `.py` files, etc.). Documents are then split into overlapping chunks of 1000 characters (with 200-character overlap to preserve context across boundaries) using `RecursiveCharacterTextSplitter`. Each chunk is embedded into a 256-dimensional vector using OpenAI's `text-embedding-3-small` model and stored in a ChromaDB vector store with cosine similarity as the distance metric. Indexing is done in batches of 100 to avoid memory issues with large document sets.

### 2. Live File Watching (`watcher.py`)

Rather than re-indexing every time a file changes, a `watchdog` observer monitors the target directories and triggers on file creation or modification. When a change is detected, only the affected file is loaded, chunked, embedded, and added to the existing vector store — keeping it up to date without a full rebuild.

**Trade-off:** The watcher handles additions and modifications, but not deletions. If a file is deleted, its embeddings remain in the vector store until a full re-index is run via `backend.py`. This was a deliberate simplification — handling deletions would require tracking file-to-embedding mappings, which adds complexity for a relatively rare operation.

### 3. Querying — Terminal Interface (`query.py`)

This was the first interface I built as a proof of concept. It uses LangChain's `RetrievalQA` chain which:
1. Takes your question and searches the vector store for the 5 most semantically similar document chunks
2. Sends those chunks along with your question to GPT-4
3. GPT-4 generates an answer grounded in the retrieved context
4. The answer and source file paths are printed to the terminal

Each question is independent — there's no conversational memory at this stage. I built this first to validate the RAG pipeline before investing time in the frontend.

### 4. Querying — Desktop App (`server.py` + `frontend/`)

Once the terminal interface was working, I built the full desktop experience:

**API Server (`server.py`):** A FastAPI server that exposes a `POST /query` endpoint. It upgrades from the terminal version's `RetrievalQA` to `ConversationalRetrievalChain`, which maintains a rolling chat history (last 6 messages via a `deque`) so follow-up questions understand context from previous answers.

**Desktop Frontend (`frontend/`):** A Tauri + React + TypeScript app that provides a clean UI for asking questions. The React frontend sends queries to the FastAPI server and displays answers. Tauri was chosen over Electron for its smaller bundle size and native performance.

The two query interfaces exist independently — `query.py` is useful for quick terminal-based lookups, while the desktop app is the full-featured experience with conversational memory.

## Project Structure

| File | Description |
|---|---|
| `backend.py` | Document indexer — scans directories, chunks documents, builds the vector store |
| `watcher.py` | File system watcher — keeps the vector store in sync with file changes in real time |
| `query.py` | Terminal-based Q&A interface (stateless, no memory between questions) |
| `server.py` | FastAPI server — exposes the RAG pipeline as an API with conversational memory |
| `API_test.py` | Smoke test to verify the OpenAI API key and embedding model |
| `frontend/` | Tauri + React + TypeScript desktop application |

## Supported File Types

- `.txt` — Plain text
- `.py` — Python source code
- `.pdf` — PDF documents
- `.docx` — Word documents
- `.md` — Markdown
- `.r` / `.rmd` — R and R Markdown files

## Setup

### Prerequisites

- Python 3.10+
- Node.js (for the frontend)
- Rust (for Tauri)
- An OpenAI API key

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/rag-desktop-assistant.git
cd rag-desktop-assistant
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-api-key-here
TARGET_DIR=/path/to/your/documents
VECTOR_DB_PATH=./vector_store
```

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key | *(required)* |
| `TARGET_DIR` | Directory of documents to index | `./documents` |
| `VECTOR_DB_PATH` | Where the vector database is stored | `./vector_store` |

### 3. Build the vector store

```bash
python backend.py
```

This scans your `TARGET_DIR`, loads all supported files, and creates the vector store. Run this again whenever you want a full re-index.

### 4. (Optional) Start the file watcher

```bash
python watcher.py
```

Watches your `TARGET_DIR` for file changes and automatically updates the vector store. Run in the background with:

```bash
nohup python watcher.py &
```

### 5. Query from the terminal

```bash
python query.py
```

### 6. Run the full desktop app

Start the API server:

```bash
uvicorn server:app --reload --port 8000
```

In a separate terminal, start the frontend:

```bash
cd frontend
npm install
npm run tauri dev
```

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Document Loading | LangChain | File parsing, text splitting, RAG chain orchestration |
| Vector Store | ChromaDB | Local embedding storage and similarity search |
| Embeddings | OpenAI `text-embedding-3-small` | 256-dimensional document embeddings |
| LLM | OpenAI GPT-4 | Answer generation from retrieved context |
| API Server | FastAPI | REST API connecting the backend to the frontend |
| Desktop App | Tauri + React + TypeScript | Native desktop UI |
| File Monitoring | Watchdog | Real-time file system event detection |
