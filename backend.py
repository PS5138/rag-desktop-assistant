from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    TextLoader,
    PythonLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    # UnstructuredNotebookLoader,
    UnstructuredFileLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# List of specific directories to index
TARGET_DIRS = [
    Path(os.getenv("TARGET_DIR", "./documents"))
]

# Supported file extensions and corresponding loaders
EXTENSION_LOADER_MAP = {
    ".txt": TextLoader,
    ".py": PythonLoader,
    # ".ipynb": UnstructuredNotebookLoader,
    ".pdf": UnstructuredPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".md": UnstructuredMarkdownLoader,
    ".r": UnstructuredFileLoader,
    ".rmd": UnstructuredFileLoader,
}

# Directories to skip
skip_dirs = {"__pycache__", "node_modules", ".git", ".DS_Store"}

# Step 1: Load all documents from selected directories
docs = []
print("Starting document loading...")

for base_dir in TARGET_DIRS:
    for i, filepath in enumerate(base_dir.rglob("*")):
        if any(part in skip_dirs for part in filepath.parts):
            continue
        if any("env" in part.lower() for part in filepath.parts):
            continue
        if filepath.suffix.lower() in EXTENSION_LOADER_MAP:
            loader_class = EXTENSION_LOADER_MAP[filepath.suffix.lower()]
            try:
                loader = loader_class(str(filepath))
                docs.extend(loader.load())
                print(f"[{i}] Loaded: {filepath.name}")
            except Exception as e:
                print(f"[{i}] Failed to load {filepath.name}: {e}")

print(f"Finished loading. Total documents: {len(docs)}")

# Step 2: Split documents into chunks
print("Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(docs)
print(f"Total chunks: {len(split_docs)}")

# Step 3: Setup embeddings
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=256
)

# Step 4: Create vector store in batches of 100
print("Starting vector store creation...")
batch_size = 100
vectordb = None

for i in range(0, len(split_docs), batch_size):
    chunk = split_docs[i:i+batch_size]
    print(f"Processing batch {i // batch_size + 1}: documents {i} to {i+len(chunk)-1}")

    if vectordb is None:
        vectordb = Chroma.from_documents(
            documents=chunk,
            embedding=embedding,
            persist_directory=os.getenv("VECTOR_DB_PATH", "./vector_store"),
            collection_metadata={"hnsw:space": "cosine"}
        )
    else:
        vectordb.add_documents(chunk)

print("Persisting vector store to disk...")
vectordb.persist()
print("Vector store created and saved using text-embedding-3-small with 256 dimensions.")