import time
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_community.document_loaders import (
    TextLoader,
    PythonLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    UnstructuredFileLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
TARGET_DIRS = [
    Path(os.getenv("TARGET_DIR", "./documents")),
]
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_store")
EXTENSION_LOADER_MAP = {
    ".txt": TextLoader,
    ".py": PythonLoader,
    ".pdf": UnstructuredPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".md": UnstructuredMarkdownLoader,
    ".r": UnstructuredFileLoader,
    ".rmd": UnstructuredFileLoader,
}
EMBEDDINGS = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=256
)
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
VECTORDATABASE = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=EMBEDDINGS
)

# --- FILE WATCHER HANDLER ---
class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            self.process(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self.process(event.src_path)

    def process(self, path_str):
        path = Path(path_str)
        ext = path.suffix.lower()
        if ext in EXTENSION_LOADER_MAP:
            print(f"[+] Detected change: {path.name}")
            try:
                loader = EXTENSION_LOADER_MAP[ext](str(path))
                docs = loader.load()
                chunks = SPLITTER.split_documents(docs)
                VECTORDATABASE.add_documents(chunks)
                VECTORDATABASE.persist()
                print(f"[✓] Updated vector DB with: {path.name}")
            except Exception as e:
                print(f"[!] Failed to process {path.name}: {e}")

# --- START OBSERVER ---
if __name__ == "__main__":
    event_handler = FileChangeHandler()
    observer = Observer()
    for target in TARGET_DIRS:
        observer.schedule(event_handler, str(target), recursive=True)
        print(f"[•] Watching: {target}")

    observer.start()
    print("[✓] File watcher started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n[!] File watcher stopped.")
    observer.join()