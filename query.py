from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_store")

# Embedding model (must match what you used in backend.py)
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=256
)

# Load Chroma vector store
vectordb = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embedding
)

# Set up the language model (GPT-4 turbo)
llm = ChatOpenAI(
    model="gpt-4-1106-preview",
    temperature=0
)

# RAG pipeline
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
    return_source_documents=True  # Optional: helpful to see where info came from
)

# --- QUERY LOOP ---
print("RAG assistant ready. Ask a question (or type 'exit' to quit):\n")

while True:
    query = input("🔍 Your question: ").strip()
    if query.lower() in {"exit", "quit"}:
        print("👋 Exiting.")
        break

    try:
        result = qa(query)
        print("\n Answer:\n", result["result"])
        
        # Optional: Show where the answer came from
        print("\n Source documents:")
        for doc in result["source_documents"]:
            print("-", doc.metadata.get("source", "Unknown source"))
    except Exception as e:
        print(f"[!] Error: {e}")