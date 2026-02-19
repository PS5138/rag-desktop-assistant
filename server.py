# server.py

import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from collections import deque

load_dotenv()

app = FastAPI()

# Allow local frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock this down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load vector store
VECTOR_PATH = os.getenv("VECTOR_DB_PATH", "./vector_store")
embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=256)
vectordb = Chroma(persist_directory=VECTOR_PATH, embedding_function=embedding)

# Setup chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4", temperature=0),
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

chat_history = deque(maxlen=6)

# Request model
class QueryRequest(BaseModel):
    question: str

# Response model
class QueryResponse(BaseModel):
    answer: str
    sources: list

@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    result = qa_chain({"question": req.question, "chat_history": list(chat_history)})
    chat_history.append(("user", req.question))
    chat_history.append(("ai", result["answer"]))

    sources = [doc.metadata.get("source", "") for doc in result["source_documents"]]
    return QueryResponse(answer=result["answer"], sources=sources)