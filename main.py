# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import pdfplumber
import requests
import tempfile
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss

app = FastAPI()

# Load models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Define request and response structure
class QueryRequest(BaseModel):
    documents: str  # URL to PDF
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

def extract_text_from_pdf_url(pdf_url):
    response = requests.get(pdf_url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    text = ""
    with pdfplumber.open(tmp_file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def split_text(text, max_length=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_length]
        chunks.append(" ".join(chunk))
        i += max_length - overlap
    return chunks

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
def run_query(request: QueryRequest):
    # 1. Extract text from document
    raw_text = extract_text_from_pdf_url(request.documents)
    
    # 2. Chunk and embed
    chunks = split_text(raw_text)
    embeddings = embed_model.encode(chunks)
    
    # 3. Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    # 4. For each question, find answer
    answers = []
    for query in request.questions:
        query_embedding = embed_model.encode([query])
        D, I = index.search(np.array(query_embedding), k=5)
        context = " ".join([chunks[i] for i in I[0]])
        result = qa_pipeline(question=query, context=context)
        answers.append(result["answer"])
    
    return QueryResponse(answers=answers)
