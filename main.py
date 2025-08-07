# main.py
from flask import Flask, request, jsonify
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

app = Flask(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
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

@app.route("/api/v1/hackrx/run", methods=["POST"])
def run_qa():
    # Receive uploaded file and question
    if 'file' not in request.files or 'question' not in request.form:
        return jsonify({"error": "Missing file or question"}), 400

    file = request.files['file']
    question = request.form['question']

    file_path = "temp.pdf"
    file.save(file_path)

    try:
        raw_text = extract_text_from_pdf(file_path)
        chunks = split_text(raw_text)
        embeddings = model.encode(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))

        query_embedding = model.encode([question])
        D, I = index.search(np.array(query_embedding), k=5)
        context = " ".join([chunks[i] for i in I[0]])
        result = qa_pipeline(question=question, context=context)

        return jsonify({"answer": result["answer"]})
    finally:
        os.remove(file_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
