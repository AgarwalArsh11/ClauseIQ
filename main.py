import gradio as gr
import pdfplumber
from transformers import pipeline

# Load a Question Answering model (use your own if different)
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Function to extract text from uploaded PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

# Main function for answering
def answer_question(question, pdf_file=None, context_text=None):
    if pdf_file is not None:
        context = extract_text_from_pdf(pdf_file)
    elif context_text:
        context = context_text
    else:
        return "Please provide either a PDF file or paste text context."

    if not context.strip():
        return "Could not extract context from PDF."

    result = qa_pipeline(question=question, context=context)
    return result["answer"]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📄 ClauseIQ - Legal Document Q&A System")
    gr.Markdown("Ask questions about legal, HR, insurance, or compliance documents. Upload a PDF or paste text.")

    with gr.Row():
        question_input = gr.Textbox(label="Ask your question")
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF (optional)", type="file")
        context_input = gr.Textbox(label="Or paste context text", lines=10, placeholder="Paste document content here...")

    output = gr.Textbox(label="Answer")

    submit_btn = gr.Button("Submit")
    submit_btn.click(fn=answer_question, inputs=[question_input, pdf_input, context_input], outputs=output)

demo.launch()
