This project allows you to upload a Wikipedia-style PDF and ask questions about its content using Retrieval-Augmented Generation (RAG). It uses LangChain, FAISS, and Hugging Face Transformers to extract, process, and search through the text, then answer your queries using a local language model.

## Features

- Upload and process Wikipedia-style PDFs
- Automatic text cleaning (removes citations, edit links, page numbers)
- Smart chunking of long text for better context retention
- Semantic search using FAISS and MiniLM sentence embeddings
- Local question-answering with Hugging Face's FLAN-T5
- Streamlit interface for simple and interactive usage

## How It Works

1. Upload a PDF
2. Text is extracted and cleaned
3. The text is split into smaller, overlapping chunks
4. Chunks are converted into embeddings (vectors)
5. FAISS indexes these vectors for fast similarity search
6. Relevant chunks are retrieved for your query
7. A local model generates an answer based on the retrieved content

## Tech Stack

- LangChain
- FAISS
- Hugging Face Transformers
- sentence-transformers/all-MiniLM-L6-v2
- FLAN-T5 (`google/flan-t5-small`)
- Streamlit
- PyPDF

## Installation

1. Clone the repository:
git clone https://github.com/your-username/rag-chat-pdf.git

cd rag-chat-pdf

## Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  #_On Windows: venv\Scripts\activate_

## Install dependencies:
pip install -r requirements.txt

## Running the App
streamlit run app.py

