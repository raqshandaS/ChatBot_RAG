# brain.py
from io import BytesIO
import re
from typing import List, Tuple
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from pypdf import PdfReader


def parse_pdf(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text() or ""

        # Wikipedia cleanup
        text = re.sub(r"\[\d+\]", "", text)       # Remove citations like [1]
        text = re.sub(r"\s?\[edit\]", "", text)   # Remove [edit]
        text = re.sub(r"Page \d+", "", text)      # Remove page numbers

        # Formatting cleanup
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output, filename


def text_to_docs(text: List[str], filename: str) -> List[Document]:
    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Wikipedia-specific: short facts
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )

    for doc in page_docs:
        chunks = splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            new_doc = Document(
                page_content=chunk,
                metadata={
                    "page": doc.metadata["page"],
                    "chunk": i,
                    "filename": filename,
                },
            )
            doc_chunks.append(new_doc)
    return doc_chunks


def docs_to_index(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = FAISS.from_documents(docs, embeddings)
    return index


def get_index_for_pdf(pdf_files, pdf_names, openai_api_key=None):
    documents = []
    for pdf_file, pdf_name in zip(pdf_files, pdf_names):
        text, filename = parse_pdf(BytesIO(pdf_file), pdf_name)
        documents += text_to_docs(text, filename)
    index = docs_to_index(documents)
    return index
