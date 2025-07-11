# app.py
import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import tempfile
from brain import get_index_for_pdf

# Set up the Streamlit app
st.set_page_config(page_title="RAG Chat with PDF", layout="wide")
st.title(" RAG Chat UCM")

# Save uploaded file to a temporary path
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name

# Load local model
def load_llm():
    import torch
    device = 0 if torch.cuda.is_available() else -1

    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        tokenizer="google/flan-t5-small",
        max_length=512,
        do_sample=False,
        device=device,
    )
    return HuggingFacePipeline(pipeline=pipe)

# Main Streamlit logic
def main():
    uploaded_file = st.file_uploader("Upload a Wikipedia PDF", type=["pdf"])
    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")
        bytes_data = uploaded_file.read()

        with st.spinner("Processing PDF and creating vector index..."):
            index = get_index_for_pdf([bytes_data], [uploaded_file.name])

        llm = load_llm()
        retriever = index.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        query = st.text_input("Ask a question from the Wikipedia article:")
        if query:
            result = qa_chain.invoke({"query": query})
            st.markdown(f"**Answer:** {result['result']}")

if __name__ == "__main__":
    main()
