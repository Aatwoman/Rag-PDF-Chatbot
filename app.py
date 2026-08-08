"""
app.py
Streamlit UI for the RAG PDF Q&A chatbot.
"""

import os

import streamlit as st

from rag_pipeline import (
    build_vectorstore,
    collection_exists,
    get_answer,
    load_and_chunk_pdf,
    load_existing_vectorstore,
)
from utils import file_hash, format_sources, save_uploaded_file, truncate_history

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄", layout="wide")

st.title("📄 RAG PDF Q&A Chatbot")
st.caption("Upload a PDF, ask questions, get answers grounded in the document with page citations.")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("Set OPENAI_API_KEY in your .env file (see .env.example) before asking questions.", icon="⚠️")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        collection_name = f"doc_{file_hash(file_bytes)}"

        if st.session_state.current_file != collection_name:
            with st.spinner("Processing PDF (chunking + embedding)..."):
                if collection_exists(collection_name):
                    vectorstore = load_existing_vectorstore(collection_name)
                else:
                    tmp_path = save_uploaded_file(uploaded_file)
                    chunks = load_and_chunk_pdf(tmp_path)
                    vectorstore = build_vectorstore(chunks, collection_name)
                    os.remove(tmp_path)

            st.session_state.vectorstore = vectorstore
            st.session_state.current_file = collection_name
            st.session_state.messages = []
            st.success(f"Loaded **{uploaded_file.name}** — ready for questions.")

    st.divider()
    top_k = st.slider("Chunks to retrieve (k)", min_value=2, max_value=10, value=4)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something about the document...")

if question:
    if st.session_state.vectorstore is None:
        st.error("Upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history = truncate_history(st.session_state.messages)
                answer, sources = get_answer(st.session_state.vectorstore, question, k=top_k)
                st.markdown(answer)
                with st.expander("Show sources"):
                    st.markdown(format_sources(sources))

        st.session_state.messages.append({"role": "assistant", "content": answer})
