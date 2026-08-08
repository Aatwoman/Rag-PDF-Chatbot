"""
rag_pipeline.py
Core RAG logic: load PDF -> chunk -> embed -> store in Chroma -> retrieve -> generate.

Design notes:
- Each uploaded PDF gets its own Chroma collection, keyed by a content hash,
  so re-uploading the same file doesn't re-embed it, and different files
  don't bleed into each other's context.
- Retrieval uses MMR (max marginal relevance) instead of plain similarity
  search to reduce redundant chunks in the context window.
"""

import os
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
PERSIST_ROOT = "chroma_db"

SYSTEM_PROMPT = """You are a careful research assistant. Answer the user's \
question using ONLY the context below, which was retrieved from a PDF the \
user uploaded. If the answer isn't in the context, say you don't know — \
do not make anything up.

Context:
{context}

Question: {question}

Answer clearly and concisely. Where relevant, mention which part of the \
document (by topic, not page number) the information came from — page \
numbers are already shown separately to the user."""


def load_and_chunk_pdf(file_path: str) -> List[Document]:
    """Load a PDF and split it into overlapping chunks."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(pages)


def build_vectorstore(chunks: List[Document], collection_name: str) -> Chroma:
    """Embed chunks and persist them in a Chroma collection."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    persist_dir = os.path.join(PERSIST_ROOT, collection_name)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    return vectorstore


def load_existing_vectorstore(collection_name: str) -> Chroma:
    """Reconnect to a previously-built collection instead of re-embedding."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    persist_dir = os.path.join(PERSIST_ROOT, collection_name)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def collection_exists(collection_name: str) -> bool:
    return os.path.isdir(os.path.join(PERSIST_ROOT, collection_name))


def _format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def get_answer(vectorstore: Chroma, question: str, k: int = 4) -> Tuple[str, List[Document]]:
    """
    Run the retrieve -> generate chain for a single question.
    Returns (answer_text, retrieved_documents) so the caller can render citations.
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3},
    )
    retrieved_docs = retriever.invoke(question)

    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    chain = (
        {
            "context": lambda x: _format_docs(retrieved_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    return answer, retrieved_docs
