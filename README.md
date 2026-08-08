# 📄 RAG PDF Q&A Chatbot

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-6E56CF)
![License](https://img.shields.io/badge/License-MIT-green)

Upload any PDF and ask it questions in plain English. Answers are grounded strictly in the document's content, with page-level citations so you can verify every claim.

## Demo

> _Add a screenshot or short GIF here: `docs/demo.gif`_

## How it works

```
PDF Upload
   │
   ▼
PyPDFLoader ── extracts text per page
   │
   ▼
RecursiveCharacterTextSplitter ── chunks (1000 chars, 200 overlap)
   │
   ▼
OpenAI Embeddings ── text-embedding-3-small
   │
   ▼
ChromaDB ── persisted vector store (per-document collection)
   │
   ▼
MMR Retriever ── top-k diverse chunks
   │
   ▼
gpt-4o-mini ── answer grounded in retrieved context
   │
   ▼
Streamlit Chat UI ── answer + page citations
```

## Features

- Drag-and-drop PDF upload with automatic chunking and embedding
- Persistent per-document vector store — re-opening the same PDF skips re-embedding
- MMR retrieval to reduce redundant context and improve answer diversity
- Page-level source citations shown alongside every answer
- Adjustable retrieval depth (`k`) from the sidebar
- Chat history within a session, with a one-click reset

## Project structure

```
rag-pdf-chatbot/
├── app.py              # Streamlit UI
├── rag_pipeline.py      # Chunking, embedding, retrieval, generation
├── utils.py              # File handling, citation formatting
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/rag-pdf-chatbot.git
cd rag-pdf-chatbot
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY
streamlit run app.py
```

## Tech stack

`LangChain` · `ChromaDB` · `OpenAI API` · `Streamlit` · `PyPDF`

## Possible extensions

- Swap OpenAI embeddings for a local `sentence-transformers` model to run fully offline
- Add multi-document support with a document picker
- Highlight the exact retrieved passage on the PDF page (not just the page number)
- Add conversation-aware retrieval (rephrase follow-up questions using chat history)

