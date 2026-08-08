"""
utils.py
Helper functions for file handling, text cleanup, and citation formatting.
"""

import hashlib
import os
import tempfile
from typing import List

from langchain_core.documents import Document


def save_uploaded_file(uploaded_file) -> str:
    """
    Persist a Streamlit UploadedFile object to a temp path on disk so that
    LangChain's PyPDFLoader (which needs a filesystem path) can read it.

    Returns the path to the saved temp file.
    """
    suffix = os.path.splitext(uploaded_file.name)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def file_hash(file_bytes: bytes) -> str:
    """Stable hash used as a collection name / cache key per uploaded file."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def format_sources(documents: List[Document]) -> str:
    """
    Turn retrieved chunks into a human-readable citation block, e.g.:

        **Sources**
        - Page 3
        - Page 7
        - Page 12
    """
    if not documents:
        return "_No sources retrieved._"

    seen = set()
    lines = []
    for doc in documents:
        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", "document")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        # PyPDFLoader pages are 0-indexed
        display_page = page + 1 if isinstance(page, int) else page
        lines.append(f"- Page {display_page}")

    return "**Sources**\n" + "\n".join(lines)


def truncate_history(messages: list, max_turns: int = 6) -> list:
    """Keep only the last N turns so the prompt doesn't grow unbounded."""
    return messages[-max_turns * 2:] if len(messages) > max_turns * 2 else messages


def clean_answer(text: str) -> str:
    """Strip stray markdown fences the LLM sometimes wraps answers in."""
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`").strip()
    return text
