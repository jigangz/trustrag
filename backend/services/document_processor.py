"""PDF document parsing and chunking service.

Handles the ingestion pipeline:
1. Extract text from PDF pages using pdfplumber
2. Split text into semantically meaningful chunks
3. Track page numbers and positions for source attribution
"""

import io
import re

import pdfplumber
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_enc.encode(text))


async def parse_pdf(file_bytes: bytes) -> list[dict]:
    """
    Parse a PDF file into structured page content.

    Returns:
        List of dicts: [{page_number, text}, ...]
    """
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"page_number": i + 1, "text": text})
    return pages


def chunk_text(pages: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Split page text into overlapping chunks for embedding.

    Each chunk preserves its source page number and position
    so we can trace answers back to exact locations.

    Returns:
        List of dicts: [{content, page_number, chunk_index}, ...]
    """
    # Split all pages into sentences with page tracking
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    sentences: list[tuple[str, int]] = []  # (sentence_text, page_number)

    for page in pages:
        page_sentences = sentence_pattern.split(page["text"])
        for s in page_sentences:
            s = s.strip()
            if s:
                sentences.append((s, page["page_number"]))

    if not sentences:
        return []

    chunks = []
    chunk_index = 0
    i = 0

    while i < len(sentences):
        chunk_sentences = []
        chunk_tokens = 0
        start_i = i

        # Build a chunk up to chunk_size tokens
        while i < len(sentences) and chunk_tokens < chunk_size:
            s_text, _ = sentences[i]
            s_tokens = _token_len(s_text)
            if chunk_sentences and chunk_tokens + s_tokens > chunk_size:
                break
            chunk_sentences.append(i)
            chunk_tokens += s_tokens
            i += 1

        if not chunk_sentences:
            break

        content = " ".join(sentences[idx][0] for idx in chunk_sentences)
        # Use the page number of the first sentence in the chunk
        page_number = sentences[chunk_sentences[0]][1]

        chunks.append({
            "content": content,
            "page_number": page_number,
            "chunk_index": chunk_index,
        })
        chunk_index += 1

        # Overlap: step back by roughly `overlap` tokens worth of sentences
        overlap_tokens = 0
        rewind = 0
        for j in range(len(chunk_sentences) - 1, -1, -1):
            s_tokens = _token_len(sentences[chunk_sentences[j]][0])
            if overlap_tokens + s_tokens > overlap:
                break
            overlap_tokens += s_tokens
            rewind += 1

        if rewind > 0 and i < len(sentences):
            i = i - rewind

    return chunks
