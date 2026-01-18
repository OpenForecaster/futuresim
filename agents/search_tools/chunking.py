"""Text chunking utilities."""

from dataclasses import dataclass
from typing import List, Tuple
import re


@dataclass
class Chunk:
    text: str
    start_char: int
    end_char: int


def chunk_text(text: str, max_tokens: int = 512, chars_per_token: float = 4.0) -> List[Chunk]:
    """Split text into non-overlapping chunks at sentence boundaries."""
    if not text or not text.strip():
        return []
    
    max_chars = int(max_tokens * chars_per_token)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    chunks = []
    current_text = ""
    current_start = 0
    char_pos = 0
    
    for sentence in sentences:
        if len(current_text) + len(sentence) + 1 > max_chars and current_text:
            chunks.append(Chunk(current_text.strip(), current_start, char_pos))
            current_text = sentence
            current_start = char_pos
        else:
            current_text = f"{current_text} {sentence}".strip() if current_text else sentence
        char_pos += len(sentence) + 1
    
    if current_text.strip():
        chunks.append(Chunk(current_text.strip(), current_start, char_pos))
    
    return chunks


def chunk_article(article_id: str, title: str, content: str, max_tokens: int = 512) -> List[Tuple[str, str]]:
    """Chunk article content, prepending title to each chunk. Returns [(chunk_id, text), ...]"""
    if not content or not content.strip():
        return [(f"{article_id}_chunk0", title)] if title else []
    
    chunks = chunk_text(content, max_tokens)
    if not chunks:
        return [(f"{article_id}_chunk0", f"{title}\n\n{content}" if title else content)]
    
    return [(f"{article_id}_chunk{i}", f"{title}\n\n{c.text}" if title else c.text) 
            for i, c in enumerate(chunks)]
