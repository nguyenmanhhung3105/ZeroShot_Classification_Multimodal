"""Sentence-first input chunking; no labels, prompt changes or learned parameters."""
import re
import numpy as np


def token_counter(vlm):
    tokenizer = getattr(vlm, "tokenizer", None)
    context = getattr(getattr(vlm, "model", None), "context_length", None)
    if not isinstance(context, int) or context < 1:
        raise ValueError("Chunking requires model.context_length")
    name = type(tokenizer).__name__
    if name == "SimpleTokenizer" and not getattr(tokenizer, "reduction_fn", None):
        count = lambda text: len(tokenizer.encode(text.strip())) + 2
    elif name == "HFTokenizer" and not getattr(tokenizer, "tokenizer_mode", ""):
        count = lambda text: len(tokenizer.tokenizer.encode(
            tokenizer.clean_fn(text.strip()), add_special_tokens=True, truncation=False))
    else:
        raise ValueError(f"Chunking unsupported tokenizer: {name}; refusing silent truncation")
    return count, context


def split_text(text, count, context):
    """Keep sentences when possible; fall back to words then Unicode characters.

    Whitespace may be normalized at joins; no non-whitespace content is dropped.
    Every emitted chunk is checked with the tokenizer INCLUDING special tokens.
    """
    text = text.strip()
    if not text:
        return []
    if count(text) <= context:
        return [text]
    chunks, current = [], ""

    def add(piece, separator=" "):
        nonlocal current
        candidate = current + separator + piece if current else piece
        if count(candidate) <= context:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if count(piece) > context:
                raise ValueError("A single character exceeds tokenizer context")
            current = piece

    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if not sentence.strip():
            continue
        if count(sentence) <= context:
            add(sentence)
        else:
            for word in sentence.split():
                if count(word) <= context:
                    add(word)
                else:
                    # Start an oversized word separately, then preserve its characters.
                    if current:
                        chunks.append(current)
                        current = ""
                    for char in word:
                        add(char, separator="")
    if current:
        chunks.append(current)
    if any(count(chunk) > context for chunk in chunks):
        raise ValueError("Chunk exceeded context length")
    return chunks


def encode_chunked_texts(vlm, texts, batch_size):
    """Bound encoder calls by batch_size; retain only a running embedding sum."""
    count, context = token_counter(vlm)
    embeddings, counts = [], []
    for text in texts:
        chunks = split_text(text, count, context)
        if not chunks:
            raise ValueError("Empty texts must use the existing image fallback")
        total = None
        for start in range(0, len(chunks), batch_size):
            values = vlm.encode_texts(chunks[start:start + batch_size]).detach().cpu().numpy()
            if len(chunks) == 1:
                total = values[0]  # Preserve the short-text baseline exactly.
            else:
                values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
                subtotal = values.sum(axis=0)
                total = subtotal if total is None else total + subtotal
        if len(chunks) > 1:
            total = total / len(chunks)
            norm = np.linalg.norm(total)
            if not np.isfinite(norm) or norm <= 1e-12:
                raise ValueError("Cannot normalize mean chunk embedding")
            total = total / norm
        embeddings.append(total)
        counts.append(len(chunks))
    return np.stack(embeddings), counts
