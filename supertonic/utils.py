"""Utility functions for Supertonic TTS.

This module provides various helper functions for text processing, file operations,
and timing operations used throughout the Supertonic TTS package.
"""

from __future__ import annotations

import logging
import os
import re
import time
import unicodedata
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Common abbreviations that should not trigger sentence splitting
_COMMON_ABBREVIATIONS_PATTERN = (
    r"(?<!Mr\.)"
    r"(?<!Mrs\.)"
    r"(?<!Ms\.)"
    r"(?<!Dr\.)"
    r"(?<!Prof\.)"
    r"(?<!Sr\.)"
    r"(?<!Jr\.)"
    r"(?<!Ph\.D\.)"
    r"(?<!etc\.)"
    r"(?<!e\.g\.)"
    r"(?<!i\.e\.)"
    r"(?<!vs\.)"
    r"(?<!Inc\.)"
    r"(?<!Ltd\.)"
    r"(?<!Co\.)"
    r"(?<!Corp\.)"
    r"(?<!St\.)"
    r"(?<!Ave\.)"
    r"(?<!Blvd\.)"
    r"(?<!\b[A-Z]\.)"  # Single capital letter abbreviations
    r"(?:(?<=[.!?])\s+|(?<=[。！？])\s*)"  # ASCII or CJK sentence boundaries
)


def _sentence_segments(paragraph: str) -> list[tuple[str, str]]:
    """Return trimmed sentences with their normalized preceding separator."""
    segments = []
    start = 0
    separator = ""
    for match in re.finditer(_COMMON_ABBREVIATIONS_PATTERN, paragraph):
        sentence = paragraph[start : match.start()].strip()
        if sentence:
            segments.append((sentence, separator))
        separator = " " if match.group() else ""
        start = match.end()

    sentence = paragraph[start:].strip()
    if sentence:
        segments.append((sentence, separator))
    return segments


def _hangul_type(char: str) -> str:
    codepoint = ord(char)
    if 0x1100 <= codepoint <= 0x115F or 0xA960 <= codepoint <= 0xA97C:
        return "L"
    if 0x1160 <= codepoint <= 0x11A7 or 0xD7B0 <= codepoint <= 0xD7C6:
        return "V"
    if 0x11A8 <= codepoint <= 0x11FF or 0xD7CB <= codepoint <= 0xD7FB:
        return "T"
    if 0xAC00 <= codepoint <= 0xD7A3:
        return "LV" if (codepoint - 0xAC00) % 28 == 0 else "LVT"
    return "other"


def _safe_unicode_boundary(text: str, max_len: int) -> int:
    """Find a code-point boundary that avoids splitting common grapheme forms."""

    def is_extend(char: str) -> bool:
        codepoint = ord(char)
        return (
            unicodedata.combining(char) != 0
            or unicodedata.category(char) in {"Mn", "Mc", "Me"}
            or 0x1F3FB <= codepoint <= 0x1F3FF  # emoji skin-tone modifiers
            or 0xE0020 <= codepoint <= 0xE007F  # emoji tag characters
        )

    def is_regional_indicator(char: str) -> bool:
        return 0x1F1E6 <= ord(char) <= 0x1F1FF

    for boundary in range(min(max_len, len(text) - 1), 0, -1):
        left = text[boundary - 1]
        right = text[boundary]
        if is_extend(right) or left == "\u200d" or right == "\u200d":
            continue

        if is_regional_indicator(left) and is_regional_indicator(right):
            preceding_indicators = 0
            for char in reversed(text[:boundary]):
                if not is_regional_indicator(char):
                    break
                preceding_indicators += 1
            if preceding_indicators % 2:
                continue

        left_hangul = _hangul_type(left)
        right_hangul = _hangul_type(right)
        if left_hangul == "L" and right_hangul in {"L", "V", "LV", "LVT"}:
            continue
        if left_hangul in {"LV", "V"} and right_hangul in {"V", "T"}:
            continue
        if left_hangul in {"LVT", "T"} and right_hangul == "T":
            continue
        return boundary

    # If a joined grapheme itself exceeds the configured bound, split at the
    # code-point limit to keep the public maximum strict.
    return min(max_len, len(text) - 1)


def _split_overlong_sentence(sentence: str, max_len: int) -> list[str]:
    """Split a sentence at whitespace, then at safe bounded Unicode boundaries."""
    remaining = sentence.strip()
    chunks = []
    while len(remaining) > max_len:
        whitespace_boundaries = [
            match.start() for match in re.finditer(r"\s+", remaining[: max_len + 1])
        ]
        boundary = whitespace_boundaries[-1] if whitespace_boundaries else 0
        if boundary:
            chunks.append(remaining[:boundary].rstrip())
            remaining = remaining[boundary:].lstrip()
        else:
            boundary = _safe_unicode_boundary(remaining, max_len)
            chunks.append(remaining[:boundary])
            remaining = remaining[boundary:]

    if remaining:
        chunks.append(remaining)
    return chunks


def sanitize_filename(text: str, max_len: int = 50) -> str:
    """
    Sanitize filename by replacing non-alphanumeric characters.

    Args:
        text: Input text to convert to filename
        max_len: Maximum length of filename

    Returns:
        Sanitized filename string
    """
    prefix = text[:max_len]
    return re.sub(r"[^a-zA-Z0-9_-]", "_", prefix)


@contextmanager
def timer(name: str, verbose: bool = True):
    """
    Context manager for timing code execution.

    Args:
        name: Name of the operation being timed
        verbose: Whether to log timing information

    Example:
        ```python
        with timer("Processing"):
            # Your code here
            process_data()
        ```
    """
    if verbose:
        logger.info(f"{name}...")
    start = time.time()
    yield
    elapsed = time.time() - start
    if verbose:
        logger.info(f"{name} completed in {elapsed:.2f}s")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1.23s", "2m 30s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def get_audio_duration(wav_length: int, sample_rate: int) -> float:
    """
    Calculate audio duration from waveform length.

    Args:
        wav_length: Number of samples in waveform
        sample_rate: Audio sample rate (Hz)

    Returns:
        Duration in seconds
    """
    return wav_length / sample_rate


def ensure_dir(path: str) -> str:
    """
    Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Absolute path to directory
    """
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def validate_voice_style_format(style_data: dict) -> bool:
    """
    Validate voice style JSON format.

    Args:
        style_data: Voice style dictionary

    Returns:
        True if valid, False otherwise
    """
    required_keys = ["style_ttl", "style_dp"]
    if not all(key in style_data for key in required_keys):
        return False

    for key in required_keys:
        if "dims" not in style_data[key] or "data" not in style_data[key]:
            return False

    return True


def chunk_text(text: str, max_len: int = 300) -> list[str]:
    """
    Split text into chunks by paragraphs and sentences.

    This function intelligently splits long text into smaller chunks suitable
    for TTS processing, respecting paragraph and sentence boundaries. Long
    sentences split at whitespace and then at bounded Unicode boundaries.

    Args:
        text: Input text to chunk
        max_len: Maximum length of each chunk in characters (default: 300)

    Returns:
        List of text chunks

    Example:
        ```python
        text = "This is a long paragraph. It has multiple sentences. " * 10
        chunks = chunk_text(text, max_len=100)
        for chunk in chunks:
            print(f"Chunk ({len(chunk)} chars): {chunk[:50]}...")
        ```
    """
    # Validate minimum chunk length
    if max_len < 10:
        raise ValueError(
            f"max_len must be at least 10, got {max_len}. "
            f"Very small chunks may produce poor quality speech."
        )

    # Split by paragraph (two or more newlines)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]

    chunks = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        sentences = _sentence_segments(paragraph)

        current_chunk = ""

        for sentence, separator in sentences:
            if len(sentence) > max_len:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.extend(_split_overlong_sentence(sentence, max_len))
                continue

            joiner = separator if current_chunk else ""
            if len(current_chunk) + len(joiner) + len(sentence) <= max_len:
                current_chunk += joiner + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        # Add final chunk if not empty
        if current_chunk and current_chunk.strip():
            chunks.append(current_chunk.strip())

    return chunks
