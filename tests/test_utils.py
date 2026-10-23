"""Tests for utility functions."""

import pytest

from supertonic.utils import (
    chunk_text,
    ensure_dir,
    format_duration,
    get_audio_duration,
    sanitize_filename,
    validate_voice_style_format,
)


def test_sanitize_filename():
    """Test filename sanitization."""
    # Test basic sanitization
    assert sanitize_filename("hello world") == "hello_world"
    assert sanitize_filename("test@file#name") == "test_file_name"

    # Test max length
    long_name = "a" * 100
    result = sanitize_filename(long_name, max_len=50)
    assert len(result) == 50


def test_format_duration():
    """Test duration formatting."""
    # Short duration
    assert format_duration(1.5) == "1.50s"
    assert format_duration(45.2) == "45.20s"

    # Minutes
    assert format_duration(90) == "1m 30s"
    assert format_duration(150) == "2m 30s"

    # Hours
    assert format_duration(3665) == "1h 1m"
    assert format_duration(7200) == "2h 0m"


def test_get_audio_duration():
    """Test audio duration calculation."""
    # 24000 samples at 24000 Hz = 1 second
    duration = get_audio_duration(24000, 24000)
    assert duration == 1.0

    # 48000 samples at 24000 Hz = 2 seconds
    duration = get_audio_duration(48000, 24000)
    assert duration == 2.0


def test_ensure_dir(tmp_path):
    """Test directory creation."""
    test_dir = tmp_path / "test_dir" / "nested"
    result = ensure_dir(str(test_dir))

    assert test_dir.exists()
    assert test_dir.is_dir()
    assert result == str(test_dir.absolute())


def test_validate_voice_style_format_valid():
    """Test voice style format validation with valid data."""
    valid_style = {
        "style_ttl": {"dims": [1, 512], "data": [0.1] * 512},
        "style_dp": {"dims": [1, 256], "data": [0.2] * 256},
    }

    assert validate_voice_style_format(valid_style) is True


def test_validate_voice_style_format_missing_keys():
    """Test validation fails with missing keys."""
    # Missing style_dp
    invalid_style = {"style_ttl": {"dims": [1, 512], "data": [0.1] * 512}}

    assert validate_voice_style_format(invalid_style) is False


def test_validate_voice_style_format_missing_fields():
    """Test validation fails with missing fields."""
    # Missing 'dims' field
    invalid_style = {
        "style_ttl": {"data": [0.1] * 512},
        "style_dp": {"dims": [1, 256], "data": [0.2] * 256},
    }

    assert validate_voice_style_format(invalid_style) is False


# 이슈 #7, #8: chunk_text 테스트
def test_chunk_text_min_length_validation():
    """Test that chunk_text validates minimum length."""
    with pytest.raises(ValueError, match="at least 10"):
        chunk_text("Some text", max_len=5)

    with pytest.raises(ValueError, match="at least 10"):
        chunk_text("Some text", max_len=0)

    with pytest.raises(ValueError, match="at least 10"):
        chunk_text("Some text", max_len=-1)


def test_chunk_text_min_length_boundary():
    """Test chunk_text with minimum valid length."""
    text = "Short."
    chunks = chunk_text(text, max_len=10)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_no_empty_chunks():
    """Test that chunk_text doesn't produce empty chunks."""
    # Text with lots of whitespace and punctuation
    text = "Hello.  \n\n  World.  \n  Test."
    chunks = chunk_text(text, max_len=50)

    # All chunks should be non-empty after stripping
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) > 0 for chunk in chunks)


def test_chunk_text_consecutive_punctuation():
    """Test chunk_text with consecutive punctuation."""
    text = "First sentence...  Second sentence!!  Third sentence??"
    chunks = chunk_text(text, max_len=100)

    assert len(chunks) >= 1
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_text_only_whitespace():
    """Test chunk_text with only whitespace."""
    text = "   \n\n\t  "
    chunks = chunk_text(text, max_len=100)

    # Should return empty list (no valid chunks)
    assert chunks == []


def test_chunk_text_empty_after_split():
    """Test chunk_text where sentences become empty after stripping."""
    text = "   .  .  .  "
    chunks = chunk_text(text, max_len=100)

    # Should handle gracefully without creating empty chunks
    for chunk in chunks:
        assert len(chunk.strip()) > 0


def test_chunk_text_splits_overlong_tokens_and_obeys_exact_limit():
    exact = "x" * 10
    overlong = exact + "y"

    assert chunk_text("x" * 9, max_len=10) == ["x" * 9]
    assert chunk_text(exact, max_len=10) == [exact]
    chunks = chunk_text(overlong, max_len=10)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 10 for chunk in chunks)
    assert "".join(chunks) == overlong


def test_chunk_text_splits_long_unpunctuated_words_at_whitespace():
    text = "word " * 40
    chunks = chunk_text(text, max_len=20)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 20 for chunk in chunks)
    assert "".join("".join(chunks).split()) == "".join(text.split())


def test_chunk_text_bounds_japanese_without_ascii_spaces():
    text = "日本語" * 50
    chunks = chunk_text(text, max_len=31)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 31 for chunk in chunks)
    assert "".join(chunks) == text


def test_chunk_text_keeps_combining_emoji_and_hangul_sequences_together():
    decomposed = "x" * 9 + "e\u0301" + "z" * 10
    emoji = "👨‍👩‍👧‍👦"
    emoji_text = "x" * 9 + emoji + "z" * 10
    tag_emoji = "🏴\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f"
    tag_text = "x" * 9 + tag_emoji + "z" * 10
    regional_flag = "🇺🇸"
    flag_text = "x" * 9 + regional_flag + "z" * 10
    hangul_cluster = "\u1100\u1161\u11a8"
    hangul_text = "x" * 8 + hangul_cluster * 3 + "z" * 5
    combined_hangul = "안녕하세요" * 4

    for text, cluster in (
        (decomposed, "e\u0301"),
        (emoji_text, emoji),
        (tag_text, tag_emoji),
        (flag_text, regional_flag),
        (hangul_text, hangul_cluster),
        (combined_hangul, "안녕"),
    ):
        chunks = chunk_text(text, max_len=10)
        assert all(0 < len(chunk) <= 10 for chunk in chunks)
        assert "".join(chunks) == text
        assert any(cluster in chunk for chunk in chunks)


def test_chunk_text_hard_splits_an_indivisible_cluster_over_the_limit():
    long_joined_sequence = "👨‍" + ("👩‍" * 5) + "👧"
    chunks = chunk_text(long_joined_sequence, max_len=10)

    assert all(0 < len(chunk) <= 10 for chunk in chunks)
    assert "".join(chunks) == long_joined_sequence


def test_chunk_text_recognizes_unicode_sentence_punctuation_and_keeps_paragraphs():
    japanese = "日本語です。次の文です！おしまい？"
    assert chunk_text(japanese, max_len=100) == [japanese]

    long_japanese = "日本語です。" * 25
    long_chunks = chunk_text(long_japanese, max_len=20)
    assert all(0 < len(chunk) <= 20 for chunk in long_chunks)
    assert "".join(long_chunks) == long_japanese

    text = "Dr. Smith arrived. He left.\n\n次の段落。まだ続く。"
    chunks = chunk_text(text, max_len=80)
    assert chunks == ["Dr. Smith arrived. He left.", "次の段落。まだ続く。"]
