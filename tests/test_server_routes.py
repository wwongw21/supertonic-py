"""Tests for the optional ``supertonic.server`` subpackage.

The real ``TTS`` is replaced with a tiny stand-in so these tests never touch
ONNX Runtime or download a model. Every endpoint is exercised against the
stand-in to verify request validation, response shape, and the OpenAI-shaped
error envelope.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import numpy as np
import pytest

# ``fastapi`` is an optional install (the ``[serve]`` extra). Skip the whole
# module gracefully when it is not present so the SDK-only test environment
# stays green.
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from supertonic.server import ServerState, create_app  # noqa: E402
from supertonic.server.routes import MAX_STYLE_IMPORT_BYTES  # noqa: E402
from supertonic.server.schemas import BatchItem  # noqa: E402

SAMPLE_RATE = 44100


class _FakeStyle:
    def __init__(self, source: str) -> None:
        self.source = source


class FakeTTS:
    """Minimal stand-in for :class:`supertonic.TTS` used in route tests.

    Records calls so individual tests can assert how the route forwarded
    parameters to the underlying synthesizer.
    """

    def __init__(self) -> None:
        self.sample_rate = SAMPLE_RATE
        self.voice_style_names = ["M1", "F1", "F2"]
        self.calls: list[dict] = []

    # --- voice style resolution -----------------------------------------
    def get_voice_style(self, name: str) -> _FakeStyle:
        if name not in self.voice_style_names:
            # mirror what the real loader raises for unknown built-in names
            raise FileNotFoundError(f"voice not found: {name}")
        return _FakeStyle(source=f"builtin:{name}")

    def get_voice_style_from_path(self, path) -> _FakeStyle:
        return _FakeStyle(source=f"custom:{path}")

    # --- synthesize ------------------------------------------------------
    def synthesize(
        self,
        text: str,
        voice_style,
        total_steps: int = 8,
        speed: float = 1.05,
        max_chunk_length=None,
        silence_duration: float = 0.3,
        lang=None,
        verbose: bool = False,
    ):
        self.calls.append(
            {
                "text": text,
                "voice_style_source": getattr(voice_style, "source", None),
                "total_steps": total_steps,
                "speed": speed,
                "max_chunk_length": max_chunk_length,
                "silence_duration": silence_duration,
                "lang": lang,
            }
        )
        # 0.1 s of silence — small enough to keep the test fast, large enough
        # to verify the duration header rounds sensibly.
        n = int(SAMPLE_RATE * 0.1)
        wav = np.zeros((1, n), dtype=np.float32)
        duration = np.array([0.1], dtype=np.float32)
        return wav, duration


@pytest.fixture
def fake_state(tmp_path):
    fake = FakeTTS()
    state = ServerState(
        model="supertonic-3",
        tts=fake,
        custom_styles_dir=tmp_path,
    )
    return state, fake


@pytest.fixture
def client(fake_state):
    state, _ = fake_state
    app = create_app(state=state)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /v1/health
# ---------------------------------------------------------------------------


def test_health_ok(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "supertonic-3"
    assert body["sample_rate"] == SAMPLE_RATE
    assert body["voices_loaded"] == 3  # M1, F1, F2


# ---------------------------------------------------------------------------
# /v1/styles + /v1/styles/import
# ---------------------------------------------------------------------------


def test_list_styles_lists_builtins(client):
    r = client.get("/v1/styles")
    assert r.status_code == 200
    names = {s["name"]: s for s in r.json()["styles"]}
    assert {"M1", "F1", "F2"}.issubset(names.keys())
    assert all(names[n]["kind"] == "builtin" for n in ["M1", "F1", "F2"])


def _valid_style_payload() -> dict:
    return {
        "style_ttl": {"dims": [1, 4], "data": [0.0, 0.0, 0.0, 0.0]},
        "style_dp": {"dims": [1, 4], "data": [0.0, 0.0, 0.0, 0.0]},
    }


def _json_body_at_size(size: int, name: str = "boundary_voice") -> bytes:
    body = {**_valid_style_payload(), "name": name, "ignored": ""}
    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
    assert size >= len(encoded)
    body["ignored"] = "x" * (size - len(encoded))
    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
    assert len(encoded) == size
    return encoded


def _chunks(body: bytes, size: int = 64 * 1024):
    return (body[offset : offset + size] for offset in range(0, len(body), size))


def _multipart_body(payload: bytes, *, boundary: bytes = b"supertonic-test") -> bytes:
    return (
        b"--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="file"; filename="voice.json"\r\n'
        + b"Content-Type: application/json\r\n\r\n"
        + payload
        + b"\r\n--"
        + boundary
        + b"--\r\n"
    )


def test_import_style_via_multipart(client, fake_state):
    state, _ = fake_state
    payload = _valid_style_payload()
    files = {"file": ("custom_voice.json", json.dumps(payload), "application/json")}
    r = client.post("/v1/styles/import", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "custom_voice"
    assert Path(body["stored_at"]).exists()

    # second import without overwrite → 409
    r2 = client.post("/v1/styles/import", files=files)
    assert r2.status_code == 409
    assert r2.json()["error"]["code"] == "style_name_conflict"

    # with ?overwrite=true → 200
    r3 = client.post("/v1/styles/import?overwrite=true", files=files)
    assert r3.status_code == 200

    # newly imported style now appears in /v1/styles
    listed = client.get("/v1/styles").json()["styles"]
    custom_names = [s["name"] for s in listed if s["kind"] == "custom"]
    assert "custom_voice" in custom_names


def test_import_style_via_json(client):
    payload = {**_valid_style_payload(), "name": "json_voice"}
    r = client.post("/v1/styles/import", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "json_voice"


def test_chunked_json_import_accepts_body_exactly_at_limit(client, fake_state):
    state, _ = fake_state
    body = _json_body_at_size(MAX_STYLE_IMPORT_BYTES, "exact_limit")
    response = client.post(
        "/v1/styles/import",
        content=_chunks(body),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["name"] == "exact_limit"
    assert (state.custom_styles_dir / "exact_limit.json").exists()


def test_chunked_json_import_rejects_limit_plus_one_before_json_parsing(client, fake_state):
    state, _ = fake_state
    malformed = b"{" + (b" " * MAX_STYLE_IMPORT_BYTES)
    assert len(malformed) == MAX_STYLE_IMPORT_BYTES + 1

    response = client.post(
        "/v1/styles/import",
        content=_chunks(malformed, size=8192),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert "content-length" not in response.request.headers
    assert response.request.headers.get("transfer-encoding") == "chunked"
    assert response.json()["error"]["code"] == "payload_too_large"
    assert state.custom_styles == {}
    assert not (state.custom_styles_dir / "boundary_voice.json").exists()


def test_understated_content_length_does_not_bypass_actual_byte_limit(client):
    body = _json_body_at_size(MAX_STYLE_IMPORT_BYTES + 1, "understated_length")
    response = client.post(
        "/v1/styles/import",
        content=_chunks(body),
        headers={"content-type": "application/json", "content-length": "1"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_chunked_multipart_import_survives_body_replay(client, fake_state):
    state, _ = fake_state
    boundary = b"multipart-under-limit"
    body = _multipart_body(json.dumps(_valid_style_payload()).encode(), boundary=boundary)
    response = client.post(
        "/v1/styles/import",
        content=_chunks(body, size=7),
        headers={"content-type": f"multipart/form-data; boundary={boundary.decode()}"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["name"] == "voice"
    assert "content-length" not in response.request.headers
    assert (state.custom_styles_dir / "voice.json").exists()


def test_oversized_chunked_multipart_is_rejected_before_parser_or_save(
    client, fake_state, monkeypatch
):
    from starlette.formparsers import MultiPartParser

    state, _ = fake_state
    boundary = b"multipart-over-limit"
    envelope_size = len(_multipart_body(b"", boundary=boundary))
    file_payload = _json_body_at_size(
        MAX_STYLE_IMPORT_BYTES + 1 - envelope_size, "multipart_oversize"
    )
    body = _multipart_body(file_payload, boundary=boundary)
    assert len(body) == MAX_STYLE_IMPORT_BYTES + 1
    parse_calls = 0
    original_parse = MultiPartParser.parse

    async def track_parse(parser):
        nonlocal parse_calls
        parse_calls += 1
        return await original_parse(parser)

    monkeypatch.setattr(MultiPartParser, "parse", track_parse)
    response = client.post(
        "/v1/styles/import",
        content=_chunks(body),
        headers={"content-type": f"multipart/form-data; boundary={boundary.decode()}"},
    )

    assert response.status_code == 413
    assert "content-length" not in response.request.headers
    assert response.json()["error"]["code"] == "payload_too_large"
    assert parse_calls == 0
    assert state.custom_styles == {}
    assert not (state.custom_styles_dir / "multipart_oversize.json").exists()


def test_chunked_disconnect_does_not_import_partial_json(fake_state):
    state, _ = fake_state
    app = create_app(state=state)
    messages = iter(
        [
            {"type": "http.request", "body": b'{"name":"partial"', "more_body": True},
            {"type": "http.disconnect"},
        ]
    )
    sent = []
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/styles/import",
        "raw_path": b"/v1/styles/import",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("testclient", 123),
        "server": ("testserver", 80),
    }

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))

    assert sent == []
    assert state.custom_styles == {}
    assert not (state.custom_styles_dir / "partial.json").exists()


def test_large_non_import_body_is_not_subject_to_style_limit(client):
    body = json.dumps({"text": "hello", "ignored": "x" * MAX_STYLE_IMPORT_BYTES})
    response = client.post("/v1/tts", content=body, headers={"content-type": "application/json"})

    assert response.status_code == 200, response.text


def test_import_rejects_builtin_name(client):
    files = {"file": ("M1.json", json.dumps(_valid_style_payload()), "application/json")}
    r = client.post("/v1/styles/import", files=files, data={"name": "M1"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] in {"style_name_conflict", "invalid_style_name"}


def test_import_rejects_bad_name(client):
    files = {"file": ("../escape.json", json.dumps(_valid_style_payload()), "application/json")}
    r = client.post("/v1/styles/import", files=files, data={"name": "../escape"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_style_name"


def test_import_rejects_bad_payload(client):
    bad = {"style_ttl": {"dims": [1]}}  # missing "data" and style_dp
    files = {"file": ("bad.json", json.dumps(bad), "application/json")}
    r = client.post("/v1/styles/import", files=files, data={"name": "bad"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_style_payload"


def test_oversize_upload_rejected_at_content_length(client):
    """Pre-flight ``Content-Length`` middleware rejects oversized uploads
    before the handler buffers the body."""
    # ~1.6 MiB JSON body, comfortably above the 1 MiB cap.
    big = (
        '{"name":"x","style_ttl":{"dims":[1],"data":['
        + ",".join(["0"] * 800_000)
        + ']},"style_dp":{"dims":[1],"data":[0]}}'
    ).encode("utf-8")
    assert len(big) > 1 * 1024 * 1024
    r = client.post(
        "/v1/styles/import",
        content=big,
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "payload_too_large"


def test_invalid_content_length_header(client):
    """Non-numeric ``Content-Length`` is a 400 (rather than a confusing 500)."""
    r = client.post(
        "/v1/styles/import",
        content=b"{}",
        headers={
            "content-type": "application/json",
            "content-length": "not-a-number",
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_content_length"


# ---------------------------------------------------------------------------
# /v1/tts
# ---------------------------------------------------------------------------


def test_tts_returns_wav_bytes(client, fake_state):
    _, fake = fake_state
    r = client.post("/v1/tts", json={"text": "hello", "voice": "M1", "lang": "en"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/wav"
    assert r.headers.get("X-Sample-Rate") == str(SAMPLE_RATE)
    assert float(r.headers["X-Audio-Duration"]) == pytest.approx(0.1, abs=1e-3)
    # WAV magic
    assert r.content[:4] == b"RIFF"
    # synthesize was called with our params
    assert fake.calls[-1]["text"] == "hello"
    assert fake.calls[-1]["lang"] == "en"
    assert fake.calls[-1]["voice_style_source"] == "builtin:M1"


def test_tts_unknown_voice(client):
    r = client.post("/v1/tts", json={"text": "hi", "voice": "DOES_NOT_EXIST"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unknown_voice"


def test_tts_unsupported_format(client):
    r = client.post(
        "/v1/tts",
        json={"text": "hi", "voice": "M1", "response_format": "mp3"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_response_format"


def test_tts_invalid_lang(client):
    r = client.post("/v1/tts", json={"text": "hi", "voice": "M1", "lang": "zz"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_lang"


def test_tts_flac(client):
    r = client.post(
        "/v1/tts",
        json={"text": "hi", "voice": "M1", "response_format": "flac"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/flac"
    # FLAC magic
    assert r.content[:4] == b"fLaC"


def test_tts_custom_voice_after_import(client, fake_state):
    _, fake = fake_state
    payload = {**_valid_style_payload(), "name": "demo"}
    assert client.post("/v1/styles/import", json=payload).status_code == 200

    r = client.post("/v1/tts", json={"text": "hi", "voice": "demo"})
    assert r.status_code == 200, r.text
    assert fake.calls[-1]["voice_style_source"].startswith("custom:")


# ---------------------------------------------------------------------------
# /v1/audio/speech (OpenAI compat alias)
# ---------------------------------------------------------------------------


def test_openai_compat_default_wav(client, fake_state):
    _, fake = fake_state
    r = client.post(
        "/v1/audio/speech",
        json={
            "model": "supertonic-3",
            "input": "hello from openai client",
            "voice": "M1",
            "response_format": "wav",
        },
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"
    assert fake.calls[-1]["text"] == "hello from openai client"


def test_openai_compat_mp3_rejected(client):
    r = client.post(
        "/v1/audio/speech",
        json={
            "model": "supertonic-3",
            "input": "hi",
            "voice": "M1",
            "response_format": "mp3",
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_response_format"


def test_openai_compat_model_mismatch(client):
    r = client.post(
        "/v1/audio/speech",
        json={"model": "supertonic-2", "input": "hi", "voice": "M1"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "model_not_loaded"


def test_openai_compat_unknown_model(client):
    r = client.post(
        "/v1/audio/speech",
        json={"model": "supertonic-99", "input": "hi", "voice": "M1"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unknown_model"


# ---------------------------------------------------------------------------
# /v1/tts/batch
# ---------------------------------------------------------------------------


def test_batch_two_items(client, fake_state):
    _, fake = fake_state
    payload = {
        "items": [
            {"text": "one", "voice": "M1", "lang": "en"},
            {"text": "둘", "voice": "F1", "lang": "ko"},
        ],
        "defaults": {"speed": 1.1, "steps": 12},
    }
    r = client.post("/v1/tts/batch", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["items"]) == 2
    # each item carries WAV bytes
    raw0 = base64.b64decode(body["items"][0]["audio_base64"])
    assert raw0[:4] == b"RIFF"
    assert body["items"][0]["format"] == "wav"
    assert body["items"][0]["sample_rate"] == SAMPLE_RATE
    # defaults propagated to call site when item didn't override
    assert fake.calls[-2]["speed"] == pytest.approx(1.1)
    assert fake.calls[-1]["speed"] == pytest.approx(1.1)
    assert fake.calls[-2]["total_steps"] == 12
    assert fake.calls[-1]["total_steps"] == 12


def test_batch_item_voice_is_optional_and_nullable_in_pydantic_schema():
    assert BatchItem(text="hello").voice is None
    assert BatchItem.model_validate({"text": "hello", "voice": None}).voice is None

    voice_schema = BatchItem.model_json_schema()["properties"]["voice"]
    assert voice_schema["default"] is None
    assert any(option.get("type") == "null" for option in voice_schema["anyOf"])


def test_batch_item_inherits_default_voice_and_keeps_explicit_m1(client, fake_state):
    _, fake = fake_state
    response = client.post(
        "/v1/tts/batch",
        json={
            "defaults": {"voice": "F1"},
            "items": [
                {"text": "inherit from default"},
                {"text": "explicit override", "voice": "M1"},
                {"text": "null inherits", "voice": None},
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert [call["voice_style_source"] for call in fake.calls[-3:]] == [
        "builtin:F1",
        "builtin:M1",
        "builtin:F1",
    ]


def test_batch_item_without_voice_settings_keeps_m1_default(client, fake_state):
    _, fake = fake_state
    response = client.post("/v1/tts/batch", json={"items": [{"text": "default voice"}]})

    assert response.status_code == 200, response.text
    assert fake.calls[-1]["voice_style_source"] == "builtin:M1"


def test_batch_item_unknown_inherited_voice_returns_structured_error(client):
    response = client.post(
        "/v1/tts/batch",
        json={"defaults": {"voice": "NOPE"}, "items": [{"text": "invalid inherited voice"}]},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unknown_voice"
    assert "items[0]" in response.json()["error"]["message"]


def test_batch_explicit_empty_voice_does_not_fall_back_to_default(client):
    response = client.post(
        "/v1/tts/batch",
        json={"defaults": {"voice": "F1"}, "items": [{"text": "empty", "voice": ""}]},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unknown_voice"


def test_batch_item_unknown_voice(client):
    payload = {
        "items": [
            {"text": "ok", "voice": "M1"},
            {"text": "bad", "voice": "NOPE"},
        ]
    }
    r = client.post("/v1/tts/batch", json=payload)
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "unknown_voice"
    assert "items[1]" in body["error"]["message"]


# ---------------------------------------------------------------------------
# Validation errors from pydantic (422 → our envelope is not used here,
# FastAPI returns its default 422 body, which is acceptable for now).
# ---------------------------------------------------------------------------


def test_tts_empty_text_is_422(client):
    r = client.post("/v1/tts", json={"text": "", "voice": "M1"})
    # pydantic min_length=1 produces a 422 from FastAPI's default validator.
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Default custom-styles directory is scoped per model so the same name
# cannot collide across model versions.
# ---------------------------------------------------------------------------


def test_default_custom_styles_dir_is_per_model(monkeypatch):
    from supertonic.server import styles_store

    # Make sure no user-level override leaks in.
    monkeypatch.delenv("SUPERTONIC_CUSTOM_STYLES_DIR", raising=False)
    monkeypatch.delenv("SUPERTONIC_CACHE_DIR", raising=False)

    d3 = styles_store.default_custom_styles_dir("supertonic-3")
    d2 = styles_store.default_custom_styles_dir("supertonic-2")
    d1 = styles_store.default_custom_styles_dir("supertonic")

    assert d3.name == "custom_styles"
    assert d3.parent.name == "supertonic3"
    assert d2.parent.name == "supertonic2"
    assert d1.parent.name == "supertonic"
    assert d3 != d2 != d1


def test_env_override_wins_over_model_scope(monkeypatch, tmp_path):
    from supertonic.server import styles_store

    monkeypatch.setenv("SUPERTONIC_CUSTOM_STYLES_DIR", str(tmp_path / "shared"))
    assert (
        styles_store.default_custom_styles_dir("supertonic-3")
        == tmp_path / "shared"
        == styles_store.default_custom_styles_dir("supertonic-2")
    )


def test_default_custom_styles_dir_inherits_cache_dir_env_var(monkeypatch, tmp_path):
    """``SUPERTONIC_CACHE_DIR`` propagates through to the server's custom-style
    directory: ``<env>/custom_styles``.

    Verifies that the 1.3.1 fix to :func:`supertonic.config.get_model_cache_dir`
    transitively repairs ``default_custom_styles_dir`` without any change to
    the server package itself.
    """
    from supertonic.server import styles_store

    # SUPERTONIC_CUSTOM_STYLES_DIR is a stronger override and would mask the
    # behavior under test; explicitly clear it.
    monkeypatch.delenv("SUPERTONIC_CUSTOM_STYLES_DIR", raising=False)
    monkeypatch.setenv("SUPERTONIC_CACHE_DIR", str(tmp_path))

    assert styles_store.default_custom_styles_dir("supertonic-3") == tmp_path / "custom_styles"
    assert styles_store.default_custom_styles_dir("supertonic-2") == tmp_path / "custom_styles"
