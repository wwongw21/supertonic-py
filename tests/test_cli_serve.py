"""Tests for the ``supertonic serve`` CLI surface.

These tests verify the *parser* and the *error UX* when fastapi/uvicorn are
missing. They never start a real server (uvicorn.run is monkeypatched).
"""

from __future__ import annotations

from argparse import Namespace

import pytest

from supertonic.cli import cmd_serve, create_parser


def test_parser_defaults():
    parser = create_parser()
    args = parser.parse_args(["serve"])
    assert args.command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 7788
    assert args.model == "supertonic-3"
    assert args.cors is None
    assert args.log_level == "info"


def test_parser_overrides():
    parser = create_parser()
    args = parser.parse_args(
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--model",
            "supertonic-2",
            "--cors",
            "http://localhost:*,chrome-extension://*",
            "--log-level",
            "debug",
            "--verbose",
        ]
    )
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.model == "supertonic-2"
    assert args.cors == "http://localhost:*,chrome-extension://*"
    assert args.log_level == "debug"
    assert args.verbose is True


def test_parser_invalid_model_rejected():
    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--model", "supertonic-99"])


def test_cmd_serve_missing_fastapi_friendly_error(monkeypatch, capsys):
    """If the [serve] extra isn't installed, give a clean install hint."""
    # Force the inline ``import uvicorn`` inside cmd_serve to fail, regardless
    # of whether fastapi is actually installed in the test environment.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__  # type: ignore[index]

    def fail_uvicorn(name, *a, **kw):
        if name == "uvicorn" or name.startswith("supertonic.server"):
            raise ImportError("simulated missing dep")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", fail_uvicorn)

    args = Namespace(
        verbose=False,
        host="127.0.0.1",
        port=7788,
        model="supertonic-3",
        cors=None,
        log_level="info",
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_serve(args)
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "pip install supertonic[serve]" in out


def test_cmd_serve_non_loopback_warns(monkeypatch, capsys):
    """Binding to anything other than loopback emits a stderr warning."""
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    pytest.importorskip("uvicorn")

    called = {}

    def fake_uvicorn_run(app, **kwargs):
        called["host"] = kwargs.get("host")
        called["port"] = kwargs.get("port")
        # Don't actually start a server.

    def fake_create_app(**kwargs):
        called["model"] = kwargs.get("model")
        return object()

    monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)
    monkeypatch.setattr("supertonic.server.create_app", fake_create_app)

    args = Namespace(
        verbose=False,
        host="0.0.0.0",
        port=7788,
        model="supertonic-3",
        cors=None,
        log_level="info",
    )
    cmd_serve(args)
    err = capsys.readouterr().err
    assert "Warning" in err and "0.0.0.0" in err
    assert called["host"] == "0.0.0.0"
    assert called["port"] == 7788
    assert called["model"] == "supertonic-3"
