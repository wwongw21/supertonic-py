#!/usr/bin/env python3
"""Reproduce the reviewed pre-fix paths without optional packages or model files.

Run from any working directory with Python 3.9 or later. The script locates
the repository from its own path, checks LF-normalized source hashes against
the recorded source pin, and executes the relevant source AST with small
standard-library mocks. It intentionally demonstrates the old behavior.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import logging
import platform
import shutil
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SOURCE_PIN = "908a56486e821e833a80530ff0cae3ad0b046fce"
SOURCE_HASHES = {
    "supertonic/loader.py": "9478b085ccae88fd44849a87d8a55d0ddada03e4fbb7f47f1707f04ac06e621f",
    "supertonic/server/schemas.py": "c678f17799777f08193212f66df8394e123789b01c1a5b22eb7898b46984c3db",
    "supertonic/server/routes.py": "6dceb93c9a7fb6766fded4b3ec4f0ad09c5ae655aab4a0f3d4c8755accc65720",
    "supertonic/server/app.py": "11e9c7b9961922167d585c4e4f1ce3275f796f33d41ca2ebdc3da72094a324e0",
    "supertonic/utils.py": "8bf2e0a14783de9266edba1135597e6b77b143907e445b08dcc45f694ff51ed3",
}


def read_source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def check_source_hashes() -> None:
    print(f"repository=supertonic-py; source_pin={SOURCE_PIN}")
    print("hash_method=SHA-256 of UTF-8 source bytes after CRLF-to-LF normalization")
    for relative_path, expected in SOURCE_HASHES.items():
        raw = (ROOT / relative_path).read_bytes().replace(b"\r\n", b"\n")
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"source mismatch for {relative_path}: expected {expected}, got {actual}"
            )
        print(f"source_sha256={actual}  path={relative_path}")


def parse_source(relative_path: str) -> ast.Module:
    return ast.parse(read_source(relative_path), filename=relative_path)


def find_node(tree: ast.AST, node_type: type, name: str, parent_name: str | None = None):
    if parent_name is None:
        for node in ast.walk(tree):
            if isinstance(node, node_type) and getattr(node, "name", None) == name:
                return node
    else:
        parent = find_node(tree, ast.FunctionDef, parent_name)
        for node in ast.walk(parent):
            if isinstance(node, node_type) and getattr(node, "name", None) == name:
                return node
    raise LookupError(f"could not find {name} in source AST")


def cleaned_function(node):
    node = copy.deepcopy(node)
    node.decorator_list = []
    node.returns = None
    args = node.args
    for arg in args.posonlyargs + args.args + args.kwonlyargs:
        arg.annotation = None
    if args.vararg:
        args.vararg.annotation = None
    if args.kwarg:
        args.kwarg.annotation = None
    args.defaults = [ast.Constant(value=None) for _ in args.defaults]
    args.kw_defaults = [None if value is None else ast.Constant(value=None) for value in args.kw_defaults]
    return ast.fix_missing_locations(node)


def compile_function(node, namespace: dict):
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            cleaned_function(node),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, "<pinned-source-function>", "exec"), namespace)
    return namespace[node.name]


def compile_class(node, namespace: dict):
    node = copy.deepcopy(node)
    node.decorator_list = []
    cleaned_body = []
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cleaned_body.append(cleaned_function(child))
        else:
            cleaned_body.append(child)
    node.body = cleaned_body
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, "<pinned-source-class>", "exec"), namespace)
    return namespace[node.name]


def probe_downloader() -> None:
    tree = parse_source("supertonic/loader.py")
    function = find_node(tree, ast.FunctionDef, "download_model")
    calls = []

    def snapshot_download(repo_id, local_dir, revision):
        calls.append((repo_id, revision))
        destination = Path(local_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "model.bin").write_bytes(b"mock model snapshot")

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    previous_hub = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = hub
    namespace = {
        "Path": Path,
        "shutil": shutil,
        "logger": logging.getLogger("repro.supertonic.loader"),
        "DEFAULT_MODEL": "supertonic-3",
        "DEFAULT_MODEL_REPO": "Supertone/supertonic-3",
        "get_model_repo": lambda name: f"Supertone/{name}",
        "get_model_revision": lambda name: "mock-revision",
    }
    download_model = compile_function(function, namespace)
    try:
        with tempfile.TemporaryDirectory(prefix="supertonic-py-spy1-") as temporary:
            model_dir = Path(temporary) / "model"
            voice = model_dir / "custom_styles" / "my_voice.json"
            voice.parent.mkdir(parents=True)
            voice.write_bytes(b'{"voice":"user data"}')
            download_model(model_dir)
            voice_removed = not voice.exists()
            model_published = (model_dir / "model.bin").is_file()
    finally:
        if previous_hub is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = previous_hub

    assert len(calls) == 1 and voice_removed and model_published
    print(
        "SPY-1 downloader: existing custom voice removed by replacement="
        f"{voice_removed}; mocked snapshot published={model_published}; calls={len(calls)}"
    )


def probe_batch_voice_default() -> None:
    schema_tree = parse_source("supertonic/server/schemas.py")
    batch_item = find_node(schema_tree, ast.ClassDef, "BatchItem")
    voice_default = None
    for statement in batch_item.body:
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "voice"
        ):
            voice_default = ast.literal_eval(statement.value.args[0])
            break
    if voice_default is None:
        raise LookupError("BatchItem.voice default not found")

    route_tree = parse_source("supertonic/server/routes.py")
    synth_batch = find_node(route_tree, ast.FunctionDef, "synth_batch")
    assignment = next(
        node
        for node in ast.walk(synth_batch)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "voice" for target in node.targets)
    )
    expression = ast.Expression(body=copy.deepcopy(assignment.value))
    ast.fix_missing_locations(expression)
    item = types.SimpleNamespace(voice=voice_default)
    defaults = types.SimpleNamespace(voice="F1")
    resolved = eval(compile(expression, "<pinned-batch-resolver>", "eval"), {}, {"item": item, "defaults": defaults})

    assert voice_default == "M1" and resolved == "M1"
    print(
        "SPY-2 batch voice: defaults.voice=F1; omitted item voice receives schema "
        f"default={voice_default}; actual route resolver selects={resolved}"
    )


def probe_unbounded_json_import() -> None:
    app_tree = parse_source("supertonic/server/app.py")
    middleware_node = find_node(app_tree, ast.ClassDef, "StyleImportSizeLimit")
    limited_paths = None
    for node in ast.walk(app_tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_SIZE_LIMITED_PATHS"
            for target in node.targets
        ):
            limited_paths = ast.literal_eval(node.value)
            break
    if limited_paths is None:
        raise LookupError("_SIZE_LIMITED_PATHS not found")
    route_tree = parse_source("supertonic/server/routes.py")
    handler_node = find_node(route_tree, ast.AsyncFunctionDef, "import_style", "register_routes")

    constant_tree = route_tree
    limit_value = None
    for node in ast.walk(constant_tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MAX_STYLE_IMPORT_BYTES"
            for target in node.targets
        ):
            limit_value = eval(compile(ast.Expression(node.value), "<pinned-size-limit>", "eval"), {})
            break
    if limit_value is None:
        raise LookupError("MAX_STYLE_IMPORT_BYTES not found")

    class StyleStore:
        class InvalidStyleName(Exception):
            pass

        class StyleNameConflict(Exception):
            pass

        def __init__(self, root: Path):
            self.root = root
            self.called = False

        def save(self, directory, name, data, builtin_names, overwrite):
            self.called = True
            path = Path(directory) / f"{name}.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return path

    class FakeTTS:
        voice_style_names = ["M1", "F1"]

    class FakeResponse:
        def __init__(self, status_code=200, **values):
            self.status_code = status_code
            self.values = values

    with tempfile.TemporaryDirectory(prefix="supertonic-py-spy3-") as temporary:
        temporary_path = Path(temporary)
        store = StyleStore(temporary_path)
        state = types.SimpleNamespace(
            tts=FakeTTS(),
            custom_styles_dir=temporary_path,
            custom_styles={},
        )
        handler_namespace = {
            "Path": Path,
            "json": json,
            "MAX_STYLE_IMPORT_BYTES": limit_value,
            "styles_store": store,
            "_state": lambda request: state,
            "_error": lambda status, message, code, **kwargs: FakeResponse(status),
            "StyleImportResponse": lambda **values: FakeResponse(200, **values),
        }
        import_style = compile_function(handler_node, handler_namespace)

        class MockRequest:
            def __init__(self, body):
                self._body = body
                self.headers = {"content-type": "application/json"}

            async def json(self):
                return json.loads(self._body)

        class DownstreamApp:
            def __init__(self):
                self.body_size = 0
                self.response_status = None

            async def __call__(self, scope, receive, send):
                body_parts = []
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        break
                    body_parts.append(message.get("body", b""))
                    if not message.get("more_body", False):
                        break
                body = b"".join(body_parts)
                self.body_size = len(body)
                result = await import_style(MockRequest(body), False, None, None)
                self.response_status = result.status_code
                await send({"type": "http.response.start", "status": result.status_code, "headers": []})
                await send({"type": "http.response.body", "body": b""})

        class SendCollector:
            def __init__(self):
                self.messages = []

            async def __call__(self, message):
                self.messages.append(message)

        payload = {
            "name": "probe_voice",
            "style_ttl": {"dims": [1, 4], "data": [0.0, 0.0, 0.0, 0.0]},
            "style_dp": {"dims": [1, 4], "data": [0.0, 0.0, 0.0, 0.0]},
            "ignored": "",
        }
        target_size = limit_value + 128
        empty_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        payload["ignored"] = "x" * (target_size - len(empty_body))
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        assert len(body) == target_size

        chunks = [body[start : start + 65536] for start in range(0, len(body), 65536)]
        chunk_index = 0

        async def receive():
            nonlocal chunk_index
            if chunk_index >= len(chunks):
                return {"type": "http.request", "body": b"", "more_body": False}
            part = chunks[chunk_index]
            chunk_index += 1
            return {
                "type": "http.request",
                "body": part,
                "more_body": chunk_index < len(chunks),
            }

        downstream = DownstreamApp()
        middleware = compile_class(
            middleware_node,
            {
                "_SIZE_LIMITED_PATHS": limited_paths,
                "ErrorEnvelope": object,
                "ErrorDetail": object,
            },
        )(downstream, max_bytes=limit_value)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/styles/import",
            "headers": [
                (b"content-type", b"application/json"),
                (b"transfer-encoding", b"chunked"),
            ],
        }
        sender = SendCollector()

        import asyncio

        asyncio.run(middleware(scope, receive, sender))
        response_start = next(message for message in sender.messages if message["type"] == "http.response.start")
        status = response_start["status"]
        assert downstream.body_size == target_size and status == 200 and store.called

    print(
        "SPY-3 chunked JSON import: bytes={}; Content-Length=absent; middleware="
        "forwarded body; mocked handler status={}; mocked style store called={}".format(
            target_size, status, store.called
        )
    )


def probe_oversized_chunk() -> None:
    tree = parse_source("supertonic/utils.py")
    function = find_node(tree, ast.FunctionDef, "chunk_text")
    abbreviation_pattern = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_COMMON_ABBREVIATIONS_PATTERN"
            for target in node.targets
        ):
            abbreviation_pattern = ast.literal_eval(node.value)
            break
    if abbreviation_pattern is None:
        raise LookupError("_COMMON_ABBREVIATIONS_PATTERN not found")
    chunk_text = compile_function(
        function,
        {"re": __import__("re"), "_COMMON_ABBREVIATIONS_PATTERN": abbreviation_pattern},
    )
    lengths = [len(chunk) for chunk in chunk_text("word " * 400, max_len=300)]
    assert lengths == [1999]
    print(f"SPY-4 chunk length: configured max=300; actual chunk lengths={lengths}")


def main() -> None:
    print("Portable pre-fix reproduction; source functions are AST-extracted from pinned files.")
    print("invocation=python docs/reviews/2026-10-22/diagnostics/repro_pinned_sources.py")
    print(f"run_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"python={platform.python_version()}; platform={platform.system()} {platform.release()}")
    check_source_hashes()
    probe_downloader()
    probe_batch_voice_default()
    probe_unbounded_json_import()
    probe_oversized_chunk()
    print("result=all four historical source-level bug behaviors reproduced; no runtime fix applied")


if __name__ == "__main__":
    main()
