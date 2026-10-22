"""Regression tests for safe model refreshes without live model downloads."""

import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from supertonic import loader
from supertonic.config import (
    CFG_REL_PATH,
    DP_ONNX_REL_PATH,
    TEXT_ENC_ONNX_REL_PATH,
    UNICODE_INDEXER_REL_PATH,
    VECTOR_EST_ONNX_REL_PATH,
    VOCODER_ONNX_REL_PATH,
    VOICE_STYLES_DIR,
)
from supertonic.loader import (
    has_all_onnx_modules,
    list_available_voice_style_names,
    load_voice_style_from_json_file,
    load_voice_style_from_name,
)

MODEL_FILES = (
    CFG_REL_PATH,
    UNICODE_INDEXER_REL_PATH,
    DP_ONNX_REL_PATH,
    TEXT_ENC_ONNX_REL_PATH,
    VECTOR_EST_ONNX_REL_PATH,
    VOCODER_ONNX_REL_PATH,
)


def _style_payload(version: str) -> dict:
    return {
        "style_ttl": {"dims": [1, 1], "data": [1.0], "version": version},
        "style_dp": {"dims": [1, 1], "data": [2.0], "version": version},
    }


def _write_model_tree(directory: Path, version: str) -> None:
    """Write a tiny local model-shaped tree; no ONNX weights are needed."""
    for relative_path in MODEL_FILES:
        path = directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative_path == CFG_REL_PATH:
            path.write_text(json.dumps({"version": version}), encoding="utf-8")
        elif relative_path == UNICODE_INDEXER_REL_PATH:
            path.write_text("[0]", encoding="utf-8")
        else:
            path.write_bytes(f"{version}:{relative_path.name}".encode())

    styles = directory / VOICE_STYLES_DIR
    styles.mkdir(parents=True, exist_ok=True)
    for name in ("M1", "F1"):
        (styles / f"{name}.json").write_text(json.dumps(_style_payload(version)), encoding="utf-8")


def _seed_custom_style(directory: Path, name: str = "ImportedVoice") -> bytes:
    directory.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(_style_payload("user"), separators=(",", ":")).encode("utf-8")
    (directory / f"{name}.json").write_bytes(contents)
    return contents


def _fake_hub(monkeypatch, writer):
    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = writer
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


def test_download_preserves_custom_styles_and_unmanaged_files(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    _write_model_tree(model_dir, "old")
    custom_dir = model_dir / "custom_styles"
    custom_bytes = _seed_custom_style(custom_dir)
    unmanaged = model_dir / "user-notes.txt"
    unmanaged.write_text("keep this file", encoding="utf-8")

    external_custom_dir = tmp_path / "external-voices"
    external_bytes = _seed_custom_style(external_custom_dir, "ExternalVoice")

    def fake_snapshot_download(*, local_dir, **_kwargs):
        snapshot = Path(local_dir)
        _write_model_tree(snapshot, "new")
        # A downloaded snapshot does not own the local user-style directory.
        _seed_custom_style(snapshot / "custom_styles", "SnapshotVoice")

    _fake_hub(monkeypatch, fake_snapshot_download)

    loader.download_model(model_dir)

    assert has_all_onnx_modules(model_dir)
    assert json.loads((model_dir / CFG_REL_PATH).read_text(encoding="utf-8"))["version"] == "new"
    assert set(list_available_voice_style_names(model_dir)) == {"M1", "F1"}
    assert load_voice_style_from_name(model_dir, "M1").ttl.shape == (1, 1)
    assert (custom_dir / "ImportedVoice.json").read_bytes() == custom_bytes
    assert load_voice_style_from_json_file(custom_dir / "ImportedVoice.json") is not None
    assert not (custom_dir / "SnapshotVoice.json").exists()
    assert unmanaged.read_text(encoding="utf-8") == "keep this file"
    assert (external_custom_dir / "ExternalVoice.json").read_bytes() == external_bytes
    assert load_voice_style_from_json_file(external_custom_dir / "ExternalVoice.json") is not None


def test_repeated_download_updates_model_and_cleans_its_staging_dirs(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    versions = iter(("first", "second"))

    def fake_snapshot_download(*, local_dir, **_kwargs):
        _write_model_tree(Path(local_dir), next(versions))

    _fake_hub(monkeypatch, fake_snapshot_download)

    loader.download_model(model_dir)
    loader.download_model(model_dir)

    assert has_all_onnx_modules(model_dir)
    assert json.loads((model_dir / CFG_REL_PATH).read_text(encoding="utf-8"))["version"] == "second"
    assert not list(tmp_path.glob(".model.download-*"))
    assert not list(tmp_path.glob(".model.backup-*"))


def test_partial_download_failure_keeps_previous_model_and_cleans_stage(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    _write_model_tree(model_dir, "old")
    custom_bytes = _seed_custom_style(model_dir / "custom_styles")
    unmanaged = model_dir / "user-data.bin"
    unmanaged.write_bytes(b"user-owned")

    def fake_snapshot_download(*, local_dir, **_kwargs):
        partial = Path(local_dir) / "onnx" / "partial.onnx"
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(b"partial download")
        raise OSError("simulated download interruption")

    _fake_hub(monkeypatch, fake_snapshot_download)

    with pytest.raises(RuntimeError, match="simulated download interruption"):
        loader.download_model(model_dir)

    assert has_all_onnx_modules(model_dir)
    assert json.loads((model_dir / CFG_REL_PATH).read_text(encoding="utf-8"))["version"] == "old"
    assert (model_dir / "custom_styles" / "ImportedVoice.json").read_bytes() == custom_bytes
    assert unmanaged.read_bytes() == b"user-owned"
    assert not list(tmp_path.glob(".model.download-*"))


def test_incomplete_snapshot_is_rejected_before_replacing_model(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    _write_model_tree(model_dir, "old")
    custom_bytes = _seed_custom_style(model_dir / "custom_styles")

    def fake_snapshot_download(*, local_dir, **_kwargs):
        snapshot = Path(local_dir)
        _write_model_tree(snapshot, "partial")
        (snapshot / VOCODER_ONNX_REL_PATH).unlink()

    _fake_hub(monkeypatch, fake_snapshot_download)

    with pytest.raises(RuntimeError, match="missing a required file"):
        loader.download_model(model_dir)

    assert has_all_onnx_modules(model_dir)
    assert json.loads((model_dir / CFG_REL_PATH).read_text(encoding="utf-8"))["version"] == "old"
    assert (model_dir / "custom_styles" / "ImportedVoice.json").read_bytes() == custom_bytes
    assert not list(tmp_path.glob(".model.download-*"))
    assert not list(tmp_path.glob(".model.backup-*"))


def test_auto_download_repair_preserves_custom_voice(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    _write_model_tree(model_dir, "old")
    custom_bytes = _seed_custom_style(model_dir / "custom_styles")
    (model_dir / VOCODER_ONNX_REL_PATH).unlink()

    def fake_snapshot_download(*, local_dir, **_kwargs):
        _write_model_tree(Path(local_dir), "repaired")

    _fake_hub(monkeypatch, fake_snapshot_download)
    monkeypatch.setattr(loader, "load_onnx_modules", lambda *_args, **_kwargs: (1, 2, 3, 4))
    monkeypatch.setattr(loader, "load_text_processor", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(loader, "Supertonic", lambda *_args, **_kwargs: "loaded-engine")

    result = loader.load_model(model_dir, auto_download=True)

    assert result == "loaded-engine"
    assert has_all_onnx_modules(model_dir)
    repaired_config = json.loads((model_dir / CFG_REL_PATH).read_text(encoding="utf-8"))
    assert repaired_config["version"] == "repaired"
    assert (model_dir / "custom_styles" / "ImportedVoice.json").read_bytes() == custom_bytes


def test_failed_publish_restores_previous_model_and_custom_voice(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    _write_model_tree(model_dir, "old")
    custom_bytes = _seed_custom_style(model_dir / "custom_styles")
    unmanaged = model_dir / "user-data.txt"
    unmanaged.write_text("untouched", encoding="utf-8")
    staged_roots = []

    def fake_snapshot_download(*, local_dir, **_kwargs):
        staged_roots.append(Path(local_dir))
        _write_model_tree(Path(local_dir), "new")

    _fake_hub(monkeypatch, fake_snapshot_download)
    replace = os.replace
    failed = False

    def fail_when_publishing_styles(source, destination):
        nonlocal failed
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            not failed
            and staged_roots
            and source_path == staged_roots[0] / VOICE_STYLES_DIR
            and destination_path == model_dir / VOICE_STYLES_DIR
        ):
            failed = True
            raise OSError("simulated final publish failure")
        return replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_when_publishing_styles)

    with pytest.raises(RuntimeError, match="simulated final publish failure"):
        loader.download_model(model_dir)

    assert failed
    assert has_all_onnx_modules(model_dir)
    assert json.loads((model_dir / CFG_REL_PATH).read_text(encoding="utf-8"))["version"] == "old"
    assert set(list_available_voice_style_names(model_dir)) == {"M1", "F1"}
    assert (model_dir / "custom_styles" / "ImportedVoice.json").read_bytes() == custom_bytes
    assert unmanaged.read_text(encoding="utf-8") == "untouched"


def test_concurrent_download_returns_clear_conflict(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    _write_model_tree(model_dir, "old")

    def fake_snapshot_download(*, local_dir, **_kwargs):
        with pytest.raises(RuntimeError, match="already in progress"):
            loader.download_model(model_dir)
        _write_model_tree(Path(local_dir), "new")

    _fake_hub(monkeypatch, fake_snapshot_download)

    loader.download_model(model_dir)

    assert json.loads((model_dir / CFG_REL_PATH).read_text(encoding="utf-8"))["version"] == "new"


def test_retry_recovers_an_interrupted_publish_and_leaves_old_dirs_untouched(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    _write_model_tree(model_dir, "old")
    custom_bytes = _seed_custom_style(model_dir / "custom_styles")

    # These look like abandoned work from an earlier process. A new operation
    # must use its own unique stage/backup and leave unrelated old work alone.
    stale_stage = tmp_path / ".model.download-abandoned"
    stale_backup = tmp_path / ".model.backup-abandoned"
    stale_stage.mkdir()
    stale_backup.mkdir()
    (stale_stage / "sentinel").write_text("stage", encoding="utf-8")
    (stale_backup / "sentinel").write_text("backup", encoding="utf-8")

    staged_roots = []

    def fake_snapshot_download(*, local_dir, **_kwargs):
        staged_roots.append(Path(local_dir))
        _write_model_tree(Path(local_dir), "new")

    _fake_hub(monkeypatch, fake_snapshot_download)
    replace = os.replace

    def interrupt_after_first_managed_dir(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            staged_roots
            and source_path == staged_roots[0] / "onnx"
            and destination_path == model_dir / "onnx"
        ):
            raise KeyboardInterrupt("simulated process interruption")
        return replace(source, destination)

    monkeypatch.setattr(os, "replace", interrupt_after_first_managed_dir)
    with pytest.raises(KeyboardInterrupt, match="simulated process interruption"):
        loader.download_model(model_dir)

    monkeypatch.setattr(os, "replace", replace)
    loader.download_model(model_dir)

    assert has_all_onnx_modules(model_dir)
    assert json.loads((model_dir / CFG_REL_PATH).read_text(encoding="utf-8"))["version"] == "new"
    assert (model_dir / "custom_styles" / "ImportedVoice.json").read_bytes() == custom_bytes
    assert (stale_stage / "sentinel").read_text(encoding="utf-8") == "stage"
    assert (stale_backup / "sentinel").read_text(encoding="utf-8") == "backup"
    assert list(tmp_path.glob(".model.download-*")) == [stale_stage]
    assert list(tmp_path.glob(".model.backup-*")) == [stale_backup]
