"""Regression tests for safe model refreshes without live model downloads."""

import errno
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
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


def _deny_lock_file_write(
    monkeypatch, lock_path: Path, *, parent_writable: bool, open_error=None
) -> None:
    original_open = Path.open
    original_access = os.access

    def deny_append(path, mode="r", *args, **kwargs):
        if Path(path) == lock_path and mode == "a+b":
            if open_error is not None:
                raise open_error
            raise PermissionError("simulated read-only model directory")
        return original_open(path, mode, *args, **kwargs)

    def check_access(path, mode):
        if Path(path) == lock_path.parent and mode & os.W_OK:
            return parent_writable
        return original_access(path, mode)

    monkeypatch.setattr(Path, "open", deny_append)
    monkeypatch.setattr(loader.os, "access", check_access)


def _stub_model_loading(monkeypatch) -> None:
    monkeypatch.setattr(loader, "load_configs", lambda *_args: {})
    monkeypatch.setattr(loader, "load_onnx_modules", lambda *_args, **_kwargs: (1, 2, 3, 4))
    monkeypatch.setattr(loader, "load_text_processor", lambda *_args: object())
    monkeypatch.setattr(loader, "Supertonic", lambda *_args: "loaded-engine")


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


def test_older_huggingface_hub_uses_regular_files(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    symlink_options = []

    def old_snapshot_download(*, repo_id, local_dir, revision, local_dir_use_symlinks="auto"):
        symlink_options.append(local_dir_use_symlinks)
        _write_model_tree(Path(local_dir), "old-hub")

    _fake_hub(monkeypatch, old_snapshot_download)

    loader.download_model(model_dir)

    assert symlink_options == [False]
    assert all(
        not path.is_symlink()
        for managed_dir in (model_dir / "onnx", model_dir / VOICE_STYLES_DIR)
        for path in managed_dir.rglob("*")
        if path.is_file()
    )


def test_newer_huggingface_signature_does_not_receive_removed_symlink_option(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"

    def current_snapshot_download(*, repo_id, local_dir, revision):
        _write_model_tree(Path(local_dir), "new-hub")

    _fake_hub(monkeypatch, current_snapshot_download)

    loader.download_model(model_dir)

    assert (
        json.loads((model_dir / CFG_REL_PATH).read_text(encoding="utf-8"))["version"] == "new-hub"
    )


def test_download_materializes_hub_symlinks_without_changing_cache_target(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    cache_target = tmp_path / "hub-cache" / "large-model.onnx"
    cache_target.parent.mkdir()
    cache_target.write_bytes(b"cached ONNX content")

    def fake_snapshot_download(*, local_dir, **_kwargs):
        snapshot = Path(local_dir)
        _write_model_tree(snapshot, "linked")
        staged_weight = snapshot / DP_ONNX_REL_PATH
        staged_weight.unlink()
        try:
            os.symlink(cache_target, staged_weight)
        except OSError as error:
            pytest.skip(f"file symlinks are unavailable on this Windows host: {error}")

    _fake_hub(monkeypatch, fake_snapshot_download)

    loader.download_model(model_dir)

    published_weight = model_dir / DP_ONNX_REL_PATH
    assert published_weight.is_file()
    assert not published_weight.is_symlink()
    assert published_weight.read_bytes() == b"cached ONNX content"
    assert cache_target.read_bytes() == b"cached ONNX content"


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


def test_load_model_allows_complete_read_only_model_without_lock_file(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    _write_model_tree(model_dir, "installed")
    lock_path = tmp_path / ".model.download.lock"
    _deny_lock_file_write(monkeypatch, lock_path, parent_writable=False)
    _stub_model_loading(monkeypatch)

    assert loader.load_model(model_dir, auto_download=False) == "loaded-engine"
    assert not lock_path.exists()


def test_load_model_allows_read_only_model_when_lock_open_returns_erofs(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    _write_model_tree(model_dir, "installed")
    lock_path = tmp_path / ".model.download.lock"
    _deny_lock_file_write(
        monkeypatch,
        lock_path,
        parent_writable=True,
        open_error=OSError(errno.EROFS, "read-only file system"),
    )
    _stub_model_loading(monkeypatch)

    assert loader.load_model(model_dir, auto_download=False) == "loaded-engine"
    assert not lock_path.exists()


def test_load_model_does_not_skip_lock_when_writable_parent_denies_creation(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    _write_model_tree(model_dir, "installed")
    lock_path = tmp_path / ".model.download.lock"
    _deny_lock_file_write(monkeypatch, lock_path, parent_writable=True)

    with pytest.raises(PermissionError, match="simulated read-only model directory"):
        loader.load_model(model_dir, auto_download=False)


def test_load_model_does_not_skip_pending_journal_when_lock_is_unavailable(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    _write_model_tree(model_dir, "installed")
    marker = loader._transaction_path(model_dir)
    marker.write_text("{}", encoding="utf-8")
    lock_path = tmp_path / ".model.download.lock"
    _deny_lock_file_write(monkeypatch, lock_path, parent_writable=False)

    with pytest.raises(PermissionError, match="simulated read-only model directory"):
        loader.load_model(model_dir, auto_download=False)

    assert marker.read_text(encoding="utf-8") == "{}"


def test_read_only_fallback_does_not_bypass_pending_journal_on_erofs(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    _write_model_tree(model_dir, "installed")
    marker = loader._transaction_path(model_dir)
    marker.write_text("{}", encoding="utf-8")
    lock_path = tmp_path / ".model.download.lock"
    _deny_lock_file_write(
        monkeypatch,
        lock_path,
        parent_writable=True,
        open_error=OSError(errno.EROFS, "read-only file system"),
    )
    monkeypatch.setattr(
        loader,
        "has_all_onnx_modules",
        lambda *_args: pytest.fail("completeness checked before pending journal handling"),
    )

    with pytest.raises(OSError) as error:
        loader.load_model(model_dir, auto_download=False)

    assert error.value.errno == errno.EROFS
    assert marker.read_text(encoding="utf-8") == "{}"


def test_unrelated_lock_open_error_is_not_treated_as_read_only(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    _write_model_tree(model_dir, "installed")
    lock_path = tmp_path / ".model.download.lock"
    _deny_lock_file_write(
        monkeypatch,
        lock_path,
        parent_writable=False,
        open_error=OSError(errno.EIO, "simulated disk I/O error"),
    )

    with pytest.raises(OSError) as error:
        loader.load_model(model_dir, auto_download=False)

    assert error.value.errno == errno.EIO


def test_read_only_lock_fallback_waits_for_active_refresh(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    _write_model_tree(model_dir, "installed")
    lock_path = tmp_path / ".model.download.lock"
    lock_path.write_bytes(b"\0")
    _stub_model_loading(monkeypatch)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with loader._model_download_lock(model_dir):
            _deny_lock_file_write(monkeypatch, lock_path, parent_writable=False)
            loading = pool.submit(loader.load_model, model_dir, False)
            with pytest.raises(FutureTimeoutError):
                loading.result(timeout=0.1)
        assert loading.result(timeout=2) == "loaded-engine"


def test_concurrent_loaders_wait_for_recovery_lock_and_both_succeed(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    _write_model_tree(model_dir, "installed")
    _stub_model_loading(monkeypatch)

    first_inside_check = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    check_lock = threading.Lock()
    check_count = 0

    def delayed_completeness_check(_model_dir):
        nonlocal check_count
        with check_lock:
            check_count += 1
            call_number = check_count
        if call_number == 1:
            first_inside_check.set()
            assert release_first.wait(timeout=3)
        return True

    monkeypatch.setattr(loader, "has_all_onnx_modules", delayed_completeness_check)

    def second_load():
        second_started.set()
        return loader.load_model(model_dir, auto_download=False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(loader.load_model, model_dir, False)
        assert first_inside_check.wait(timeout=2)
        second = pool.submit(second_load)
        assert second_started.wait(timeout=2)

        try:
            with pytest.raises(FutureTimeoutError):
                second.result(timeout=0.1)
        finally:
            release_first.set()

        assert first.result(timeout=2) == "loaded-engine"
        assert second.result(timeout=2) == "loaded-engine"

    assert check_count == 2


def test_waiting_for_model_lock_has_a_bounded_timeout(tmp_path):
    model_dir = tmp_path / "model"
    outer_lock = loader._model_download_lock(model_dir)
    expected_timeout = pytest.raises(TimeoutError, match="Timed out waiting")
    inner_lock = loader._model_download_lock(model_dir, wait=True, wait_timeout=0.02)
    with outer_lock, expected_timeout, inner_lock:
        pytest.fail("the lock should remain held by the outer context")


def test_load_model_recovers_after_onnx_publish_before_completeness_check(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    _write_model_tree(model_dir, "old")
    custom_bytes = _seed_custom_style(model_dir / "custom_styles")
    staged_roots = []
    download_calls = 0

    def fake_snapshot_download(*, local_dir, **_kwargs):
        nonlocal download_calls
        download_calls += 1
        staged_roots.append(Path(local_dir))
        _write_model_tree(Path(local_dir), "new")

    _fake_hub(monkeypatch, fake_snapshot_download)
    replace = os.replace

    def interrupt_before_voice_styles_publish(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            staged_roots
            and source_path == staged_roots[0] / VOICE_STYLES_DIR
            and destination_path == model_dir / VOICE_STYLES_DIR
        ):
            raise KeyboardInterrupt("simulated interruption after ONNX publish")
        return replace(source, destination)

    monkeypatch.setattr(os, "replace", interrupt_before_voice_styles_publish)
    with pytest.raises(KeyboardInterrupt, match="after ONNX publish"):
        loader.download_model(model_dir)
    monkeypatch.setattr(os, "replace", replace)

    # This ONNX-only predicate was already true before recovery used to run.
    assert has_all_onnx_modules(model_dir)
    assert not (model_dir / VOICE_STYLES_DIR / "M1.json").exists()

    loaded_versions = []
    original_load_configs = loader.load_configs

    def record_loaded_config(path):
        configs = original_load_configs(path)
        loaded_versions.append(configs["version"])
        return configs

    def fake_load_onnx_modules(path, *_args, **_kwargs):
        assert (Path(path) / VOICE_STYLES_DIR / "M1.json").is_file()
        return (1, 2, 3, 4)

    monkeypatch.setattr(loader, "load_configs", record_loaded_config)
    monkeypatch.setattr(loader, "load_onnx_modules", fake_load_onnx_modules)
    monkeypatch.setattr(loader, "load_text_processor", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        loader, "Supertonic", lambda configs, *_args, **_kwargs: f"{configs['version']}-loaded"
    )
    _deny_lock_file_write(monkeypatch, tmp_path / ".model.download.lock", parent_writable=True)

    result = loader.load_model(model_dir, auto_download=True)

    assert result == "old-loaded"
    assert loaded_versions == ["old"]
    assert download_calls == 1
    assert (model_dir / "custom_styles" / "ImportedVoice.json").read_bytes() == custom_bytes
    assert not (tmp_path / ".model.download-transaction.json").exists()


def test_retry_recovers_an_interruption_after_onnx_publish_and_leaves_stale_dirs_untouched(
    monkeypatch, tmp_path
):
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

    def interrupt_before_voice_styles_publish(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            staged_roots
            and source_path == staged_roots[0] / VOICE_STYLES_DIR
            and destination_path == model_dir / VOICE_STYLES_DIR
        ):
            raise KeyboardInterrupt("simulated interruption after ONNX publish")
        return replace(source, destination)

    monkeypatch.setattr(os, "replace", interrupt_before_voice_styles_publish)
    with pytest.raises(KeyboardInterrupt, match="after ONNX publish"):
        loader.download_model(model_dir)

    # This is the dangerous intermediate state: all ONNX paths exist, but the
    # prior voice_styles directory is still in the transaction backup.
    assert has_all_onnx_modules(model_dir)
    assert not (model_dir / VOICE_STYLES_DIR / "M1.json").exists()

    monkeypatch.setattr(os, "replace", replace)
    loader.download_model(model_dir)

    assert has_all_onnx_modules(model_dir)
    assert json.loads((model_dir / CFG_REL_PATH).read_text(encoding="utf-8"))["version"] == "new"
    assert (model_dir / "custom_styles" / "ImportedVoice.json").read_bytes() == custom_bytes
    assert (stale_stage / "sentinel").read_text(encoding="utf-8") == "stage"
    assert (stale_backup / "sentinel").read_text(encoding="utf-8") == "backup"
    assert list(tmp_path.glob(".model.download-*")) == [stale_stage]
    assert list(tmp_path.glob(".model.backup-*")) == [stale_backup]
