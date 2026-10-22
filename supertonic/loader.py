"""Model loading and voice style management utilities.

This module handles downloading, loading, and managing Supertonic TTS models
and voice styles from HuggingFace Hub.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

from .config import (
    CFG_REL_PATH,
    DEFAULT_INTER_OP_NUM_THREADS,
    DEFAULT_INTRA_OP_NUM_THREADS,
    DEFAULT_MODEL,
    DEFAULT_MODEL_REPO,
    DEFAULT_ONNX_PROVIDERS,
    DP_ONNX_REL_PATH,
    ONNX_DIR,
    TEXT_ENC_ONNX_REL_PATH,
    UNICODE_INDEXER_REL_PATH,
    VECTOR_EST_ONNX_REL_PATH,
    VOCODER_ONNX_REL_PATH,
    VOICE_STYLES_DIR,
    get_model_cache_dir,
    get_model_repo,
    get_model_revision,
)
from .core import Style, Supertonic, UnicodeProcessor
from .utils import validate_voice_style_format

logger = logging.getLogger(__name__)

_MODEL_MANAGED_DIRECTORIES = (ONNX_DIR, VOICE_STYLES_DIR)
_DOWNLOAD_TRANSACTION_VERSION = 1


def get_cache_dir(model_name: Optional[str] = None) -> Path:
    """Get or create the cache directory for Supertonic models.

    Args:
        model_name: Model name (one of ``AVAILABLE_MODELS``).
            If None, uses default cache directory.

    Returns:
        Path object pointing to the cache directory

    Note:
        Default location depends on the loaded model
        (e.g., ``~/.cache/supertonic3`` for supertonic-3), but can be
        overridden with the ``SUPERTONIC_CACHE_DIR`` environment variable.
        The override is resolved on every call (not snapshotted at import),
        so late-set env vars are honored.
    """
    cache_dir = get_model_cache_dir(model_name or DEFAULT_MODEL)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Using cache directory: {cache_dir}")
    return cache_dir


def get_all_onnx_module_relative_paths() -> list[Path]:
    """Get list of all required ONNX model file paths.

    Returns:
        List of Path objects for each required ONNX model file
    """
    return [
        DP_ONNX_REL_PATH,
        TEXT_ENC_ONNX_REL_PATH,
        VECTOR_EST_ONNX_REL_PATH,
        VOCODER_ONNX_REL_PATH,
    ]


def has_all_onnx_modules(model_dir: Union[Path, str]) -> bool:
    """Check if all required ONNX model files exist in the directory.

    Args:
        model_dir: Directory to check for model files (str or Path)

    Returns:
        True if all required ONNX files exist, False otherwise
    """
    model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir
    module_rel_paths = get_all_onnx_module_relative_paths()
    all_exist = all((model_dir / path).exists() for path in module_rel_paths)

    if not all_exist:
        missing = [str(path) for path in module_rel_paths if not (model_dir / path).exists()]
        logger.debug(f"Missing ONNX files: {missing}")

    return all_exist


def _path_entry_exists(path: Path) -> bool:
    """Check for a directory entry without treating inspection errors as absence."""
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


@contextmanager
def _model_download_lock(model_dir: Path, *, allow_read_only: bool = False) -> Iterator[None]:
    """Prevent refreshes from publishing while a model is recovered or checked."""
    lock_path = model_dir.parent / f".{model_dir.name}.download.lock"
    read_only_lock_file = False
    try:
        lock_file = lock_path.open("a+b")
    except PermissionError:
        if not allow_read_only:
            raise

        lock_exists = _path_entry_exists(lock_path)
        if lock_exists and lock_path.is_symlink():
            raise

        if lock_exists:
            # The persistent lock file can still serialize readers and writers
            # when this installation can read, but cannot append to, the file.
            lock_file = lock_path.open("rb")
            read_only_lock_file = True
        else:
            marker = _transaction_path(model_dir)
            parent = model_dir.parent
            if (
                _path_entry_exists(marker)
                or os.access(parent, os.W_OK | os.X_OK)
                or not os.access(parent, os.R_OK | os.X_OK)
            ):
                # An absent marker is not enough: a writer may start after the
                # check. Without a lock file, only a parent this process cannot
                # write to rules out a new refresh under the same permissions.
                raise

            # No lock file means no writer has started; the read-only parent
            # prevents one from starting. Pending or suspicious journals above
            # fail closed because they cannot be recovered without the lock.
            yield
            return

    try:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0 and not read_only_lock_file:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)

            def unlock() -> None:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)

        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            def unlock() -> None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    except OSError as e:
        lock_file.close()
        raise RuntimeError(f"Another model download is already in progress for {model_dir}") from e

    try:
        yield
    finally:
        try:
            unlock()
        finally:
            lock_file.close()


def _transaction_path(model_dir: Path) -> Path:
    return model_dir.parent / f".{model_dir.name}.download-transaction.json"


def _write_download_transaction(marker: Path, transaction: dict) -> None:
    """Atomically write the small recovery journal beside the model directory."""
    temp_marker = marker.with_name(f"{marker.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temp_marker.open("x", encoding="utf-8") as f:
            json.dump(transaction, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_marker, marker)
    finally:
        if temp_marker.exists():
            temp_marker.unlink()


def _read_download_transaction(model_dir: Path, marker: Path) -> tuple[dict, Path]:
    if marker.is_symlink() or not marker.is_file():
        raise RuntimeError(f"Unsafe model download recovery marker: {marker}")

    try:
        transaction = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Could not read model download recovery marker: {marker}") from e

    managed = [str(path) for path in _MODEL_MANAGED_DIRECTORIES]
    if (
        not isinstance(transaction, dict)
        or transaction.get("version") != _DOWNLOAD_TRANSACTION_VERSION
        or transaction.get("model_dir") != model_dir.name
        or transaction.get("managed") != managed
        or transaction.get("state") not in {"prepared", "committed"}
        or not isinstance(transaction.get("originals"), list)
        or any(path not in managed for path in transaction["originals"])
        or not isinstance(transaction.get("model_dir_existed"), bool)
    ):
        raise RuntimeError(f"Invalid model download recovery marker: {marker}")

    backup_name = transaction.get("backup_dir")
    backup_prefix = f".{model_dir.name}.backup-"
    if (
        not isinstance(backup_name, str)
        or Path(backup_name).name != backup_name
        or not backup_name.startswith(backup_prefix)
    ):
        raise RuntimeError(f"Invalid backup path in model download recovery marker: {marker}")

    backup_dir = model_dir.parent / backup_name
    if backup_dir.is_symlink():
        raise RuntimeError(f"Unsafe model download backup directory: {backup_dir}")
    return transaction, backup_dir


def _remove_managed_directory(path: Path) -> None:
    """Remove one reserved model asset directory without following links."""
    if path.is_symlink():
        raise RuntimeError(f"Refusing to replace a symlink at a managed model path: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _remove_download_backup(model_dir: Path, backup_dir: Path) -> None:
    expected_prefix = f".{model_dir.name}.backup-"
    if (
        backup_dir.parent != model_dir.parent
        or not backup_dir.name.startswith(expected_prefix)
        or backup_dir.is_symlink()
    ):
        raise RuntimeError(f"Refusing to clean an unowned model backup path: {backup_dir}")
    if not backup_dir.exists():
        return

    managed_names = {path.name for path in _MODEL_MANAGED_DIRECTORIES}
    for child in backup_dir.iterdir():
        if child.name not in managed_names or child.is_symlink() or not child.is_dir():
            raise RuntimeError(f"Unexpected content in model backup directory: {child}")
    shutil.rmtree(backup_dir)


def _recover_model_download(model_dir: Path) -> None:
    """Restore a prepared publish or finish cleanup after a committed publish."""
    if model_dir.is_symlink() or (model_dir.exists() and not model_dir.is_dir()):
        raise RuntimeError(f"Model destination is not a safe directory: {model_dir}")

    marker = _transaction_path(model_dir)
    if not marker.exists() and not marker.is_symlink():
        return

    transaction, backup_dir = _read_download_transaction(model_dir, marker)
    originals = set(transaction["originals"])

    if transaction["state"] == "prepared":
        if not model_dir.exists():
            model_dir.mkdir(parents=True)
        for relative_path in _MODEL_MANAGED_DIRECTORIES:
            destination = model_dir / relative_path
            backup = backup_dir / relative_path
            if backup.exists():
                _remove_managed_directory(destination)
                os.replace(backup, destination)
            elif str(relative_path) not in originals:
                _remove_managed_directory(destination)
            elif not destination.exists():
                raise RuntimeError(
                    f"Previous model asset is missing from both its original and backup path: "
                    f"{destination}"
                )

    _remove_download_backup(model_dir, backup_dir)
    marker.unlink()

    if not transaction["model_dir_existed"] and model_dir.exists():
        try:
            model_dir.rmdir()
        except OSError:
            # User data or a later operation may have populated the directory.
            pass


def _validate_downloaded_model(snapshot_dir: Path) -> None:
    """Reject incomplete or unsafe snapshots before changing the installed model."""
    if snapshot_dir.is_symlink() or not snapshot_dir.is_dir():
        raise ValueError("Downloaded model snapshot is not a safe directory")

    onnx_dir = snapshot_dir / ONNX_DIR
    voice_styles_dir = snapshot_dir / VOICE_STYLES_DIR
    for directory in (onnx_dir, voice_styles_dir):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Downloaded model is missing a required directory: {directory}")

    required_files = (
        CFG_REL_PATH,
        UNICODE_INDEXER_REL_PATH,
        *get_all_onnx_module_relative_paths(),
    )
    for relative_path in required_files:
        path = snapshot_dir / relative_path
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Downloaded model is missing a required file: {path}")

    configs = load_configs(snapshot_dir)
    if not isinstance(configs, dict):
        raise TypeError(
            f"Downloaded model configuration is not an object: {snapshot_dir / CFG_REL_PATH}"
        )
    load_text_processor(snapshot_dir)

    style_paths = sorted(voice_styles_dir.glob("*.json"))
    if not style_paths:
        raise ValueError(f"Downloaded model has no built-in voice styles: {voice_styles_dir}")
    for path in style_paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Downloaded model contains an unsafe voice style: {path}")
        try:
            load_voice_style_from_json_file(path)
        except (OSError, TypeError, ValueError) as e:
            raise ValueError(f"Downloaded model has an invalid voice style: {path}") from e


def _materialize_downloaded_symlinks(snapshot_dir: Path) -> None:
    """Copy Hub-managed symlinks in owned model folders into the staging tree."""
    for relative_dir in _MODEL_MANAGED_DIRECTORIES:
        directory = snapshot_dir / relative_dir
        if not directory.is_dir():
            continue
        for path in list(directory.rglob("*")):
            if not path.is_symlink():
                continue
            if not path.is_file():
                raise ValueError(f"Downloaded model contains an invalid file link: {path}")

            materialized = path.with_name(f".{path.name}.copy-{uuid.uuid4().hex}")
            try:
                shutil.copyfile(path, materialized)
                os.replace(materialized, path)
            finally:
                if materialized.exists():
                    materialized.unlink()


def _publish_downloaded_model(model_dir: Path, snapshot_dir: Path) -> None:
    """Publish only model-owned directories, preserving custom and unrelated files."""
    if model_dir.is_symlink() or (model_dir.exists() and not model_dir.is_dir()):
        raise RuntimeError(f"Model destination is not a safe directory: {model_dir}")

    if model_dir.exists():
        for relative_path in _MODEL_MANAGED_DIRECTORIES:
            destination = model_dir / relative_path
            if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
                raise RuntimeError(f"Unsafe existing model asset path: {destination}")

    backup_dir = Path(
        tempfile.mkdtemp(prefix=f".{model_dir.name}.backup-", dir=str(model_dir.parent))
    )
    originals = [
        str(relative_path)
        for relative_path in _MODEL_MANAGED_DIRECTORIES
        if (model_dir / relative_path).is_dir()
    ]
    transaction = {
        "version": _DOWNLOAD_TRANSACTION_VERSION,
        "model_dir": model_dir.name,
        "backup_dir": backup_dir.name,
        "managed": [str(path) for path in _MODEL_MANAGED_DIRECTORIES],
        "originals": originals,
        "model_dir_existed": model_dir.exists(),
        "state": "prepared",
    }
    marker = _transaction_path(model_dir)

    try:
        _write_download_transaction(marker, transaction)
        model_dir.mkdir(parents=True, exist_ok=True)
        for relative_path in _MODEL_MANAGED_DIRECTORIES:
            destination = model_dir / relative_path
            if destination.exists():
                os.replace(destination, backup_dir / relative_path)
        for relative_path in _MODEL_MANAGED_DIRECTORIES:
            destination = model_dir / relative_path
            os.replace(snapshot_dir / relative_path, destination)

        transaction["state"] = "committed"
        _write_download_transaction(marker, transaction)
    except Exception:
        try:
            if marker.exists() or marker.is_symlink():
                _recover_model_download(model_dir)
            else:
                _remove_download_backup(model_dir, backup_dir)
        except Exception as recovery_error:  # noqa: BLE001 - preserve the original publish failure
            logger.error(
                "Model publish failed and recovery is still pending at %s: %s",
                marker,
                recovery_error,
            )
        raise

    try:
        _recover_model_download(model_dir)
    except (OSError, RuntimeError) as cleanup_error:
        # The new model is already committed. Keep the journal so a later
        # download can safely finish removing only this operation's backup.
        logger.warning("Model updated, but old model backup cleanup is pending: %s", cleanup_error)


def _cleanup_download_stage(model_dir: Path, snapshot_dir: Path) -> None:
    expected_prefix = f".{model_dir.name}.download-"
    if snapshot_dir.parent != model_dir.parent or not snapshot_dir.name.startswith(expected_prefix):
        logger.warning("Refusing to clean an unowned model staging path: %s", snapshot_dir)
        return
    try:
        if snapshot_dir.is_symlink():
            snapshot_dir.unlink()
        elif snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
    except OSError as cleanup_error:
        logger.warning(
            "Failed to clean model staging directory %s: %s", snapshot_dir, cleanup_error
        )


def download_model(model_dir: Union[Path, str], model_name: Optional[str] = None) -> None:
    """Download Supertonic model from HuggingFace Hub.

    Args:
        model_dir: Directory where the model should be downloaded (str or Path)
        model_name: Model name (one of ``AVAILABLE_MODELS``).
            If None, uses default model repository.

    Only the model-owned ``onnx`` and ``voice_styles`` directories are replaced.
    Imported ``custom_styles`` and other files in ``model_dir`` are preserved.
    """
    model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir

    # Determine which repo and revision to download
    effective_model = model_name or DEFAULT_MODEL
    if model_name is not None:
        repo_id = get_model_repo(model_name)
    else:
        repo_id = DEFAULT_MODEL_REPO
    revision = get_model_revision(effective_model)

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _model_download_lock(model_dir):
            _recover_model_download(model_dir)
            temp_dir = Path(
                tempfile.mkdtemp(prefix=f".{model_dir.name}.download-", dir=str(model_dir.parent))
            )
            try:
                from huggingface_hub import snapshot_download

                logger.info(
                    "Downloading model from %s @ %s to temporary location: %s",
                    repo_id,
                    revision[:12],
                    temp_dir,
                )
                download_options = {
                    "repo_id": repo_id,
                    "local_dir": str(temp_dir),
                    "revision": revision,
                }
                try:
                    supported_options = inspect.signature(snapshot_download).parameters
                except (TypeError, ValueError):
                    supported_options = {}
                if "local_dir_use_symlinks" in supported_options:
                    # Older Hub versions default to symlinking files over 5 MiB.
                    # Newer versions removed the option, so feature-detect it.
                    download_options["local_dir_use_symlinks"] = False
                snapshot_download(**download_options)
                _materialize_downloaded_symlinks(temp_dir)
                _validate_downloaded_model(temp_dir)
                _publish_downloaded_model(model_dir, temp_dir)
                logger.info("Model download completed successfully")
            finally:
                _cleanup_download_stage(model_dir, temp_dir)

    except ImportError as e:
        logger.error("huggingface_hub not installed")
        raise RuntimeError(
            "Failed to import huggingface_hub. Please install it with: "
            "pip install huggingface-hub"
        ) from e
    except Exception as e:
        logger.error("Model download failed: %s", e)
        raise RuntimeError(
            f"Failed to download model from {repo_id}. "
            f"Please check your internet connection and try again. "
            f"Error: {e}"
        ) from e


def load_configs(model_dir: Union[Path, str]) -> dict:
    """Load model configuration from JSON file.

    Args:
        model_dir: Directory containing the model files (str or Path)

    Returns:
        Dictionary containing model configuration
    """
    model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir
    cfg_path = model_dir / CFG_REL_PATH

    if not cfg_path.exists():
        logger.error(f"Config file not found: {cfg_path}")
        raise FileNotFoundError(
            f"Model configuration file not found at {cfg_path}. "
            f"Please ensure the model is properly downloaded."
        )

    try:
        with open(cfg_path, "r") as f:
            cfgs = json.load(f)
        logger.debug(f"Loaded config from {cfg_path}")
        return cfgs
    except json.JSONDecodeError as e:
        logger.error(f"Invalid config format: {e}")
        raise ValueError(
            f"Model configuration file is malformed at {cfg_path}. "
            f"Please re-download the model."
        ) from e


def load_onnx_modules(
    model_dir: Union[Path, str],
    intra_op_num_threads: Optional[int] = None,
    inter_op_num_threads: Optional[int] = None,
) -> tuple[
    ort.InferenceSession,
    ort.InferenceSession,
    ort.InferenceSession,
    ort.InferenceSession,
]:
    """Load all ONNX model modules for TTS synthesis.

    Args:
        model_dir: Directory containing the ONNX model files (str or Path)
        intra_op_num_threads: Number of threads for intra-op parallelism.
            None (default) lets ONNX Runtime auto-detect optimal value
        inter_op_num_threads: Number of threads for inter-op parallelism.
            None (default) lets ONNX Runtime auto-detect optimal value

    Returns:
        Tuple of (duration_predictor, text_encoder, vector_estimator, vocoder)
        ONNX Runtime inference sessions
    """

    def _load_onnx(
        onnx_path: Path, opts: ort.SessionOptions, providers: list[str]
    ) -> ort.InferenceSession:
        """Load a single ONNX model file."""
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model file not found: {onnx_path}. "
                f"Please ensure the model is properly downloaded."
            )

        logger.debug(f"Loading ONNX model: {onnx_path}")
        return ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)

    opts = ort.SessionOptions()
    # Performance optimizations
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Only set thread counts if explicitly specified
    # Otherwise, let ONNX Runtime automatically determine optimal values
    intra_threads = (
        intra_op_num_threads if intra_op_num_threads is not None else DEFAULT_INTRA_OP_NUM_THREADS
    )
    inter_threads = (
        inter_op_num_threads if inter_op_num_threads is not None else DEFAULT_INTER_OP_NUM_THREADS
    )

    if intra_threads is not None:
        opts.intra_op_num_threads = intra_threads
    if inter_threads is not None:
        opts.inter_op_num_threads = inter_threads

    # Setup execution providers with fallback to CPU
    providers = DEFAULT_ONNX_PROVIDERS
    available_providers = ort.get_available_providers()

    # Filter to only use available providers
    valid_providers = [p for p in providers if p in available_providers]
    if not valid_providers:
        logger.warning(
            f"Requested providers {providers} not available. "
            f"Falling back to CPUExecutionProvider"
        )
        valid_providers = ["CPUExecutionProvider"]
    else:
        logger.info(f"Using ONNX providers: {valid_providers}")

    thread_info = (
        f"intra_threads={intra_threads if intra_threads is not None else 'auto'}, "
        f"inter_threads={inter_threads if inter_threads is not None else 'auto'}"
    )
    logger.info(f"ONNX Runtime config: {thread_info}")

    dp_onnx_path = model_dir / DP_ONNX_REL_PATH
    text_enc_onnx_path = model_dir / TEXT_ENC_ONNX_REL_PATH
    vector_est_onnx_path = model_dir / VECTOR_EST_ONNX_REL_PATH
    vocoder_onnx_path = model_dir / VOCODER_ONNX_REL_PATH

    logger.info(f"Loading ONNX models with providers: {valid_providers}")
    dp_ort = _load_onnx(dp_onnx_path, opts, valid_providers)
    text_enc_ort = _load_onnx(text_enc_onnx_path, opts, valid_providers)
    vector_est_ort = _load_onnx(vector_est_onnx_path, opts, valid_providers)
    vocoder_ort = _load_onnx(vocoder_onnx_path, opts, valid_providers)

    logger.info("Successfully loaded all ONNX models")
    return dp_ort, text_enc_ort, vector_est_ort, vocoder_ort


def load_text_processor(model_dir: Union[Path, str]) -> UnicodeProcessor:
    """Load the unicode text processor for the model.

    Args:
        model_dir: Directory containing the model files (str or Path)

    Returns:
        Initialized UnicodeProcessor instance
    """
    model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir
    unicode_indexer_path = model_dir / UNICODE_INDEXER_REL_PATH
    logger.debug(f"Loading text processor from {unicode_indexer_path}")
    text_processor = UnicodeProcessor(str(unicode_indexer_path))
    return text_processor


def load_model(
    model_dir: Union[Path, str],
    auto_download: bool,
    intra_op_num_threads: Optional[int] = None,
    inter_op_num_threads: Optional[int] = None,
    model_name: Optional[str] = None,
) -> Supertonic:
    """Load the complete Supertonic TTS model.

    This function loads all model components including ONNX modules,
    configuration, and text processor. If model files are missing and
    auto_download is enabled, it will download them from HuggingFace Hub.

    Args:
        model_dir: Directory containing (or to contain) the model files (str or Path)
        auto_download: If True, automatically download missing model files
        intra_op_num_threads: Number of threads for intra-op parallelism.
            None (default) lets ONNX Runtime auto-detect optimal value
        inter_op_num_threads: Number of threads for inter-op parallelism.
            None (default) lets ONNX Runtime auto-detect optimal value
        model_name: Model name (one of ``AVAILABLE_MODELS``).
            Used for downloading the correct model if auto_download is True.

    Returns:
        Initialized Supertonic TTS engine
    """
    logger.info(f"Loading model from {model_dir}")
    model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir

    # A previous process may have stopped between publishing ONNX assets and
    # voice styles. Recover that transaction before the ONNX-only completeness
    # check can incorrectly make the partial model look ready.
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    with _model_download_lock(model_dir, allow_read_only=True):
        _recover_model_download(model_dir)
        model_is_complete = has_all_onnx_modules(model_dir)

    if not model_is_complete:
        if not auto_download:
            logger.error(f"ONNX models not found in {model_dir}")
            raise FileNotFoundError(
                f"ONNX model files not found in {model_dir}. "
                f"Set auto_download=True to automatically download from HuggingFace Hub, "
                f"or manually download the model to this directory."
            )
        download_model(model_dir, model_name)

    cfgs = load_configs(model_dir)
    dp_ort, text_enc_ort, vector_est_ort, vocoder_ort = load_onnx_modules(
        model_dir, intra_op_num_threads, inter_op_num_threads
    )
    text_processor = load_text_processor(model_dir)

    logger.info("Model loaded successfully")
    return Supertonic(cfgs, text_processor, dp_ort, text_enc_ort, vector_est_ort, vocoder_ort)


def list_available_voice_style_paths(model_dir: Union[Path, str]) -> list[Path]:
    """List all available voice style JSON files in the model directory.

    Args:
        model_dir: Directory containing the model files (str or Path)

    Returns:
        Sorted list of paths to voice style JSON files
    """
    model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir
    voice_styles_dir = model_dir / VOICE_STYLES_DIR

    if not voice_styles_dir.exists():
        logger.error(f"Voice styles directory not found: {voice_styles_dir}")
        raise FileNotFoundError(
            f"Voice styles directory not found at {voice_styles_dir}. "
            f"Please ensure the model is properly downloaded."
        )

    paths = sorted(list(voice_styles_dir.glob("*.json")))
    logger.debug(f"Found {len(paths)} voice styles in {voice_styles_dir}")
    return paths


def list_available_voice_style_names(model_dir: Union[Path, str]) -> list[str]:
    """List names of all available voice styles.

    Args:
        model_dir: Directory containing the model files (str or Path)

    Returns:
        List of voice style names (without .json extension)
    """
    voice_style_paths = list_available_voice_style_paths(model_dir)
    names = [path.stem for path in voice_style_paths]
    return names


def load_voice_style_from_json_file(voice_style_path: Union[Path, str]) -> Style:
    """Load a voice style from a JSON file.

    Args:
        voice_style_path: Path to the voice style JSON file (str or Path)

    Returns:
        Style object with loaded style vectors
    """
    # Convert to Path if string
    voice_style_path = Path(voice_style_path)

    def _load_style_from_json(json_data: dict) -> np.ndarray:
        """Parse style vector from JSON data."""
        try:
            dims = json_data["dims"]
            data = json_data["data"]
            return np.array(data, dtype=np.float32).reshape(*dims)
        except KeyError as e:
            raise ValueError(f"Invalid style format: missing key {e}") from e

    if not voice_style_path.exists():
        raise FileNotFoundError(f"Voice style file not found: {voice_style_path}")

    try:
        with open(voice_style_path, "r") as f:
            voice_style_json = json.load(f)

        # Validate voice style format
        if not validate_voice_style_format(voice_style_json):
            raise ValueError(
                f"Invalid voice style format in {voice_style_path}. "
                f"Expected 'style_ttl' and 'style_dp' with 'dims' and 'data' fields."
            )

        logger.debug(f"Loading voice style from {voice_style_path}")
        ttl_style = _load_style_from_json(voice_style_json["style_ttl"])
        dp_style = _load_style_from_json(voice_style_json["style_dp"])

        return Style(ttl_style, dp_style)

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in voice style file: {e}")
        raise ValueError(
            f"Voice style file is malformed at {voice_style_path}. "
            f"Please check the file format."
        ) from e
    except (KeyError, ValueError) as e:
        logger.error(f"Invalid voice style format: {e}")
        raise ValueError(
            f"Voice style file has invalid format at {voice_style_path}. "
            f"Expected 'style_ttl' and 'style_dp' fields. Error: {e}"
        ) from e


def load_voice_style_from_name(model_dir: Union[Path, str], voice_name: str) -> Style:
    """Load a voice style by name from the model directory.

    Args:
        model_dir: Directory containing the model files (str or Path)
        voice_name: Name of the voice style (without .json extension)

    Returns:
        Style object with loaded style vectors
    """
    model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir
    voice_style_path = model_dir / VOICE_STYLES_DIR / f"{voice_name}.json"

    if not voice_style_path.exists():
        available = list_available_voice_style_names(model_dir)
        logger.error(f"Voice style '{voice_name}' not found")
        raise FileNotFoundError(
            f"Voice style '{voice_name}' not found. "
            f"Available voice styles: {', '.join(available)}"
        )

    return load_voice_style_from_json_file(voice_style_path)
