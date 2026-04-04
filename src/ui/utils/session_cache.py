"""
IDS ML Analyzer — Session Cache Manager

Parquet-based disk-backed cache for inter-tab DataFrame transfer in Streamlit.

Problem:
    Storing large DataFrames in ``st.session_state`` causes ``MemoryError``
    when datasets exceed ~300 MB. Tab switching triggers copies, and garbage
    collection is insufficient because Streamlit re-serializes state on
    every rerun.

Solution:
    - Write DataFrames to temporary Parquet files on disk.
    - Each tab reads from disk on demand — peak RAM = max(tab1, tab2),
      not sum(tab1, tab2).
    - TTL-based expiry ensures stale cache files don't accumulate.
    - ``atexit`` hook cleans up on normal interpreter shutdown.
    - Startup sweep removes orphaned files from crashed sessions.

Usage::

    from src.ui.utils.session_cache import get_session_cache

    cache = get_session_cache()

    # Store a large DataFrame
    cache.store("scan_results", result_df)

    # Load in another tab
    df = cache.load("scan_results")
    if df is not None:
        st.dataframe(df)

    # Clean up when done
    cache.clear("scan_results")

Storage Layout::

    datasets/.cache/
        session_{session_id}/
            scan_results.parquet
            training_data.parquet
            ...

Thread Safety:
    All file operations use atomic-write patterns (write to temp file,
    then rename) to prevent partial reads during Streamlit reruns.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# Default TTL for cached files (2 hours).
_DEFAULT_TTL_SECONDS: float = 2 * 60 * 60.0

# Maximum total cache size per session (2 GB).
_MAX_SESSION_CACHE_BYTES: int = 2 * 1024 * 1024 * 1024

# Orphan threshold: files older than this are swept on startup (6 hours).
_ORPHAN_THRESHOLD_SECONDS: float = 6 * 60 * 60.0

# Cache root — relative to project root, under datasets/.cache/
_CACHE_ROOT_NAME = ".cache"


def _get_cache_root() -> Path:
    """Resolve the cache root directory.

    Uses ``datasets/.cache/`` under the project root. Falls back to
    a system tempdir if the datasets directory is not writable.

    Returns:
        Absolute path to the cache root directory.
    """
    # Try project-local cache first.
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    candidates = [
        project_root / "datasets" / _CACHE_ROOT_NAME,
        Path(tempfile.gettempdir()) / "ids_ml_cache",
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            # Test write access.
            test_file = candidate / ".write_test"
            test_file.write_text("ok")
            test_file.unlink()
            return candidate
        except (OSError, PermissionError):
            continue

    # Last resort: system tempdir (always writable).
    fallback = Path(tempfile.mkdtemp(prefix="ids_ml_cache_"))
    logger.warning(
        "[SessionCache] Using fallback cache dir: %s", fallback
    )
    return fallback


class SessionCacheManager:
    """Disk-backed DataFrame cache for Streamlit session state offloading.

    Stores DataFrames as Parquet files in a session-specific subdirectory.
    Each cache entry is a single Parquet file named ``{key}.parquet``.

    Args:
        session_id: Unique identifier for this session. If None, a hash
            of the current process PID and start time is used.
        cache_root: Root directory for all cache files. If None, uses
            ``datasets/.cache/`` under the project root.
        ttl_seconds: Time-to-live for cached files. Files older than
            this are considered expired and may be cleaned up.

    Attributes:
        session_dir: The session-specific subdirectory.
        ttl_seconds: TTL for cached entries.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        cache_root: Optional[Path] = None,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._cache_root = cache_root or _get_cache_root()
        self.ttl_seconds = ttl_seconds

        # Derive a stable session directory name.
        if session_id is None:
            raw = f"{os.getpid()}_{id(self)}_{time.monotonic_ns()}"
            session_id = hashlib.md5(raw.encode()).hexdigest()[:12]

        self._session_id = session_id
        self.session_dir = self._cache_root / f"session_{session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Register cleanup on interpreter exit.
        atexit.register(self._cleanup_session)

        # Sweep orphaned sessions from previous crashes.
        self._sweep_orphans()

        logger.info(
            "[SessionCache] Initialized. session=%s, dir=%s, ttl=%ds",
            session_id,
            self.session_dir,
            int(ttl_seconds),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, key: str, df: pd.DataFrame) -> Path:
        """Write a DataFrame to disk as a Parquet file.

        Uses atomic write (write to temp, then rename) to prevent
        partial reads during concurrent Streamlit reruns.

        Args:
            key: Cache key (used as filename stem). Must be a valid
                filename component (no slashes, no dots).
            df: DataFrame to store.

        Returns:
            Path to the written Parquet file.

        Raises:
            ValueError: If key is empty or contains invalid characters.
            OSError: If write fails due to disk issues.
        """
        self._validate_key(key)
        target_path = self._key_to_path(key)

        # Atomic write: write to temp file in same directory, then rename.
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            dir=str(self.session_dir),
            suffix=".parquet.tmp",
            prefix=f"{key}_",
        )
        tmp_path = Path(tmp_path_str)
        try:
            os.close(tmp_fd)
            df.to_parquet(tmp_path, engine="pyarrow", index=True)
            # Atomic rename (same filesystem → guaranteed atomic on POSIX;
            # on Windows, os.replace is the closest equivalent).
            tmp_path.replace(target_path)
            logger.info(
                "[SessionCache] Stored '%s': %d rows, %.1f MB",
                key,
                len(df),
                target_path.stat().st_size / (1024 * 1024),
            )
        except Exception:
            # Clean up temp file on failure.
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

        # Check total session cache size.
        self._enforce_size_limit()

        return target_path

    def load(self, key: str) -> Optional[pd.DataFrame]:
        """Read a DataFrame from the disk cache.

        Returns None if the key doesn't exist or the file is expired.

        Args:
            key: Cache key to load.

        Returns:
            DataFrame if found and not expired, None otherwise.
        """
        self._validate_key(key)
        target_path = self._key_to_path(key)

        if not target_path.exists():
            logger.debug("[SessionCache] Cache miss for '%s'.", key)
            return None

        # Check TTL.
        file_age = time.time() - target_path.stat().st_mtime
        if file_age > self.ttl_seconds:
            logger.info(
                "[SessionCache] Entry '%s' expired (age=%.0fs > ttl=%.0fs).",
                key,
                file_age,
                self.ttl_seconds,
            )
            target_path.unlink(missing_ok=True)
            return None

        try:
            df = pd.read_parquet(target_path, engine="pyarrow")
            logger.debug(
                "[SessionCache] Cache hit for '%s': %d rows.", key, len(df)
            )
            return df
        except Exception as exc:
            logger.warning(
                "[SessionCache] Failed to read '%s': %s. Removing corrupt file.",
                key,
                exc,
            )
            target_path.unlink(missing_ok=True)
            return None

    def clear(self, key: Optional[str] = None) -> None:
        """Remove cached entry or all entries for this session.

        Args:
            key: If provided, remove only this key. If None, remove
                ALL cached files for this session.
        """
        if key is not None:
            self._validate_key(key)
            target_path = self._key_to_path(key)
            if target_path.exists():
                target_path.unlink(missing_ok=True)
                logger.info("[SessionCache] Cleared '%s'.", key)
        else:
            # Clear all files in session directory.
            for file in self.session_dir.glob("*.parquet"):
                file.unlink(missing_ok=True)
            logger.info(
                "[SessionCache] Cleared all entries for session '%s'.",
                self._session_id,
            )

    def exists(self, key: str) -> bool:
        """Check if a non-expired cache entry exists.

        Args:
            key: Cache key to check.

        Returns:
            True if the key exists and is not expired.
        """
        self._validate_key(key)
        target_path = self._key_to_path(key)
        if not target_path.exists():
            return False
        file_age = time.time() - target_path.stat().st_mtime
        return file_age <= self.ttl_seconds

    def list_keys(self) -> list[str]:
        """List all non-expired cache keys for this session.

        Returns:
            List of key strings.
        """
        keys: list[str] = []
        for file in self.session_dir.glob("*.parquet"):
            file_age = time.time() - file.stat().st_mtime
            if file_age <= self.ttl_seconds:
                keys.append(file.stem)
        return sorted(keys)

    @property
    def total_size_bytes(self) -> int:
        """Total size of all cached files in this session (bytes)."""
        return sum(
            f.stat().st_size
            for f in self.session_dir.glob("*.parquet")
            if f.exists()
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key_to_path(self, key: str) -> Path:
        """Convert a cache key to an absolute file path."""
        return self.session_dir / f"{key}.parquet"

    @staticmethod
    def _validate_key(key: str) -> None:
        """Validate that a cache key is safe for use as a filename.

        Args:
            key: The key to validate.

        Raises:
            ValueError: If key is empty, contains path separators,
                dots, or other unsafe characters.
        """
        if not key or not key.strip():
            raise ValueError("Cache key cannot be empty.")
        unsafe = set("/\\.:*?\"<>|")
        if any(char in unsafe for char in key):
            raise ValueError(
                f"Cache key '{key}' contains unsafe characters. "
                f"Use only alphanumeric characters and underscores."
            )

    def _enforce_size_limit(self) -> None:
        """Remove oldest entries if total cache size exceeds the limit."""
        total = self.total_size_bytes
        if total <= _MAX_SESSION_CACHE_BYTES:
            return

        # Sort files by modification time (oldest first) and remove
        # until we're under the limit.
        files = sorted(
            self.session_dir.glob("*.parquet"),
            key=lambda f: f.stat().st_mtime,
        )
        for file in files:
            if total <= _MAX_SESSION_CACHE_BYTES:
                break
            size = file.stat().st_size
            file.unlink(missing_ok=True)
            total -= size
            logger.info(
                "[SessionCache] Evicted '%s' (%.1f MB) to enforce size limit.",
                file.stem,
                size / (1024 * 1024),
            )

    def _sweep_orphans(self) -> None:
        """Remove session directories from previous crashed sessions.

        A session directory is considered orphaned if ALL its files
        are older than ``_ORPHAN_THRESHOLD_SECONDS``.
        """
        now = time.time()
        try:
            for session_dir in self._cache_root.iterdir():
                if not session_dir.is_dir():
                    continue
                if session_dir == self.session_dir:
                    continue  # Don't sweep ourselves.
                if not session_dir.name.startswith("session_"):
                    continue

                # Check if all files are older than the orphan threshold.
                files = list(session_dir.glob("*"))
                if not files:
                    # Empty directory — remove it.
                    shutil.rmtree(session_dir, ignore_errors=True)
                    continue

                newest_mtime = max(f.stat().st_mtime for f in files if f.exists())
                if (now - newest_mtime) > _ORPHAN_THRESHOLD_SECONDS:
                    shutil.rmtree(session_dir, ignore_errors=True)
                    logger.info(
                        "[SessionCache] Swept orphaned session dir: %s",
                        session_dir.name,
                    )
        except (OSError, PermissionError) as exc:
            logger.debug(
                "[SessionCache] Orphan sweep failed (non-critical): %s", exc
            )

    def _cleanup_session(self) -> None:
        """Remove this session's cache directory on interpreter exit.

        Registered via ``atexit.register`` in ``__init__``.
        Silently ignores errors (common during interpreter shutdown).
        """
        try:
            if self.session_dir.exists():
                shutil.rmtree(self.session_dir, ignore_errors=True)
                logger.debug(
                    "[SessionCache] Cleaned up session dir: %s",
                    self.session_dir.name,
                )
        except Exception:
            pass  # Interpreter shutting down — nothing we can do.


# ---------------------------------------------------------------------------
# Module-level singleton accessor (for Streamlit caching)
# ---------------------------------------------------------------------------

_global_cache: Optional[SessionCacheManager] = None


def get_session_cache(
    session_id: Optional[str] = None,
) -> SessionCacheManager:
    """Get or create the global SessionCacheManager singleton.

    In Streamlit, call this once per session. The singleton pattern
    ensures all tabs share the same cache instance.

    Args:
        session_id: Optional session ID. If None, derives from PID.

    Returns:
        The global SessionCacheManager instance.
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = SessionCacheManager(session_id=session_id)
    return _global_cache
