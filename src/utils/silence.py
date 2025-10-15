"""
Utilities to temporarily silence noisy third-party imports and setup routines.

Use in non-verbose mode to suppress stdout/stderr spam and transient warnings
emitted by external libraries during lazy import or model setup. In verbose
mode, the context becomes a no-op so all messages remain visible.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
import warnings
from typing import Iterator, Optional


def _is_verbose() -> bool:
    """Detect whether the root logger is in DEBUG (verbose) mode."""
    try:
        return logging.getLogger().isEnabledFor(logging.DEBUG)
    except Exception:
        return False


@contextlib.contextmanager
def quiet_imports_and_warnings(enabled: Optional[bool] = None) -> Iterator[None]:
    """Suppress stdout/stderr and most warnings for the duration of the context.

    Args:
        enabled: If True, suppression is active; if False, no suppression; if None,
                 suppression is active when not in verbose (DEBUG) mode.
    """
    do_silence = _is_verbose() is False if enabled is None else bool(enabled)
    if not do_silence:
        # No-op in verbose mode
        yield
        return

    # Capture stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    # Backup warnings filter and showwarning
    warnings_filters_backup = warnings.filters[:]
    old_showwarning = warnings.showwarning

    try:
        sys.stdout = stdout_buf  # type: ignore[assignment]
        sys.stderr = stderr_buf  # type: ignore[assignment]

        # Aggressively silence common noisy warnings from third-party libs
        warnings.simplefilter("ignore")
        for cat in (UserWarning, FutureWarning, DeprecationWarning, RuntimeWarning):
            warnings.filterwarnings("ignore", category=cat)

        # Avoid printing by custom warning hooks
        def _noop_showwarning(*_a, **_k):  # noqa: D401
            return None

        warnings.showwarning = _noop_showwarning  # type: ignore[assignment]

        yield
    finally:
        # Restore warnings machinery
        warnings.filters = list(warnings_filters_backup)
        warnings.showwarning = old_showwarning  # type: ignore[assignment]

        # Flush and drop captured output (we do not re-emit in non-verbose mode)
        try:
            _ = stdout_buf.getvalue()
            _ = stderr_buf.getvalue()
        except Exception:
            pass

        # Restore streams
        sys.stdout = old_stdout  # type: ignore[assignment]
        sys.stderr = old_stderr  # type: ignore[assignment]


