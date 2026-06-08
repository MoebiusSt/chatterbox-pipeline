"""
Higgs Audio V2 integration for chatterbox_tester.

Requires a local boson-ai/higgs-audio checkout. Resolve repo root via:

1. ``CHATTERBOX_TESTER_HIGGS_ROOT`` or ``HIGGS_REPO``
2. ``higgs_repo_root.txt`` at the chatterbox-pipeline repository root

**Isolation (required):**

Set ``CHATTERBOX_TESTER_HIGGS_PYTHON`` to the Python executable of ``.venv-higgs``.
The tester spawns ``higgs_worker_main.py`` in that interpreter; the pipeline venv
never imports Higgs Audio (transformers pin conflict).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


class HiggsTesterError(RuntimeError):
    """Higgs Audio is misconfigured or the worker failed."""


HIGGS_WORKER_SCRIPT = Path(__file__).resolve().parent / "higgs_worker_main.py"

_WORKER_PROC: Optional[subprocess.Popen] = None
_WORKER_ID: Optional[Tuple[str, str, str]] = None


def isolated_higgs_python() -> Optional[str]:
    raw = os.environ.get("CHATTERBOX_TESTER_HIGGS_PYTHON", "").strip()
    return raw or None


def _canonical_python_executable(py_exe_raw: str) -> str:
    """Path for subprocess without resolving venv shims to system Python."""
    p = Path(py_exe_raw).expanduser()
    return str(p.absolute() if not p.is_absolute() else p)


def higgs_repo_root_hint_file() -> Path:
    """Optional file path: chatterbox-pipeline root / higgs_repo_root.txt."""
    return Path(__file__).resolve().parents[2] / "higgs_repo_root.txt"


def _coerce_higgs_root_line(raw: str) -> Path:
    """Turn env / hint file text into an absolute Path."""
    s = raw.strip().strip('"').strip("'")
    if not s:
        return Path(".")
    s = s.translate(str.maketrans("\\", "/"))
    low = s.lower()
    for marker in ("//wsl.localhost/", "//wsl$/", "//wsl/"):
        if low.startswith(marker):
            rest = s[len(marker) :]
            if "/" in rest:
                _distro, unix_part = rest.split("/", 1)
                s = "/" + unix_part.lstrip("/")
            break
    return Path(s).expanduser().resolve()


def pipeline_repo_root() -> Path:
    """chatterbox-pipeline repository root (parent of src/)."""
    return Path(__file__).resolve().parents[2]


def _default_higgs_sibling_root() -> Optional[Path]:
    """``../higgs-audio`` next to chatterbox-pipeline (optional convenience)."""
    candidate = pipeline_repo_root().parent / "higgs-audio"
    if (candidate / "boson_multimodal").is_dir():
        return candidate.resolve()
    return None


def resolve_higgs_root() -> Optional[Path]:
    for key in ("CHATTERBOX_TESTER_HIGGS_ROOT", "HIGGS_REPO"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return _coerce_higgs_root_line(raw)

    hint = higgs_repo_root_hint_file()
    if hint.is_file():
        try:
            for line in hint.read_text(encoding="utf-8").splitlines():
                s = line.split("#", 1)[0].strip()
                if s:
                    return _coerce_higgs_root_line(s)
        except OSError:
            pass

    return _default_higgs_sibling_root()


def apply_higgs_tester_env_defaults() -> None:
    """Set Higgs env vars from higgs_repo_root.txt / sibling path when unset.

    Called at chatterbox_tester startup so ``python chatterbox_tester.py`` works
    without ``run_chatterbox_tester_higgs.sh``. Existing env vars are not overwritten.
    """
    root: Optional[Path] = None
    if not os.environ.get("CHATTERBOX_TESTER_HIGGS_ROOT", "").strip() and not os.environ.get(
        "HIGGS_REPO", ""
    ).strip():
        root = resolve_higgs_root()
        if root is not None:
            os.environ["CHATTERBOX_TESTER_HIGGS_ROOT"] = str(root)
    else:
        root = resolve_higgs_root()

    if not os.environ.get("CHATTERBOX_TESTER_HIGGS_PYTHON", "").strip():
        search_root = root
        if search_root is None:
            search_root = resolve_higgs_root()
        if search_root is not None:
            venv_py = search_root / ".venv-higgs" / "bin" / "python"
            if venv_py.is_file():
                os.environ["CHATTERBOX_TESTER_HIGGS_PYTHON"] = _canonical_python_executable(
                    str(venv_py)
                )


def validate_higgs_repo(root: Path) -> None:
    if not root.is_dir():
        hp = higgs_repo_root_hint_file()
        raise HiggsTesterError(
            f"Higgs repo root is not a directory: {root}\n"
            f"Set CHATTERBOX_TESTER_HIGGS_ROOT or edit:\n  {hp}"
        )
    for rel in ("boson_multimodal", "examples/generation.py"):
        if not (root / rel).exists():
            raise HiggsTesterError(
                f"Higgs checkout missing '{rel}' under {root} — check clone / path."
            )


def _device_for_higgs(pipeline_device: str) -> str:
    if pipeline_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        raise HiggsTesterError(
            "Higgs Audio needs CUDA — no GPU available (torch.cuda.is_available() is False)."
        )
    if pipeline_device == "cpu":
        raise HiggsTesterError("Higgs Audio inference is CUDA-only on this integration.")
    if pipeline_device == "mps":
        raise HiggsTesterError("Higgs Audio worker is not supported on MPS.")
    return pipeline_device


def higgs_cancel_active_generation() -> None:
    """Terminate the Higgs worker subprocess to interrupt a blocked generate."""
    dispose_higgs_worker()


def dispose_higgs_worker() -> None:
    global _WORKER_PROC, _WORKER_ID
    proc = _WORKER_PROC
    _WORKER_PROC = None
    _WORKER_ID = None
    if proc is None:
        return
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


def _ensure_higgs_worker(py_exe_raw: str, repo: Path, cuda_dev: str) -> None:
    global _WORKER_PROC, _WORKER_ID

    canon_py = _canonical_python_executable(py_exe_raw)
    canon_repo = str(repo.resolve())

    worker_script = HIGGS_WORKER_SCRIPT.resolve()
    if not worker_script.is_file():
        raise HiggsTesterError(f"Higgs worker script missing: {worker_script}")

    ident: Tuple[str, str, str] = (canon_py, canon_repo, cuda_dev)
    if (
        _WORKER_PROC is not None
        and _WORKER_PROC.poll() is None
        and _WORKER_ID == ident
    ):
        return

    dispose_higgs_worker()

    proc = subprocess.Popen(
        [canon_py, str(worker_script), canon_repo, cuda_dev],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(canon_repo),
    )

    out = proc.stdout
    err_pipe = proc.stderr
    if out is None or err_pipe is None:
        dispose_higgs_worker()
        raise HiggsTesterError(
            "Higgs worker missing stdio pipes (unexpected subprocess configuration)"
        )

    ready_line = out.readline()
    if proc.poll() is not None:
        stderr_tail = err_pipe.read()[-4000:]
        raise HiggsTesterError(
            "Higgs worker exited during startup "
            f"(exit {proc.poll()}). Is CHATTERBOX_TESTER_HIGGS_PYTHON pointing to "
            f".venv-higgs/bin/python? STDERR tail:\n{stderr_tail}"
        )
    try:
        ready = json.loads(ready_line)
    except json.JSONDecodeError as e:
        stderr_tail = err_pipe.read()[-4000:]
        dispose_higgs_worker()
        raise HiggsTesterError(
            f"Higgs worker invalid handshake JSON ({e}); stderr:\n{stderr_tail}"
        ) from e
    if not ready.get("ok"):
        stderr_tail = err_pipe.read()[-4000:]
        dispose_higgs_worker()
        raise HiggsTesterError(
            f"Higgs worker failed handshake ({ready!r}). STDERR:\n{stderr_tail}"
        )

    _WORKER_PROC = proc
    _WORKER_ID = ident


def higgs_prepare_runtime(pipeline_device: str) -> None:
    """Warm Higgs Audio worker subprocess."""
    root = resolve_higgs_root()
    if root is None:
        hp = higgs_repo_root_hint_file()
        raise HiggsTesterError(
            "Higgs repo path not set. Export CHATTERBOX_TESTER_HIGGS_ROOT (or "
            f"HIGGS_REPO), or put the checkout path on one line in:\n  {hp}\n"
            "See CHATTERBOX_TESTER.md."
        )
    validate_higgs_repo(root)

    py = isolated_higgs_python()
    if not py:
        raise HiggsTesterError(
            "CHATTERBOX_TESTER_HIGGS_PYTHON is not set. Point it at "
            "higgs-audio/.venv-higgs/bin/python (see scripts/run_chatterbox_tester_higgs.sh)."
        )
    if not Path(py).is_file():
        raise HiggsTesterError(
            f"CHATTERBOX_TESTER_HIGGS_PYTHON is not a file: {py!r}"
        )

    cuda_dev = _device_for_higgs(pipeline_device)
    print(
        f"Higgs Audio: isolated interpreter {py}\n"
        "     (Warm worker — first load may take ~90s.)"
    )
    _ensure_higgs_worker(py, root, cuda_dev)
    print("Higgs Audio worker ready.")


def _higgs_request_worker(req: Dict[str, Any]) -> None:
    proc = _WORKER_PROC
    if proc is None or proc.stdin is None or proc.stdout is None:
        raise HiggsTesterError("Higgs worker is not running.")
    if proc.poll() is not None:
        raise HiggsTesterError("Higgs worker exited unexpectedly.")

    proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        err = ""
        if proc.stderr is not None:
            try:
                err = proc.stderr.read()[-6000:]
            except Exception:
                pass
        raise HiggsTesterError(f"Higgs worker produced no stdout. STDERR:\n{err}")

    resp = json.loads(line)
    if not resp.get("ok"):
        err = resp.get("error", resp)
        raise HiggsTesterError(f"Higgs worker error: {err}")


def higgs_generate_to_file(
    pipeline_device: str,
    *,
    transcript: str,
    output_path: str,
    scene_prompt: Optional[str],
    ref_audio_path: Optional[str],
    profile_text: Optional[str],
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> None:
    """Run one Higgs generation via the warm worker."""
    root = resolve_higgs_root()
    if root is None:
        raise HiggsTesterError("resolve_higgs_root returned None unexpectedly.")
    py = isolated_higgs_python()
    if not py:
        raise HiggsTesterError("CHATTERBOX_TESTER_HIGGS_PYTHON is not set.")

    cuda_dev = _device_for_higgs(pipeline_device)
    _ensure_higgs_worker(py, root, cuda_dev)

    req: Dict[str, Any] = {
        "cmd": "generate",
        "transcript": transcript,
        "output": output_path,
        "scene_prompt": scene_prompt,
        "ref_audio_path": ref_audio_path,
        "profile_text": profile_text,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "seed": int(seed),
    }
    _higgs_request_worker(req)
