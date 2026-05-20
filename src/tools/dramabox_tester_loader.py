"""
DramaBox integration for chatterbox_tester.

Requires a local resemble-ai/DramaBox clone. Resolve DramaBox repo root:

1. ``CHATTERBOX_TESTER_DRAMABOX_ROOT`` or ``DRAMABOX_REPO``
2. ``dramabox_repo_root.txt`` at the chatterbox-pipeline repository root

**Isolation (recommended):**

Set ``CHATTERBOX_TESTER_DRAMABOX_PYTHON`` to the Python executable of a venv
where you installed **only** DramaBox requirements (Torch 2.8 stack). The tester
runs a small stdin/stdout subprocess worker (`dramabox_worker_main.py`) that keeps
`TTSServer` warm; the pipeline venv never imports DramaBox.

**Legacy (single venv — can break chatterbox pins):**

Leave ``CHATTERBOX_TESTER_DRAMABOX_PYTHON`` unset to load ``TTSServer`` in-process
(same interpreter as chatterbox_tester).
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


class DramaboxTesterError(RuntimeError):
    """DramaBox is misconfigured or cannot be imported."""


_TTSServer: Optional[Any] = None

_PREPENDED_ENTRIES: List[str] = []

_MODULE_NAME_UPSTREAM = "dramabox_inference_server_upstream"

DRAMABOX_WORKER_SCRIPT = Path(__file__).resolve().parent / "dramabox_worker_main.py"

_WORKER_PROC: Optional[subprocess.Popen] = None
_WORKER_ID: Optional[Tuple[str, str, str]] = None


def isolated_dramabox_python() -> Optional[str]:
    raw = os.environ.get("CHATTERBOX_TESTER_DRAMABOX_PYTHON", "").strip()
    return raw or None


def dramabox_using_isolated_python() -> bool:
    return isolated_dramabox_python() is not None


def dramabox_repo_root_hint_file() -> Path:
    """Optional file path: chatterbox-pipeline root / dramabox_repo_root.txt."""
    return Path(__file__).resolve().parents[2] / "dramabox_repo_root.txt"


def _coerce_dramabox_root_line(raw: str) -> Path:
    """Turn env / hint file text into an absolute Path.

    Accepts Linux absolutes, ``~``, backslashes from Windows, and UNC
    ``//wsl.localhost/<Distro>/home/...`` / ``//wsl$/...`` so Explorer copies work.
    """
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


def resolve_dramabox_root() -> Optional[Path]:
    for key in ("CHATTERBOX_TESTER_DRAMABOX_ROOT", "DRAMABOX_REPO"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return _coerce_dramabox_root_line(raw)

    hint = dramabox_repo_root_hint_file()
    if hint.is_file():
        try:
            for line in hint.read_text(encoding="utf-8").splitlines():
                s = line.split("#", 1)[0].strip()
                if s:
                    return _coerce_dramabox_root_line(s)
        except OSError:
            pass
    return None


def validate_dramabox_repo(root: Path) -> None:
    if not root.is_dir():
        hp = dramabox_repo_root_hint_file()
        raise DramaboxTesterError(
            f"DramaBox root is not a directory: {root}\n"
            "Use the path inside Linux/WSL (e.g. /home/you/proj/DramaBox), not a mangled "
            f"\\\\wsl$\\\\ Explorer string. Set CHATTERBOX_TESTER_DRAMABOX_ROOT or edit:\n  {hp}"
        )
    ltx2 = root / "ltx2"
    src_dir = root / "src"
    if not ltx2.is_dir():
        raise DramaboxTesterError(
            f"DramaBox checkout missing 'ltx2/' under {root} — check clone / path."
        )
    if not src_dir.is_dir():
        raise DramaboxTesterError(
            f"DramaBox checkout missing 'src/' under {root} — check clone / path."
        )
    inf = src_dir / "inference_server.py"
    if not inf.is_file():
        raise DramaboxTesterError(
            f"DramaBox src/inference_server.py not found under {src_dir}"
        )


def _prepend_sys_path(entries: List[str]) -> None:
    for e in entries:
        if e in _PREPENDED_ENTRIES:
            continue
        sys.path.insert(0, e)
        _PREPENDED_ENTRIES.insert(0, e)


def _load_upstream_inference_server(root: Path) -> Any:
    """Exec DramaBox inference_server.py with repo ltx2+src on sys.path."""
    ltx_p = str((root / "ltx2").resolve())
    src_p = str((root / "src").resolve())
    _prepend_sys_path([src_p, ltx_p])

    server_path = root / "src" / "inference_server.py"
    spec = importlib.util.spec_from_file_location(_MODULE_NAME_UPSTREAM, server_path)
    if spec is None or spec.loader is None:
        raise DramaboxTesterError(f"Cannot load module spec from {server_path}")

    existing = sys.modules.get(_MODULE_NAME_UPSTREAM)
    if existing is not None:
        return existing.TTSServer

    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME_UPSTREAM] = module
    spec.loader.exec_module(module)
    TTSServer = getattr(module, "TTSServer", None)
    if TTSServer is None:
        raise DramaboxTesterError(f"{server_path} has no TTSServer class.")
    return TTSServer


def _device_for_ttsserver(pipeline_device: str) -> str:
    if pipeline_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        raise DramaboxTesterError(
            "DramaBox needs CUDA — no GPU available (torch.cuda.is_available() is False)."
        )
    if pipeline_device == "cpu":
        raise DramaboxTesterError(
            "DramaBox inference is CUDA-only (~24 GB VRAM); select a CUDA device."
        )
    if pipeline_device == "mps":
        raise DramaboxTesterError("DramaBox TTSServer is not supported on MPS.")
    return pipeline_device


def dispose_dramabox_worker() -> None:
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


def dispose_dramabox_ttsserver() -> None:
    """Release in-process DramaBox server and subprocess worker VRAM."""
    global _TTSServer
    dispose_dramabox_worker()
    server = _TTSServer
    _TTSServer = None
    if server is not None:
        del server
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _ensure_dramabox_worker(py_exe_raw: str, repo: Path, cuda_dev: str) -> None:
    global _WORKER_PROC, _WORKER_ID

    canon_py = str(Path(py_exe_raw).expanduser().resolve())
    canon_repo = str(repo.resolve())

    worker_script = DRAMABOX_WORKER_SCRIPT.resolve()
    if not worker_script.is_file():
        raise DramaboxTesterError(f"DramaBox worker script missing: {worker_script}")

    ident: Tuple[str, str, str] = (canon_py, canon_repo, cuda_dev)
    if (
        _WORKER_PROC is not None
        and _WORKER_PROC.poll() is None
        and _WORKER_ID == ident
    ):
        return

    dispose_dramabox_worker()

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
        dispose_dramabox_worker()
        raise DramaboxTesterError(
            "DramaBox worker missing stdio pipes (unexpected subprocess configuration)"
        )

    ready_line = out.readline()
    if proc.poll() is not None:
        stderr_tail = err_pipe.read()[-4000:]
        raise DramaboxTesterError(
            "DramaBox worker exited during startup "
            f"(exit {proc.poll()}). Is CHATTERBOX_TESTER_DRAMABOX_PYTHON pointing to "
            f"the DramaBox venv interpreter? STDERR tail:\n{stderr_tail}"
        )
    try:
        ready = json.loads(ready_line)
    except json.JSONDecodeError as e:
        stderr_tail = err_pipe.read()[-4000:]
        dispose_dramabox_worker()
        raise DramaboxTesterError(
            f"DramaBox worker invalid handshake JSON ({e}); stderr:\n{stderr_tail}"
        ) from e
    if not ready.get("ok"):
        stderr_tail = err_pipe.read()[-4000:]
        dispose_dramabox_worker()
        raise DramaboxTesterError(
            "DramaBox worker failed handshake "
            f"({ready!r}). STDERR:\n{stderr_tail}"
        )

    _WORKER_PROC = proc
    _WORKER_ID = ident


def dramabox_prepare_runtime(pipeline_device: str) -> None:
    """Warm DramaBox TTSServer (in-process or worker subprocess)."""
    root = resolve_dramabox_root()
    if root is None:
        hp = dramabox_repo_root_hint_file()
        raise DramaboxTesterError(
            "DramaBox repo path not set. Export CHATTERBOX_TESTER_DRAMABOX_ROOT (or "
            f"DRAMABOX_REPO), or put the checkout path on one line in:\n  {hp}\n"
            "See CHATTERBOX_TESTER.md."
        )
    validate_dramabox_repo(root)

    cuda_dev = _device_for_ttsserver(pipeline_device)
    iso = isolated_dramabox_python()

    if iso:
        py = iso
        if not Path(py).is_file():
            raise DramaboxTesterError(
                f"CHATTERBOX_TESTER_DRAMABOX_PYTHON is not a file: {py!r}"
            )
        print(
            f"DramaBox: isolated interpreter {py}\n"
            "     (Warm worker — first load downloads HF weights.)"
        )
        _ensure_dramabox_worker(py, root, cuda_dev)
        print("DramaBox worker ready.")
        return

    get_dramabox_ttsserver(pipeline_device)
    print(
        "DramaBox TTSServer ready in-process "
        "(not isolated — prefer CHATTERBOX_TESTER_DRAMABOX_PYTHON)."
    )


def _dramabox_request_worker_generate(req: Dict[str, Any]) -> None:
    proc = _WORKER_PROC
    if proc is None or proc.stdin is None or proc.stdout is None:
        raise DramaboxTesterError("DramaBox worker is not running.")
    if proc.poll() is not None:
        raise DramaboxTesterError("DramaBox worker exited unexpectedly.")

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
        raise DramaboxTesterError(f"DramaBox worker produced no stdout. STDERR:\n{err}")

    resp = json.loads(line)
    if not resp.get("ok"):
        err = resp.get("error", resp)
        raise DramaboxTesterError(f"DramaBox worker error: {err}")


def dramabox_generate_to_file(
    pipeline_device: str,
    *,
    prompt: str,
    output_path: str,
    voice_ref: Optional[str],
    watermark: bool,
    cfg_scale: float,
    stg_scale: float,
    duration_multiplier: float,
    seed: int,
    ref_duration: float,
    rescale_scale: Union[str, float],
    gen_duration: float,
    steps: int = 0,
) -> None:
    iso = isolated_dramabox_python()
    cuda_dev = _device_for_ttsserver(pipeline_device)
    req: Dict[str, Any] = {
        "cmd": "generate",
        "prompt": prompt,
        "output": output_path,
        "voice_ref": voice_ref,
        "watermark": watermark,
        "cfg_scale": cfg_scale,
        "stg_scale": stg_scale,
        "duration_multiplier": duration_multiplier,
        "seed": seed,
        "ref_duration": ref_duration,
        "rescale_scale": rescale_scale,
        "gen_duration": gen_duration,
    }
    if steps > 0:
        req["steps"] = steps

    if iso:
        root_opt = resolve_dramabox_root()
        if root_opt is None:
            raise DramaboxTesterError("resolve_dramabox_root returned None unexpectedly.")
        py = iso
        _ensure_dramabox_worker(py, root_opt, cuda_dev)
        _dramabox_request_worker_generate(req)
        return

    srv = get_dramabox_ttsserver(pipeline_device)
    gen_kw: Dict[str, Any] = {
        "prompt": prompt,
        "output": output_path,
        "voice_ref": voice_ref,
        "watermark": watermark,
        "cfg_scale": cfg_scale,
        "stg_scale": stg_scale,
        "duration_multiplier": duration_multiplier,
        "seed": seed,
        "ref_duration": ref_duration,
        "rescale_scale": rescale_scale,
        "gen_duration": gen_duration,
    }
    if steps > 0:
        gen_kw["steps"] = steps
    srv.generate_to_file(**gen_kw)


def get_dramabox_ttsserver(device: str) -> Any:
    """Return singleton TTSServer inside the pipeline interpreter."""
    global _TTSServer

    if isolated_dramabox_python():
        raise DramaboxTesterError(
            "In-process DramaBox TTSServer is disabled when "
            "CHATTERBOX_TESTER_DRAMABOX_PYTHON is set. Use dramabox_generate_to_file / "
            "dramabox_prepare_runtime instead."
        )

    cuda_dev = _device_for_ttsserver(device)
    root = resolve_dramabox_root()
    if root is None:
        hp = dramabox_repo_root_hint_file()
        raise DramaboxTesterError(
            "DramaBox repo path not set. Export CHATTERBOX_TESTER_DRAMABOX_ROOT (or "
            f"DRAMABOX_REPO), or put the checkout path on one line in:\n  {hp}\n"
            "See CHATTERBOX_TESTER.md."
        )
    validate_dramabox_repo(root)

    if _TTSServer is not None:
        return _TTSServer

    TTSServerCls = _load_upstream_inference_server(root)
    try:
        _TTSServer = TTSServerCls(device=cuda_dev)
    except Exception as e:
        raise DramaboxTesterError(
            f"Failed to construct DramaBox TTSServer ({e}). "
            "Install DramaBox/requirements.txt in this interpreter "
            "or switch to isolated mode (CHATTERBOX_TESTER_DRAMABOX_PYTHON). "
            "Ensure HF downloads succeed."
        ) from e

    return _TTSServer
