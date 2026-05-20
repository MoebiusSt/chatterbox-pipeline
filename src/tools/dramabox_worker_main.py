#!/usr/bin/env python3
"""
DramaBox warm worker stdin/stdout protocol (run ONLY with DramaBox venv Python).

Usage:
    python dramabox_worker_main.py /path/to/DramaBox cuda

Stdout: one JSON object per response line (line-buffered).
After loading TTSServer, prints: {"ok": true, "state": "ready"}

Stdin (one JSON request per line, UTF-8):
    {"cmd": "generate",
     "prompt": "...",
     "output": "/path/to/out.wav",
     "voice_ref": "/path.wav" | null,
     "watermark": true,
     "cfg_scale": 2.5,
     "stg_scale": 1.5,
     "duration_multiplier": 1.1,
     "seed": 42,
     "ref_duration": 10.0,
     "rescale_scale": "auto" | 0.0-1.0,
     "gen_duration": 0.0,
     "steps": 0 }

Response lines:
    {"ok": true}                    # generate finished
    {"ok": false, "error": "..."}    # traceback is written to stderr
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _configure_sys_path(repo: Path) -> None:
    ltx = repo / "ltx2"
    src = repo / "src"
    sys.path.insert(0, str(ltx.resolve()))
    sys.path.insert(0, str(src.resolve()))


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(
            json.dumps({"ok": False, "error": "usage: dramabox_worker_main.py REPO DEVICE"}),
            flush=True,
            file=sys.stderr,
        )
        return 2

    repo = Path(argv[1]).expanduser().resolve()
    device = argv[2]
    src_dir = repo / "src"
    if not src_dir.is_dir():
        print(
            json.dumps({"ok": False, "error": f"Bad DramaBox repo: {repo}"}),
            flush=True,
            file=sys.stderr,
        )
        return 3

    _configure_sys_path(repo)

    try:
        from inference_server import TTSServer
    except Exception as e:
        print(
            json.dumps({"ok": False, "error": f"import TTSServer: {e}"}),
            flush=True,
            file=sys.stderr,
        )
        return 4

    try:
        server = TTSServer(
            device=device,
            compile_model=_env_bool("DRAMABOX_TORCH_COMPILE", False),
            phase_offload=_env_bool("DRAMABOX_PHASE_OFFLOAD", True),
        )
    except Exception as e:
        print(
            json.dumps({"ok": False, "error": f"TTSServer ctor: {e}"}),
            flush=True,
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        return 5

    print(json.dumps({"ok": True, "state": "ready"}), flush=True)

    stdin = sys.stdin
    stdout = sys.stdout
    while True:
        line = stdin.readline()
        if not line:
            break
        stripped = line.strip()
        if not stripped:
            continue
        try:
            req = json.loads(stripped)
        except json.JSONDecodeError as e:
            print(
                json.dumps({"ok": False, "error": f"invalid json: {e}"}),
                flush=True,
            )
            continue
        if req.get("cmd") != "generate":
            print(
                json.dumps({"ok": False, "error": f'unknown cmd: {req.get("cmd")}'}),
                flush=True,
            )
            continue

        outp = Path(str(req["output"]))
        outp.parent.mkdir(parents=True, exist_ok=True)
        try:
            vc = req.get("voice_ref")
            voice_ref: str | None = None if vc in (None, "") else str(vc)
            rs_raw = req.get("rescale_scale", "auto")
            rescale_kw: float | str
            if rs_raw == "auto" or (isinstance(rs_raw, str) and rs_raw.lower() == "auto"):
                rescale_kw = "auto"
            else:
                rescale_kw = float(rs_raw)

            steps_raw = int(req.get("steps", 0))
            gen_kw: dict = {
                "prompt": str(req["prompt"]),
                "output": str(outp),
                "voice_ref": voice_ref,
                "watermark": bool(req.get("watermark", True)),
                "cfg_scale": float(req["cfg_scale"]),
                "stg_scale": float(req["stg_scale"]),
                "duration_multiplier": float(req["duration_multiplier"]),
                "seed": int(req["seed"]),
                "ref_duration": float(req["ref_duration"]),
                "rescale_scale": rescale_kw,
                "gen_duration": float(req.get("gen_duration", 0.0)),
            }
            if steps_raw > 0:
                gen_kw["steps"] = steps_raw
            server.generate_to_file(**gen_kw)
            print(json.dumps({"ok": True}), flush=True)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            print(
                json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
