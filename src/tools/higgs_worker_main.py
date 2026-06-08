#!/usr/bin/env python3
"""
Higgs Audio V2 warm worker stdin/stdout protocol (run ONLY with .venv-higgs Python).

Usage:
    python higgs_worker_main.py /path/to/higgs-audio cuda

Stdout: one JSON object per response line (line-buffered).
After loading the model, prints: {"ok": true, "state": "ready"}

Stdin (one JSON request per line, UTF-8):
    {"cmd": "generate",
     "transcript": "...",
     "output": "/path/to/out.wav",
     "scene_prompt": "..." | null,
     "ref_audio_path": "/path/to/ref.wav" | null,
     "profile_text": "..." | null,
     "temperature": 0.7,
     "top_p": 0.95,
     "top_k": 50,
     "seed": 42}

Response lines:
    {"ok": true}
    {"ok": false, "error": "..."}
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

# Token budget for tester runs (not exposed as a GUI slider).
HIGGS_MAX_NEW_TOKENS = 1024
HIGGS_CHUNK_MAX_WORD_NUM = 100
HIGGS_GENERATION_CHUNK_BUFFER_SIZE = 2


def _configure_repo_path(repo: Path) -> None:
    root = str(repo.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _normalize_transcript(transcript: str) -> str:
    """Match examples/generation.py pre-generation text cleanup."""
    from examples.generation import normalize_chinese_punctuation

    transcript = normalize_chinese_punctuation(transcript)
    transcript = transcript.replace("(", " ")
    transcript = transcript.replace(")", " ")
    transcript = transcript.replace("°F", " degrees Fahrenheit")
    transcript = transcript.replace("°C", " degrees Celsius")

    for tag, replacement in [
        ("[laugh]", "<SE>[Laughter]</SE>"),
        ("[humming start]", "<SE_s>[Humming]</SE_s>"),
        ("[humming end]", "<SE_e>[Humming]</SE_e>"),
        ("[music start]", "<SE_s>[Music]</SE_s>"),
        ("[music end]", "<SE_e>[Music]</SE_e>"),
        ("[music]", "<SE>[Music]</SE>"),
        ("[sing start]", "<SE_s>[Singing]</SE_s>"),
        ("[sing end]", "<SE_e>[Singing]</SE_e>"),
        ("[applause]", "<SE>[Applause]</SE>"),
        ("[cheering]", "<SE>[Cheering]</SE>"),
        ("[cough]", "<SE>[Cough]</SE>"),
    ]:
        transcript = transcript.replace(tag, replacement)

    lines = transcript.split("\n")
    transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
    transcript = transcript.strip()

    if not any(
        transcript.endswith(c)
        for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]
    ):
        transcript += "."
    return transcript


def _prepare_tester_context(
    scene_prompt: Optional[str],
    ref_audio_path: Optional[str],
    profile_text: Optional[str],
    audio_tokenizer,
) -> Tuple[list, list]:
    """Build ChatML messages for single-speaker tester (no [SPEAKERn] tags)."""
    from boson_multimodal.data_types import AudioContent, Message

    messages: list = []
    audio_ids: list = []

    scene_desc_parts: List[str] = []
    if scene_prompt:
        scene_desc_parts.append(scene_prompt.strip())
    if profile_text:
        scene_desc_parts.append(f"SPEAKER0: {profile_text.strip()}")

    if scene_desc_parts:
        scene_block = "\n\n".join(scene_desc_parts)
        system_content = (
            "Generate audio following instruction.\n\n"
            f"<|scene_desc_start|>\n{scene_block}\n<|scene_desc_end|>"
        )
    else:
        system_content = "Generate audio following instruction."

    messages.append(Message(role="system", content=system_content))

    if ref_audio_path:
        wav = Path(ref_audio_path)
        txt = wav.with_suffix(".txt")
        if not wav.is_file():
            raise FileNotFoundError(f"Reference WAV not found: {wav}")
        if not txt.is_file():
            raise FileNotFoundError(
                f"Missing sidecar transcript for voice clone: {txt}"
            )
        prompt_text = txt.read_text(encoding="utf-8").strip()
        audio_tokens = audio_tokenizer.encode(str(wav))
        audio_ids.append(audio_tokens)
        messages.append(Message(role="user", content=prompt_text))
        messages.append(
            Message(
                role="assistant",
                content=AudioContent(audio_url=str(wav)),
            )
        )

    return messages, audio_ids


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(
            json.dumps({"ok": False, "error": "usage: higgs_worker_main.py REPO DEVICE"}),
            flush=True,
            file=sys.stderr,
        )
        return 2

    repo = Path(argv[1]).expanduser().resolve()
    device = argv[2]

    if not (repo / "boson_multimodal").is_dir():
        print(
            json.dumps({"ok": False, "error": f"Bad Higgs repo: {repo}"}),
            flush=True,
            file=sys.stderr,
        )
        return 3

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    _configure_repo_path(repo)

    try:
        import soundfile as sf
        from examples.generation import HiggsAudioModelClient, prepare_chunk_text
    except Exception as e:
        print(
            json.dumps({"ok": False, "error": f"import Higgs modules: {e}"}),
            flush=True,
            file=sys.stderr,
        )
        return 4

    cuda_dev = device if device.startswith("cuda") else "cuda:0"
    try:
        client = HiggsAudioModelClient(
            model_path="bosonai/higgs-audio-v2-generation-3B-base",
            audio_tokenizer="bosonai/higgs-audio-v2-tokenizer",
            device=cuda_dev,
            device_id=0 if cuda_dev.startswith("cuda") else None,
            max_new_tokens=HIGGS_MAX_NEW_TOKENS,
            use_static_kv_cache=0,
            kv_cache_lengths=[2048, 4096],
        )
    except Exception as e:
        print(
            json.dumps({"ok": False, "error": f"HiggsAudioModelClient ctor: {e}"}),
            flush=True,
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        return 5

    print(json.dumps({"ok": True, "state": "ready"}), flush=True)

    stdin = sys.stdin
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
            transcript = str(req["transcript"]).strip()
            if not transcript:
                raise ValueError("transcript is empty")

            scene_raw = req.get("scene_prompt")
            scene_prompt: Optional[str] = None
            if scene_raw not in (None, ""):
                scene_prompt = str(scene_raw).strip() or None

            ref_raw = req.get("ref_audio_path")
            ref_audio_path: Optional[str] = None
            if ref_raw not in (None, ""):
                ref_audio_path = str(ref_raw)

            profile_raw = req.get("profile_text")
            profile_text: Optional[str] = None
            if profile_raw not in (None, ""):
                profile_text = str(profile_raw).strip() or None

            temperature = float(req.get("temperature", 0.7))
            top_p = float(req.get("top_p", 0.95))
            top_k_raw = int(req.get("top_k", 50))
            top_k = top_k_raw if top_k_raw > 0 else None
            seed_raw = req.get("seed")
            seed = int(seed_raw) if seed_raw not in (None, "", 0) else None

            transcript = _normalize_transcript(transcript)
            pattern = re.compile(r"\[(SPEAKER\d+)\]")
            if pattern.findall(transcript):
                raise ValueError(
                    "Multi-speaker [SPEAKERn] transcripts are not supported in the tester."
                )

            messages, audio_ids = _prepare_tester_context(
                scene_prompt=scene_prompt,
                ref_audio_path=ref_audio_path,
                profile_text=profile_text,
                audio_tokenizer=client._audio_tokenizer,
            )

            chunked_text = prepare_chunk_text(
                transcript,
                chunk_method="word",
                chunk_max_word_num=HIGGS_CHUNK_MAX_WORD_NUM,
            )

            concat_wv, sr, _text_out = client.generate(
                messages=messages,
                audio_ids=audio_ids,
                chunked_text=chunked_text,
                generation_chunk_buffer_size=HIGGS_GENERATION_CHUNK_BUFFER_SIZE,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
            )
            if concat_wv is None:
                raise RuntimeError("Higgs model produced no audio output.")

            sf.write(str(outp), concat_wv, sr)
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
