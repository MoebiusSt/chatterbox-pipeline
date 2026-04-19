"""
HF-generate streamer for live Qwen3 talker token progress (console).

Used by chatterbox_tester and TTSGenerator when running Qwen3 voice clone.
"""

import sys
import time
from typing import Any


class Qwen3ProgressStreamer:
    """Minimal HF-generate streamer that prints a live token counter.

    Implements the BaseStreamer protocol (``put``/``end``) duck-typed, so it
    can be injected into ``talker.generate`` without importing transformers.
    The counter is approximate – HF may call ``put`` with batched tokens and
    the first call on ``inputs_embeds``-style generation can include the
    implicit prompt step – but it is enough to show that generation is
    progressing and roughly how fast.
    """

    def __init__(self, total: int = 2048, label: str = "Qwen3"):
        self.total = max(int(total), 1)
        self.label = label
        self.count = 0
        self._t0 = time.perf_counter()
        self._first = True

    def put(self, value: Any) -> None:
        try:
            n = int(value.shape[-1]) if hasattr(value, "shape") else 1
        except Exception:
            n = 1
        if self._first:
            # HF's first put() contains the prompt tokens; don't let it skew tps.
            self._first = False
            self._t0 = time.perf_counter()
            return
        self.count += n
        elapsed = time.perf_counter() - self._t0
        tps = self.count / max(elapsed, 1e-6)
        bar_w = 24
        filled = min(bar_w, int(self.count / self.total * bar_w))
        bar = "█" * filled + "░" * (bar_w - filled)
        sys.stdout.write(
            f"\r🎙️ {self.label} [{bar}] {self.count}/{self.total} tok "
            f"| {tps:4.1f} t/s | {elapsed:5.1f}s"
        )
        sys.stdout.flush()

    def end(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()
