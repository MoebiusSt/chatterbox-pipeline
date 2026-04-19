#!/usr/bin/env python3
"""
Desktop window: list WAV, MP3, and TXT files in a chosen directory; transcribe
selected **audio** files with OpenAI Whisper **small** (same cache as the cbpipe
pipeline: ``~/.cache/whisper/small.pt``). Writes ``<same-basename>.txt`` next to
each audio file (UTF-8; overwrites existing ``.txt``). ``.txt`` rows are listed
for reference but are not passed to Whisper. Double-click a ``.txt`` row to open it with the default application (on Windows /
WSL this is typically Notepad). Selecting a single ``.txt`` row shows its contents in
the read-only **Text preview** panel; the **Transcription log** panel is shown for
other selections and when new log lines are written.

The watched directory is scanned with **watchdog** ``PollingObserver`` every
500 ms so new files appear without manual refresh. An optional **filter** field
narrows the list to filenames containing the typed substring (case-insensitive).

Requires:
    - openai-whisper (project requirements.txt)
    - watchdog

Usage:
    python scripts/transcribe_drop.py
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, scrolledtext

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers.polling import PollingObserver
except ImportError as e:
    print(
        "ERROR: watchdog is required.\n" "  pip install watchdog",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

# Shown in the list (directory scan + filter).
LISTED_EXTENSIONS = {".wav", ".mp3", ".txt"}
# Transcription only for these.
AUDIO_EXTENSIONS = {".wav", ".mp3"}
MAX_DURATION_S = 90.0
MODEL_NAME = "small"
POLL_INTERVAL_S = 0.5
DEBOUNCE_MS = 75
# Reset multi-letter prefix if idle longer than this (file-browser style type-ahead).
TYPEAHEAD_RESET_S = 0.85

# Default: project data/reference_audio (same as Windows \\wsl.localhost\...\reference_audio)
_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data/input/reference_audio"


def _resolve_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_whisper_model():
    import torch
    import whisper

    device = _resolve_device()
    fp16 = device == "cuda"
    model = whisper.load_model(MODEL_NAME, device=device)
    return model, device, fp16


def _audio_duration_s(audio_path: Path) -> float:
    import whisper

    audio = whisper.load_audio(str(audio_path))
    return float(len(audio)) / 16000.0


def _transcribe_to_txt(model, audio_path: Path, fp16: bool) -> tuple[str, float]:
    """Returns (transcription text, wall-clock seconds)."""
    import whisper

    t0 = time.time()
    result = model.transcribe(
        str(audio_path),
        language=None,
        task="transcribe",
        fp16=fp16,
        verbose=False,
        condition_on_previous_text=False,
    )
    text = str(result.get("text", "")).strip()
    elapsed = time.time() - t0
    return text, elapsed


def _write_txt(audio_path: Path, text: str) -> Path:
    out = audio_path.with_suffix(".txt")
    out.write_text(text + "\n", encoding="utf-8")
    return out


def _open_file_with_default_app(path: Path) -> None:
    """Open a file with the OS-registered default handler (Explorer / Notepad on Windows)."""
    p = path.resolve()
    if not p.is_file():
        raise FileNotFoundError(p)

    if sys.platform == "win32":
        os.startfile(str(p))  # type: ignore[attr-defined]
        return

    if sys.platform == "darwin":
        subprocess.Popen(["open", str(p)], start_new_session=True)
        return

    # Linux — prefer WSL → Windows default app when running under WSL
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        wslview = shutil.which("wslview")
        if wslview:
            subprocess.Popen([wslview, str(p)], start_new_session=True)
            return
        try:
            win_path = subprocess.check_output(
                ["wslpath", "-w", str(p)], text=True, timeout=15
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            win_path = ""
        if win_path:
            cmd = shutil.which("cmd.exe") or "/mnt/c/Windows/System32/cmd.exe"
            if os.path.isfile(cmd):
                subprocess.Popen(
                    [cmd, "/c", "start", "", win_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return

    xdg = shutil.which("xdg-open")
    if xdg:
        subprocess.Popen([xdg, str(p)], start_new_session=True)
        return

    raise RuntimeError("Could not find a program to open the file (xdg-open / WSL).")


class _WatchHandler(FileSystemEventHandler):
    """Marshals filesystem events to the Tk main thread."""

    def __init__(self, app: "TranscribeApp") -> None:
        self._app = app

    def on_any_event(self, event) -> None:
        # Directory-only events still matter when files are added; also catch moved/closed writes
        self._app.root.after(0, self._app._schedule_refresh_debounced)


class TranscribeApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Whisper small — reference audio")
        self.root.minsize(560, 480)

        self._watch_dir = _DEFAULT_DIR.resolve()
        self._observer: PollingObserver | None = None
        self._debounce_after: str | None = None

        self._model = None
        self._fp16 = False
        self._device = "?"
        self._work_q: queue.Queue[tuple[Path,]] = queue.Queue()
        self._worker_stop = threading.Event()

        self._all_file_paths: list[Path] = []
        self._file_paths: list[Path] = []

        self._typeahead_buffer = ""
        self._typeahead_last_mono = 0.0

        self._bottom_mode: str = "log"

        self._build_ui()
        self._start_model_loader()
        self._start_worker()
        self._set_watch_dir(self._watch_dir, quiet=True)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        pad = {"padx": 12, "pady": 6}

        top = tk.Frame(self.root)
        top.pack(fill="x", **pad)

        tk.Button(top, text="Change directory…", command=self._browse_directory).pack(
            side="left"
        )

        self._dir_label = tk.Label(
            top,
            text="",
            anchor="w",
            justify="left",
            wraplength=520,
        )
        self._dir_label.pack(side="left", fill="x", expand=True, padx=(10, 0))

        self.status = tk.Label(
            self.root,
            text="Loading Whisper model…",
            anchor="w",
            justify="left",
        )
        self.status.pack(fill="x", padx=12, pady=(0, 4))

        hint = (
            f"Lists {', '.join(sorted(LISTED_EXTENSIONS))} in the folder. "
            f"Transcribe runs on {', '.join(sorted(AUDIO_EXTENSIONS))} only (max ~{int(MAX_DURATION_S)} s). "
            "Output: .txt next to the audio file. Double-click .txt to open in default app; "
            "single-select .txt to preview text below."
        )
        tk.Label(self.root, text=hint, wraplength=520, justify="left").pack(
            fill="x", padx=12, pady=(0, 4)
        )

        list_frame = tk.LabelFrame(
            self.root, text="Files — wav / mp3 / txt (auto-refresh via watchdog, 500 ms)"
        )
        list_frame.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        filter_row = tk.Frame(list_frame)
        filter_row.pack(fill="x", padx=6, pady=(6, 4))
        tk.Label(filter_row, text="Filter (contains):").pack(side="left")
        self._filter_var = tk.StringVar()
        tk.Entry(filter_row, textvariable=self._filter_var).pack(
            side="left", fill="x", expand=True, padx=(8, 0)
        )
        self._filter_var.trace_add("write", lambda *_: self._apply_name_filter())

        scroll = tk.Scrollbar(list_frame)
        scroll.pack(side="right", fill="y")

        self._listbox = tk.Listbox(
            list_frame,
            height=22,
            selectmode=tk.EXTENDED,
            yscrollcommand=scroll.set,
            font=("TkFixedFont", 10),
        )
        self._listbox.pack(side="left", fill="both", expand=True)
        scroll.config(command=self._listbox.yview)
        self._listbox.configure(takefocus=True)
        self._listbox.bind("<Key>", self._on_listbox_key)
        self._listbox.bind("<Double-Button-1>", self._on_listbox_double_click)
        self._listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        btn_row = tk.Frame(self.root)
        btn_row.pack(fill="x", padx=12, pady=(0, 8))
        tk.Button(btn_row, text="Transcribe selected", command=self._transcribe_selected).pack(
            side="left"
        )

        bottom_area = tk.Frame(self.root)
        bottom_area.pack(fill="both", expand=False, padx=12, pady=(0, 12))

        self._log_panel = tk.LabelFrame(bottom_area, text="Transcription log")
        self.log = scrolledtext.ScrolledText(self._log_panel, height=8, state="disabled")
        self.log.pack(fill="both", expand=True, padx=4, pady=4)

        self._preview_panel = tk.LabelFrame(bottom_area, text="Text preview")
        self._preview = scrolledtext.ScrolledText(
            self._preview_panel,
            height=8,
            state="disabled",
            wrap=tk.WORD,
            font=("TkFixedFont", 10),
        )
        self._preview.pack(fill="both", expand=True, padx=4, pady=4)

        self._log_panel.pack(fill="both", expand=True)

    def _show_bottom_log(self) -> None:
        if self._bottom_mode == "log":
            return
        self._preview_panel.pack_forget()
        self._log_panel.pack(fill="both", expand=True)
        self._bottom_mode = "log"

    def _show_bottom_preview(self) -> None:
        if self._bottom_mode == "preview":
            return
        self._log_panel.pack_forget()
        self._preview_panel.pack(fill="both", expand=True)
        self._bottom_mode = "preview"

    def _load_txt_preview(self, path: Path) -> None:
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            body = f"(Could not read file: {e})"
        self._preview_panel.config(text=f"Text preview — {path.name}")
        self._preview.configure(state="normal")
        self._preview.delete("1.0", tk.END)
        self._preview.insert("1.0", body)
        self._preview.configure(state="disabled")

    def _on_listbox_select(self, _event: object = None) -> None:
        sel = self._listbox.curselection()
        if len(sel) == 1:
            idx = sel[0]
            if 0 <= idx < len(self._file_paths):
                path = self._file_paths[idx]
                if path.suffix.lower() == ".txt":
                    self._load_txt_preview(path)
                    self._show_bottom_preview()
                    return
        self._show_bottom_log()

    def _log_line(self, msg: str) -> None:
        def _do() -> None:
            self._show_bottom_log()
            self.log.configure(state="normal")
            self.log.insert("end", msg + "\n")
            self.log.see("end")
            self.log.configure(state="disabled")

        self.root.after(0, _do)

    def _set_status(self, text: str) -> None:
        self.root.after(0, lambda: self.status.config(text=text))

    def _browse_directory(self) -> None:
        initial = str(self._watch_dir) if self._watch_dir.is_dir() else str(Path.home())
        picked = filedialog.askdirectory(initialdir=initial, parent=self.root, mustexist=True)
        if picked:
            self._set_watch_dir(Path(picked))

    def _set_watch_dir(self, path: Path, quiet: bool = False) -> None:
        path = path.expanduser().resolve()
        if not path.is_dir():
            self._set_status(f"Not a directory: {path}")
            if not quiet:
                self._log_line(f"ERROR: not a directory: {path}")
            return

        self._stop_observer()
        self._watch_dir = path
        self._dir_label.config(text=str(path))
        self._start_observer()
        self._refresh_file_list()
        if not quiet:
            self._log_line(f"Watching: {path}")

    def _start_observer(self) -> None:
        self._observer = PollingObserver(timeout=POLL_INTERVAL_S)
        handler = _WatchHandler(self)
        self._observer.schedule(handler, str(self._watch_dir), recursive=False)
        self._observer.start()

    def _stop_observer(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            try:
                self._observer.join(timeout=5.0)
            except Exception:
                pass
            self._observer = None

    def _schedule_refresh_debounced(self) -> None:
        if self._debounce_after is not None:
            self.root.after_cancel(self._debounce_after)
        self._debounce_after = self.root.after(DEBOUNCE_MS, self._debounced_refresh)

    def _debounced_refresh(self) -> None:
        self._debounce_after = None
        self._refresh_file_list()

    def _apply_name_filter(self) -> None:
        """Show only files whose basename contains the filter substring (case-insensitive)."""
        self._typeahead_buffer = ""
        self._typeahead_last_mono = 0.0

        selected_names = {self._listbox.get(i) for i in self._listbox.curselection()}

        self._listbox.delete(0, tk.END)
        self._file_paths.clear()

        needle = self._filter_var.get().casefold()
        if needle:
            filtered = [p for p in self._all_file_paths if needle in p.name.casefold()]
        else:
            filtered = list(self._all_file_paths)

        for p in filtered:
            self._file_paths.append(p)
            self._listbox.insert(tk.END, p.name)

        for i, p in enumerate(self._file_paths):
            if p.name in selected_names:
                self._listbox.selection_set(i)

        self._on_listbox_select()

    def _refresh_file_list(self) -> None:
        root = self._watch_dir
        self._all_file_paths.clear()
        if not root.is_dir():
            self._apply_name_filter()
            return

        try:
            self._all_file_paths.extend(
                sorted(
                    p
                    for p in root.iterdir()
                    if p.is_file() and p.suffix.lower() in LISTED_EXTENSIONS
                )
            )
        except OSError as e:
            self._log_line(f"ERROR listing directory: {e}")
            self._apply_name_filter()
            return

        self._apply_name_filter()

    def _apply_typeahead_selection(self, idx: int) -> None:
        self._listbox.selection_clear(0, tk.END)
        self._listbox.selection_set(idx)
        self._listbox.activate(idx)
        self._listbox.see(idx)

    @staticmethod
    def _first_prefix_index(names: list[str], prefix: str) -> int | None:
        pf = prefix.casefold()
        for i, name in enumerate(names):
            if name.casefold().startswith(pf):
                return i
        return None

    def _typeahead_cycle_same_letter(self, names: list[str], ch: str) -> None:
        """Jump to next list row that starts with the same single character (Explorer-style)."""
        pf = ch.casefold()
        matches = [i for i, name in enumerate(names) if name.casefold().startswith(pf)]
        if not matches:
            return
        sel = self._listbox.curselection()
        cur = sel[0] if sel else -1
        nxt = None
        for m in matches:
            if m > cur:
                nxt = m
                break
        if nxt is None:
            nxt = matches[0]
        self._apply_typeahead_selection(nxt)

    def _on_listbox_key(self, event: tk.Event) -> str | None:
        """Prefix type-ahead and repeat-letter cycling; keep arrow keys etc. default."""
        if event.keysym in (
            "Up",
            "Down",
            "Left",
            "Right",
            "Prior",
            "Next",
            "Home",
            "End",
            "Tab",
            "Escape",
            "Return",
        ):
            return None
        if event.state & 0x4:
            return None

        ch = event.char
        if not ch:
            return None
        if ord(ch) < 32 and ch != " ":
            return None

        n = self._listbox.size()
        if n == 0:
            return "break"

        names = [self._listbox.get(i) for i in range(n)]

        now = time.monotonic()
        expired = (now - self._typeahead_last_mono) > TYPEAHEAD_RESET_S
        self._typeahead_last_mono = now

        if expired:
            self._typeahead_buffer = ""

        # Second press of the same letter: next matching row (without growing the prefix).
        if (
            not expired
            and len(self._typeahead_buffer) == 1
            and self._typeahead_buffer.casefold() == ch.casefold()
        ):
            self._typeahead_cycle_same_letter(names, ch)
            return "break"

        if not self._typeahead_buffer:
            self._typeahead_buffer = ch
        else:
            self._typeahead_buffer += ch

        prefix = self._typeahead_buffer
        idx = self._first_prefix_index(names, prefix)
        if idx is not None:
            self._apply_typeahead_selection(idx)
            return "break"

        # No row for full prefix: retry starting with this keystroke only.
        self._typeahead_buffer = ch
        idx = self._first_prefix_index(names, ch)
        if idx is not None:
            self._apply_typeahead_selection(idx)
        else:
            self._typeahead_buffer = ""
        return "break"

    def _on_listbox_double_click(self, event: tk.Event) -> None:
        idx = self._listbox.nearest(event.y)
        if idx < 0 or idx >= len(self._file_paths):
            return
        path = self._file_paths[idx]
        if path.suffix.lower() != ".txt":
            return
        try:
            _open_file_with_default_app(path)
        except Exception as e:
            self._log_line(f"ERROR opening {path.name}: {e}")

    def _transcribe_selected(self) -> None:
        if self._model is None:
            self._log_line("Model not ready yet.")
            return
        indices = self._listbox.curselection()
        if not indices:
            self._log_line("No file selected.")
            return
        for i in indices:
            if 0 <= i < len(self._file_paths):
                self._work_q.put((self._file_paths[i],))

    def _start_model_loader(self) -> None:
        def load() -> None:
            try:
                model, device, fp16 = _load_whisper_model()
                self._model = model
                self._fp16 = fp16
                self._device = device
                self._set_status(
                    f"Ready — model '{MODEL_NAME}' on {device} (openai-whisper cache)"
                )
                self._log_line(
                    f"Model loaded on {device}. Select file(s), then Transcribe (max {int(MAX_DURATION_S)} s)."
                )
            except Exception as e:
                self._set_status("Failed to load model (see log)")
                self._log_line(f"ERROR loading model: {e}")

        threading.Thread(target=load, daemon=True).start()

    def _start_worker(self) -> None:
        def worker() -> None:
            while not self._worker_stop.is_set():
                try:
                    item = self._work_q.get(timeout=0.3)
                except queue.Empty:
                    continue
                (path,) = item
                self._process_path(path)
                self._work_q.task_done()

        threading.Thread(target=worker, daemon=True).start()

    def _process_path(self, path: Path) -> None:
        if self._model is None:
            self._log_line(f"Skipped (no model): {path}")
            return
        ext = path.suffix.lower()
        if ext not in AUDIO_EXTENSIONS:
            if ext == ".txt":
                self._log_line(f"Skipped (listing only, not transcribed): {path.name}")
            else:
                self._log_line(f"Skipped (not audio {AUDIO_EXTENSIONS}): {path}")
            return
        if not path.is_file():
            self._log_line(f"Skipped (not a file): {path}")
            return

        try:
            dur = _audio_duration_s(path)
        except Exception as e:
            self._log_line(f"ERROR reading audio {path.name}: {e}")
            return

        if dur > MAX_DURATION_S:
            self._log_line(
                f"Skipped (duration {dur:.1f}s > {int(MAX_DURATION_S)}s): {path}"
            )
            return

        self._log_line(f"Transcribing ({dur:.1f}s): {path.name} …")
        try:
            text, elapsed = _transcribe_to_txt(self._model, path, self._fp16)
            out = _write_txt(path, text)
            preview = (text[:120] + "…") if len(text) > 120 else text
            self._log_line(f"  → {out.name} ({elapsed:.1f}s)\n  {preview}")
            self.root.after(0, self._refresh_file_list)
        except Exception as e:
            self._log_line(f"ERROR transcribing {path.name}: {e}")

    def _on_close(self) -> None:
        self._worker_stop.set()
        if self._debounce_after is not None:
            try:
                self.root.after_cancel(self._debounce_after)
            except Exception:
                pass
        self._stop_observer()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    TranscribeApp().run()


if __name__ == "__main__":
    main()
