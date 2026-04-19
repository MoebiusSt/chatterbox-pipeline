#!/usr/bin/env python3
"""
Chatterbox TTS Parameter Testing Tool (pygame version)

A lightweight GUI tool for quickly testing different Chatterbox / Qwen3 parameters,
models, and reference audio files to find optimal voice settings.

Supported models:
  classic        ChatterboxTTS (English only)
  multilanguage  ChatterboxMultilingualTTS (24 languages)
  turbo          ChatterboxTurboTTS (English only, adds top_k, paralinguistic tags)
  qwen3          Qwen3-TTS-12Hz-1.7B-Base voice cloning (10 languages, needs ref_audio)
  vibevoice      VibeVoice-Large-Q8 (long-form with voice cloning)
  vibevoice_1_5b VibeVoice-1.5B (reference model)
  vibevoice_q4   VibeVoice-Large-Q4 (DevParker low-VRAM quant)

GUI text field: Chatterbox models are capped at 500 characters; qwen3 and all VibeVoice
variants use 10000 so long-form input can be pasted (same cap as VibeVoice).

Environment (VibeVoice only):
  CHATTERBOX_TESTER_VIBEVOICE_ATTN  Override attention backend for the Qwen2 LM (all VV models).
    If unset, per-model defaults are used:
      vibevoice (7B Q8) → sdpa (best performance)            
      vibevoice_1_5b    → sdpa (best performance)              
      vibevoice_q4      → sdpa (best performance)             
    Allowed values: auto, flash_attention_2, sdpa, eager.

This version uses pygame for audio playback (better WSL compatibility).
"""

import tkinter as tk
from tkinter import ttk
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading
import time
import numpy as np
import soundfile as sf
import torch
import tempfile
import subprocess

# Initialize pygame mixer early (disabled for WSL2 compatibility)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import pygame.mixer as mixer
# Disable pygame mixer initialization for WSL2 - use ffplay instead
# mixer.init(frequency=24000, size=-16, channels=1, buffer=2048)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from generation.model_cache import ChatterboxModelCache
from utils.qwen3_progress_streamer import Qwen3ProgressStreamer as _Qwen3ProgressStreamer


def _maybe_suppress_transformers_console_noise() -> None:
    """Reduce HuggingFace tokenizer/model repr noise (e.g. added_tokens_decoder dumps). Re-enable with
    CHATTERBOX_TESTER_VERBOSE_TRANSFORMERS=1 for debugging."""
    if os.environ.get("CHATTERBOX_TESTER_VERBOSE_TRANSFORMERS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    import logging

    for name in (
        "transformers",
        "transformers.tokenization_utils_base",
        "transformers.modeling_utils",
        "transformers.configuration_utils",
        "huggingface_hub",
        "huggingface_hub.file_download",
        "tokenizers",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        from transformers import logging as tr_logging

        tr_logging.set_verbosity_error()
    except Exception:
        pass


_maybe_suppress_transformers_console_noise()


# ---------------------------------------------------------------------------
# Slider metadata: which models each slider is active for, and integer flag
# ---------------------------------------------------------------------------
_SLIDER_CONFIG = [
    # (name,                 min,   max,    default, res,   models,                                   integer)
    # --- classic/multilanguage/turbo-only params (hidden for qwen3) ---
    ("exaggeration",         0.0,  2.0,    0.70,   0.01,  ["classic", "multilanguage", "turbo"],    False),
    ("cfg_weight",           0.0,  1.5,    0.45,   0.01,  ["classic", "multilanguage", "turbo"],    False),
    ("min_p",                0.01, 0.5,    0.05,   0.01,  ["classic", "multilanguage", "turbo"],    False),
    # --- shared + qwen3 params (qwen3 order: temp, top_k, top_p, sub_temp, sub_k, sub_p, rep) ---
    # VibeVoice: temperature/top_p are applied BEFORE the 5-token structural constraint.
    # They affect relative probabilities of speech_diffusion vs speech_end at segment boundaries
    # (= segment length / rhythm variation). Effect is subtle; primary controls are cfg_scale,
    # diffusion_steps, and seed. Only active when use_sampling=True (do_sample=True).
    ("temperature",          0.05, 2.5,    0.80,   0.01,  ["classic", "multilanguage", "turbo",
                                                            "qwen3", "vibevoice", "vibevoice_1_5b", "vibevoice_q4"], False),
    # top_k range/default reconfigured per model in _update_model_ui()
    ("top_k",                1.0,  2000.0, 1000.0, 1.0,   ["turbo", "qwen3"],                      True),
    ("top_p",                0.5,  1.0,    0.98,   0.01,  ["classic", "multilanguage", "turbo",
                                                            "qwen3", "vibevoice", "vibevoice_1_5b", "vibevoice_q4"], False),
    # subtalker: acoustic sub-model (Qwen3 residual VQ codes 2..Q)
    ("subtalker_temperature", 0.05, 2.5,   0.9,    0.01,  ["qwen3"],                               False),
    ("subtalker_top_k",      1.0,  400.0,  50.0,   1.0,   ["qwen3"],                               True),
    ("subtalker_top_p",      0.5,  1.0,    1.0,    0.01,  ["qwen3"],                               False),
    ("repetition_penalty",   1.0,  3.0,    2.20,   0.01,  ["classic", "multilanguage", "turbo",
                                                            "qwen3"],                                False),
    ("cfg_scale",            1.0,  2.0,    1.30,   0.05,  ["vibevoice", "vibevoice_1_5b", "vibevoice_q4"], False),
    ("diffusion_steps",      5.0,  100.0,  20.0,   1.0,   ["vibevoice", "vibevoice_1_5b", "vibevoice_q4"], True),
    ("voice_speed_factor",   0.8,  1.2,    1.00,   0.01,  ["vibevoice", "vibevoice_1_5b", "vibevoice_q4"], False),
]

_SLIDER_MODELS: Dict[str, list] = {row[0]: row[5] for row in _SLIDER_CONFIG}
_VIBEVOICE_UI_MODELS = {"vibevoice", "vibevoice_1_5b", "vibevoice_q4"}

# Tester-only UI limit (characters). Chatterbox stays short; long-form models match VibeVoice.
_LONGFORM_TEXT_UI_LIMIT = 10000
_LONGFORM_TEXT_UI_MODELS = _VIBEVOICE_UI_MODELS | {"qwen3"}
# Optional global override. Empty string → per-model defaults in model_cache.py take effect.
_VIBEVOICE_TESTER_ATTN: str = os.environ.get(
    "CHATTERBOX_TESTER_VIBEVOICE_ATTN", ""
).strip().lower()
if _VIBEVOICE_TESTER_ATTN not in ("auto", "flash_attention_2", "sdpa", "eager"):
    _VIBEVOICE_TESTER_ATTN = ""

_VIBEVOICE_BRACKET_LINE = re.compile(r"^\s*\[(\d+)\]\s*:\s*(.*)$")


def _vibevoice_text_has_bracket_speaker_lines(text: str) -> bool:
    for line in text.splitlines():
        if line.strip() and _VIBEVOICE_BRACKET_LINE.match(line):
            return True
    return False


def _vibevoice_validate_multispeaker_bracket_text(text: str) -> int:
    """Require every non-empty line to be [N]: ...; return K = max speaker id."""
    ids: List[int] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _VIBEVOICE_BRACKET_LINE.match(line)
        if not m:
            raise RuntimeError(
                "VibeVoice multi-speaker: every non-empty line must start with [N]: ..."
            )
        ids.append(int(m.group(1)))
    if not ids:
        raise RuntimeError("VibeVoice multi-speaker: no [N]: lines found")
    uniq = sorted(set(ids))
    k = max(uniq)
    expected = list(range(1, k + 1))
    if uniq != expected:
        raise RuntimeError(
            "VibeVoice multi-speaker: speaker numbers must be sequential 1..K without gaps "
            f"(found speakers {uniq})"
        )
    return k


def _vibevoice_bracket_lines_to_speaker_script(text: str) -> str:
    """Map [N]: lines to Speaker N: for VibeVoiceProcessor._parse_script."""
    out: List[str] = []
    for line in text.splitlines():
        m = _VIBEVOICE_BRACKET_LINE.match(line) if line.strip() else None
        if m:
            n, rest = int(m.group(1)), m.group(2).strip()
            out.append(f"Speaker {n}: {rest}")
        else:
            out.append(line)
    return "\n".join(out)


class ChatterboxTester:
    """Main application class for Chatterbox / Qwen3 parameter testing."""

    # ------------------------------------------------------------------ #
    # Language tables                                                       #
    # ------------------------------------------------------------------ #

    # Multilanguage model (24 languages) – display label → ISO code
    LANGUAGES = {
        "Arabic (ar)": "ar",
        "Danish (da)": "da",
        "Deutsch (de)": "de",
        "Dutch (nl)": "nl",
        "English (en)": "en",
        "Español (es)": "es",
        "Finnish (fi)": "fi",
        "Français (fr)": "fr",
        "Greek (el)": "el",
        "Hebrew (he)": "he",
        "Hindi (hi)": "hi",
        "Italiano (it)": "it",
        "Japanese (ja)": "ja",
        "Korean (ko)": "ko",
        "Malay (ms)": "ms",
        "Norwegian (no)": "no",
        "Polski (pl)": "pl",
        "Português (pt)": "pt",
        "Русский (ru)": "ru",
        "Swahili (sw)": "sw",
        "Swedish (sv)": "sv",
        "Türkçe (tr)": "tr",
        "中文 (zh)": "zh",
    }

    # Qwen3 (10 languages) – display label → full language name passed to API
    QWEN3_LANGUAGES = {
        "English (en)":    "English",
        "Deutsch (de)":    "German",
        "中文 (zh)":        "Chinese",
        "Japanese (ja)":   "Japanese",
        "Korean (ko)":     "Korean",
        "Français (fr)":   "French",
        "Русский (ru)":    "Russian",
        "Português (pt)":  "Portuguese",
        "Español (es)":    "Spanish",
        "Italiano (it)":   "Italian",
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Chatterbox TTS Tester")

        # Set initial size (position will be set after loading settings)
        self.root.geometry("500x980")

        # Audio state
        self.current_audio: Optional[np.ndarray] = None
        self.sample_rate: int = 24000
        self.is_playing: bool = False
        self.playback_position: float = 0.0
        self.audio_duration: float = 0.0
        self.temp_audio_file: Optional[str] = None
        self.update_timer: Optional[str] = None
        self.external_player_process = None

        # A/B state comparison system
        self.active_state: int = 1  # 1 or 2
        self.state_a: Dict[str, Any] = {
            'audio': None, 'temp_file': None, 'text': '',
            'seed': '12345', 'params': {}, 'needs_refresh': True
        }
        self.state_b: Dict[str, Any] = {
            'audio': None, 'temp_file': None, 'text': '',
            'seed': '12345', 'params': {}, 'needs_refresh': True
        }

        self.needs_refresh: bool = True

        # Model and generation state
        self.device = self._detect_device()
        self.current_model: Optional[Any] = None
        self.current_model_type = "multilanguage"

        self.is_generating: bool = False

        # UI containers filled during _build_ui
        self.param_vars: Dict[str, tk.DoubleVar] = {}
        self.param_labels: Dict[str, tk.Label] = {}
        self.slider_frames: Dict[str, tk.Frame] = {}
        self.slider_scales: Dict[str, tk.Scale] = {}

        # UI state
        self.reference_audio_files = self._get_reference_audio_files()

        # Settings / presets persistence
        self.settings_file = Path(__file__).parent / ".chatterbox_tester_settings.json"
        self.presets_file  = Path(__file__).parent / ".chatterbox_tester_presets.yaml"
        self.presets: Dict[str, Dict[str, float]] = {}

        # History for undo/redo
        self.history: list[Dict[str, Any]] = []
        self.history_position: int = -1
        self.is_restoring_state: bool = False
        self._slider_save_timer: Optional[str] = None
        self._text_save_timer:   Optional[str] = None

        self._build_ui()
        self._load_settings()
        self._load_presets()
        self._load_initial_model()
        self._update_model_ui()
        self._update_load_button_state()

        # Keyboard shortcuts
        self.root.bind('<Control-z>', lambda e: self._undo())
        self.root.bind('<Control-y>', lambda e: self._redo())
        self.root.bind('<Control-Z>', lambda e: self._undo())
        self.root.bind('<Control-Y>', lambda e: self._redo())

        self._save_state_to_history()
        self._update_refresh_button_color()

    # ------------------------------------------------------------------ #
    # Device detection / file helpers                                       #
    # ------------------------------------------------------------------ #

    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_reference_audio_files(self) -> list:
        ref_audio_dir = Path(__file__).parent / "data" / "input" / "reference_audio"
        if not ref_audio_dir.exists():
            return []
        return sorted([f.name for f in ref_audio_dir.glob("*.wav")])

    def _get_ref_text_path(self) -> Optional[Path]:
        """Return the .txt sidecar path for the current reference audio, or None."""
        wav = self.ref_audio_var.get()
        if not wav:
            return None
        p = Path(__file__).parent / "data" / "input" / "reference_audio" / Path(wav).stem
        txt = p.with_suffix(".txt")
        return txt if txt.exists() else None

    def _reference_audio_full_path(self, filename: str) -> str:
        return str(
            Path(__file__).parent / "data" / "input" / "reference_audio" / filename
        )

    def _load_vibevoice_reference_np(self, wav_path: str) -> np.ndarray:
        """Load mono 24 kHz reference and apply voice_speed_factor (shared for all speakers)."""
        import librosa

        ref_audio_np, _ = librosa.load(wav_path, sr=24000, mono=True)
        speed_factor = float(self.param_vars["voice_speed_factor"].get())
        if speed_factor != 1.0:
            target_len = int(len(ref_audio_np) / speed_factor)
            ref_audio_np = np.interp(
                np.linspace(0, len(ref_audio_np) - 1, target_len),
                np.arange(len(ref_audio_np)),
                ref_audio_np,
            ).astype(np.float32)
        return ref_audio_np

    def _refresh_reference_audio_list(self):
        current = self.ref_audio_var.get()
        self.reference_audio_files = self._get_reference_audio_files()
        self.ref_dropdown.config(values=self.reference_audio_files)
        if current in self.reference_audio_files:
            self.ref_audio_var.set(current)
        elif self.reference_audio_files:
            self.ref_audio_var.set(self.reference_audio_files[0])
        vv_vals: List[str] = [""] + list(self.reference_audio_files)
        for var, dd in (
            (self.ref_audio_var_2, self.vv_extra_dropdowns[0]),
            (self.ref_audio_var_3, self.vv_extra_dropdowns[1]),
            (self.ref_audio_var_4, self.vv_extra_dropdowns[2]),
        ):
            cur = var.get()
            dd.config(values=vv_vals)
            if cur in vv_vals:
                var.set(cur)
            else:
                var.set("")
        self._update_load_button_state()
        self._update_ref_text_indicator()
        print(f"Reference audio list refreshed: {len(self.reference_audio_files)} files found")

    # ------------------------------------------------------------------ #
    # Preset helpers                                                        #
    # ------------------------------------------------------------------ #

    def _get_preset_key(self) -> str:
        """Return preset dict key for current ref_audio + model combination."""
        ref   = self.ref_audio_var.get()
        model = self.model_var.get()
        if model in ("turbo", "qwen3") or model in _VIBEVOICE_UI_MODELS:
            return f"{ref}::{model}"
        # classic / multilanguage keep the old-style key for backward compat
        return ref

    def _has_preset(self) -> bool:
        return self._get_preset_key() in self.presets

    def _on_load_preset(self):
        if not self._has_preset():
            return
        try:
            preset = self.presets[self._get_preset_key()]
            self.is_restoring_state = True
            for param_name, value in preset.items():
                if param_name in self.param_vars:
                    try:
                        self.param_vars[param_name].set(float(value))
                        self._format_label(param_name)
                    except Exception:
                        pass
                elif param_name == "use_sampling":
                    self.use_sampling_var.set(bool(value))
            self.is_restoring_state = False
            self._mark_needs_refresh()
            self._save_state_to_history()
            self.load_preset_btn.config(text="✓ Loaded")
            print(f"Loaded preset for '{self._get_preset_key()}'")
            self.root.after(1500, lambda: self.load_preset_btn.config(text="Load"))
        except Exception as e:
            print(f"Error loading preset: {e}")

    def _on_save_preset(self):
        self._save_preset()
        self.save_preset_btn.config(text="✓ Saved")
        self.root.after(1500, lambda: self.save_preset_btn.config(text="Save"))

    # ------------------------------------------------------------------ #
    # Build UI                                                              #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        # ---- Reference Audio row ----------------------------------------
        ref_frame = tk.Frame(self.root)
        ref_frame.pack(fill=tk.X, padx=10, pady=5)

        self.ref_audio_var = tk.StringVar(
            value=self.reference_audio_files[0] if self.reference_audio_files else ""
        )
        self.ref_dropdown = ttk.Combobox(
            ref_frame, textvariable=self.ref_audio_var,
            values=self.reference_audio_files, state="readonly", font=("Arial", 12)
        )
        self.ref_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.ref_dropdown.bind(
            '<<ComboboxSelected>>',
            lambda e: (self._on_text_change(), self._mark_needs_refresh(),
                       self._update_load_button_state(), self._update_ref_text_indicator())
        )

        tk.Button(ref_frame, text="⭯", font=("Arial", 9),
                  command=self._refresh_reference_audio_list,
                  bg="#f0f0f0", width=3, cursor="hand2").pack(side=tk.LEFT, padx=2)

        self.load_preset_btn = tk.Button(
            ref_frame, text="Load", font=("Arial", 9),
            command=self._on_load_preset, bg="#f0f0f0",
            state=tk.DISABLED, cursor="hand2"
        )
        self.load_preset_btn.pack(side=tk.LEFT, padx=2)

        self.save_preset_btn = tk.Button(
            ref_frame, text="Save", font=("Arial", 9),
            command=self._on_save_preset, bg="#f0f0f0", cursor="hand2"
        )
        self.save_preset_btn.pack(side=tk.LEFT, padx=2)

        # ---- VibeVoice extra speaker refs (2–4): dropdown only, no Load/Save ----------
        self.vv_extra_ref_frame = tk.Frame(self.root)
        # Packed from _update_model_ui when a VibeVoice model is selected.
        vv_extra_values: List[str] = [""] + list(self.reference_audio_files)
        self.ref_audio_var_2 = tk.StringVar(value="")
        self.ref_audio_var_3 = tk.StringVar(value="")
        self.ref_audio_var_4 = tk.StringVar(value="")
        self.vv_extra_dropdowns: List[ttk.Combobox] = []
        for label, var in (
            ("Speaker 2 ref:", self.ref_audio_var_2),
            ("Speaker 3 ref:", self.ref_audio_var_3),
            ("Speaker 4 ref:", self.ref_audio_var_4),
        ):
            row = tk.Frame(self.vv_extra_ref_frame)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, font=("Arial", 10), width=14, anchor=tk.W).pack(
                side=tk.LEFT, padx=(0, 6)
            )
            cb = ttk.Combobox(
                row,
                textvariable=var,
                values=vv_extra_values,
                state="readonly",
                font=("Arial", 12),
            )
            cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
            cb.bind(
                "<<ComboboxSelected>>",
                lambda e: (self._on_text_change(), self._mark_needs_refresh()),
            )
            self.vv_extra_dropdowns.append(cb)

        # ---- ref_text indicator (always packed, text set by _update_ref_text_indicator)
        self.ref_text_label = tk.Label(
            self.root, text="", font=("Arial", 9), fg="gray", anchor=tk.W
        )
        self.ref_text_label.pack(fill=tk.X, padx=12)

        # ---- Model Selection dropdown ------------------------------------
        model_frame = tk.Frame(self.root)
        model_frame.pack(fill=tk.X, padx=10, pady=(8, 2))

        tk.Label(model_frame, text="Model:", font=("Arial", 10),
                 width=7, anchor=tk.W).pack(side=tk.LEFT)

        self.model_var = tk.StringVar(value="multilanguage")
        model_dropdown = ttk.Combobox(
            model_frame, textvariable=self.model_var,
            values=[
                "classic",
                "multilanguage",
                "turbo",
                "qwen3",
                "vibevoice",
                "vibevoice_1_5b",
                "vibevoice_q4",
            ],
            state="readonly", font=("Arial", 12), width=16
        )
        model_dropdown.pack(side=tk.LEFT, padx=5)
        model_dropdown.bind('<<ComboboxSelected>>', lambda e: self._on_model_change())

        # ---- Language dropdown (always packed; disabled when not needed) --
        self.lang_var = tk.StringVar(value="English (en)")
        self.lang_dropdown = ttk.Combobox(
            self.root, textvariable=self.lang_var,
            values=list(self.LANGUAGES.keys()),
            state="readonly", font=("Arial", 12)
        )
        self.lang_dropdown.pack(fill=tk.X, padx=10, pady=3)
        self.lang_dropdown.bind(
            '<<ComboboxSelected>>',
            lambda e: (self._on_text_change(), self._mark_needs_refresh())
        )

        # ---- VibeVoice sampling toggle -----------------------------------
        # When checked: do_sample=True → temperature/top_p applied before the 5-token constraint.
        # Effect is subtle (segment length / boundary variation); primary controls remain
        # cfg_scale, diffusion_steps, and seed.
        self.sampling_frame = tk.Frame(self.root)
        self.sampling_frame.pack(fill=tk.X, padx=10, pady=(0, 3))
        self.use_sampling_var = tk.BooleanVar(value=False)
        self.sampling_check = ttk.Checkbutton(
            self.sampling_frame,
            text="VibeVoice sampling mode (temperature/top_p active)",
            variable=self.use_sampling_var,
            command=lambda: (self._on_text_change(), self._mark_needs_refresh()),
        )
        self.sampling_check.pack(anchor=tk.W)

        # ---- Seed row ----------------------------------------------------
        seed_frame = tk.Frame(self.root)
        seed_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(seed_frame, text="Seed:", font=("Arial", 10)).pack(side=tk.LEFT)
        self.seed_var = tk.StringVar(value="12345")
        seed_entry = tk.Entry(seed_frame, textvariable=self.seed_var,
                              font=("Arial", 11), width=15)
        seed_entry.pack(side=tk.LEFT, padx=5)
        seed_entry.bind('<FocusOut>', lambda e: (self._on_text_change(), self._mark_needs_refresh()))
        seed_entry.bind('<Return>',   lambda e: (self._on_text_change(), self._mark_needs_refresh()))

        def randomize_seed():
            self.seed_var.set(str(random.randint(1, 999999)))
            self._on_text_change()
            self._mark_needs_refresh()

        tk.Button(seed_frame, text="🎲 Random", command=randomize_seed,
                  font=("Arial", 9)).pack(side=tk.LEFT, padx=5)

        # ---- Text input --------------------------------------------------
        self.text_limit_label = tk.Label(
            self.root,
            text="Text to speak (max. 500 characters):",
            font=("Arial", 10),
        )
        self.text_limit_label.pack(anchor=tk.W, padx=10, pady=(10, 0))
        text_frame = tk.Frame(self.root)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Arial", 11), height=6)
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scrollbar = tk.Scrollbar(text_frame, command=self.text_widget.yview)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget.config(yscrollcommand=text_scrollbar.set)
        self.text_widget.bind('<FocusOut>', lambda e: self._on_text_change())
        self.text_widget.bind('<KeyRelease>', lambda e: self._on_text_keypress())
        self.text_widget.bind('<Control-a>', self._text_select_all)
        self.text_widget.bind('<Control-A>', self._text_select_all)
        self.text_widget.bind('<Control-v>', self._text_paste_replace)
        self.text_widget.bind('<Control-V>', self._text_paste_replace)
        default_text = ("Ibus que re endundaectis aut que pro blaborios mil is plab illam, "
                        "voluptur arum fugias aut eos ditae idendi sed ut faccull oruptas ne net fuga. "
                        "Omnimus voluptae nonsectur se liquas nestia esti dicia cuptatquia seque plis "
                        "pere eos")
        self.text_widget.insert("1.0", default_text)

        char_count_frame = tk.Frame(self.root)
        char_count_frame.pack(fill=tk.X, padx=10)
        self.char_count_label = tk.Label(char_count_frame, text="0/500 characters",
                                         font=("Arial", 9), fg="gray")
        self.char_count_label.pack(anchor=tk.W)

        # ---- A/B state tabs ----------------------------------------------
        state_tabs_frame = tk.Frame(self.root, bg="#e0e0e0")
        state_tabs_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        self.state_a_btn = tk.Button(
            state_tabs_frame, text="1", font=("Arial", 14, "bold"),
            command=lambda: self._switch_state(1),
            bg="#ffffff", fg="#000000", relief=tk.SUNKEN, borderwidth=2, width=8, cursor="hand2"
        )
        self.state_a_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.copy_a_to_b_btn = tk.Button(
            state_tabs_frame, text="→", font=("Arial", 12, "bold"),
            command=self._copy_a_to_b, bg="#d0d0d0", relief=tk.RAISED,
            borderwidth=1, width=2, cursor="hand2"
        )
        self.copy_a_to_b_btn.pack(side=tk.LEFT, padx=2)

        self.copy_b_to_a_btn = tk.Button(
            state_tabs_frame, text="←", font=("Arial", 12, "bold"),
            command=self._copy_b_to_a, bg="#d0d0d0", relief=tk.RAISED,
            borderwidth=1, width=2, cursor="hand2"
        )
        self.copy_b_to_a_btn.pack(side=tk.LEFT, padx=2)

        self.state_b_btn = tk.Button(
            state_tabs_frame, text="2", font=("Arial", 14, "bold"),
            command=lambda: self._switch_state(2),
            bg="#e0e0e0", fg="#000000", relief=tk.RAISED, borderwidth=2, width=8, cursor="hand2"
        )
        self.state_b_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ---- Refresh button ----------------------------------------------
        refresh_frame = tk.Frame(self.root)
        refresh_frame.pack(fill=tk.X, padx=10, pady=5)
        self.refresh_btn = tk.Button(
            refresh_frame, text="⟳ REFRESH - Generate Audio",
            font=("", 14, "bold"), command=self._on_refresh,
            bg="#4CAF50", fg="white", activebackground="#45a049",
            relief=tk.RAISED, borderwidth=2, cursor="hand2"
        )
        self.refresh_btn.pack(fill=tk.X, ipady=8)

        # ---- Parameter sliders (grid layout for show/hide) ---------------
        params_frame = tk.Frame(self.root)
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        params_frame.columnconfigure(0, weight=1)

        for row_idx, (name, min_val, max_val, default, res, _models, is_int) in enumerate(_SLIDER_CONFIG):
            frame = tk.Frame(params_frame)
            frame.grid(row=row_idx, column=0, sticky='ew', pady=3)
            self.slider_frames[name] = frame

            tk.Label(frame, text=name, font=("Arial", 10),
                     width=20, anchor=tk.W).pack(side=tk.LEFT)

            var = tk.DoubleVar(value=default)
            self.param_vars[name] = var

            scale = tk.Scale(
                frame, from_=min_val, to=max_val, resolution=res,
                orient=tk.HORIZONTAL, variable=var, showvalue=False,
                command=lambda v, n=name: self._on_slider_change(n, v)
            )
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.slider_scales[name] = scale

            fmt = f"{int(default)}" if is_int else f"{default:.2f}"
            val_label = tk.Label(frame, text=fmt, font=("Arial", 10),
                                 width=6, anchor=tk.E, relief=tk.SUNKEN, borderwidth=1)
            val_label.pack(side=tk.RIGHT)
            self.param_labels[name] = val_label

        # ---- Timeline scrubber ------------------------------------------
        timeline_frame = tk.Frame(self.root)
        timeline_frame.pack(fill=tk.X, padx=10, pady=10)
        self.timeline_var = tk.DoubleVar(value=0)
        self.timeline = tk.Scale(
            timeline_frame, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.timeline_var, showvalue=False,
            command=self._on_timeline_change
        )
        self.timeline.pack(fill=tk.X)

        # ---- Playback controls ------------------------------------------
        controls_frame = tk.Frame(self.root)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)

        self.play_btn = tk.Button(controls_frame, text="▶", font=("Arial", 20),
                                  width=3, command=self._on_play, bg="#f0f0f0")
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.pause_btn = tk.Button(controls_frame, text="⏸", font=("Arial", 20),
                                   width=3, command=self._on_pause,
                                   state=tk.DISABLED, bg="#f0f0f0")
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        self.redo_btn = tk.Button(controls_frame, text="↷", font=("Arial", 16),
                                  width=3, command=self._redo, bg="#f0f0f0",
                                  state=tk.DISABLED)
        self.redo_btn.pack(side=tk.RIGHT, padx=5)

        self.undo_btn = tk.Button(controls_frame, text="↶", font=("Arial", 16),
                                  width=3, command=self._undo, bg="#f0f0f0",
                                  state=tk.DISABLED)
        self.undo_btn.pack(side=tk.RIGHT)

        # ---- Action buttons ---------------------------------------------
        action_frame = tk.Frame(self.root)
        action_frame.pack(fill=tk.X, padx=10, pady=5)

        self.copy_btn = tk.Button(
            action_frame, text="📋 Copy YAML to Clipboard",
            font=("Arial", 11), command=self._copy_settings, bg="#f0f0f0"
        )
        self.copy_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.save_btn = tk.Button(
            action_frame, text="💾 Save Audio",
            font=("Arial", 11), command=self._save_audio,
            bg="#f0f0f0", state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=(5, 0))

        insert_frame = tk.Frame(self.root)
        insert_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.insert_btn = tk.Button(
            insert_frame, text="📥 Insert from Clipboard",
            font=("Arial", 11), command=self._insert_from_clipboard, bg="#f0f0f0"
        )
        self.insert_btn.pack(fill=tk.X)

    # ------------------------------------------------------------------ #
    # Model UI helpers                                                      #
    # ------------------------------------------------------------------ #

    def _update_model_ui(self):
        """Show/hide sliders and update language dropdown for the active model."""
        model = self.model_var.get()

        # Slider visibility
        for name, frame in self.slider_frames.items():
            if model in _SLIDER_MODELS.get(name, []):
                frame.grid()
            else:
                frame.grid_remove()

        # Reconfigure top_k slider range per model
        if "top_k" in self.slider_scales:
            if model == "turbo":
                self.slider_scales["top_k"].config(from_=1, to=2000, resolution=1)
                cur = self.param_vars["top_k"].get()
                if cur < 1 or cur > 2000:
                    self.param_vars["top_k"].set(1000)
                self._format_label("top_k")
            elif model == "qwen3":
                self.slider_scales["top_k"].config(from_=1, to=400, resolution=1)
                cur = self.param_vars["top_k"].get()
                if cur > 400 or cur < 1:
                    self.param_vars["top_k"].set(50)
                self._format_label("top_k")
            elif model in _VIBEVOICE_UI_MODELS:
                # Hidden for vibevoice but reset to broad defaults when switching back.
                self.slider_scales["top_k"].config(from_=1, to=2000, resolution=1)

        # Language dropdown options
        if model == "multilanguage":
            self.lang_dropdown.config(
                state="readonly", values=list(self.LANGUAGES.keys())
            )
            if self.lang_var.get() not in self.LANGUAGES:
                self.lang_var.set("English (en)")
        elif model == "qwen3":
            self.lang_dropdown.config(
                state="readonly", values=list(self.QWEN3_LANGUAGES.keys())
            )
            if self.lang_var.get() not in self.QWEN3_LANGUAGES:
                self.lang_var.set("English (en)")
        else:
            # classic / turbo / vibevoice: no language selection
            self.lang_dropdown.config(state="disabled")

        if model in _VIBEVOICE_UI_MODELS:
            self.sampling_frame.pack(fill=tk.X, padx=10, pady=(0, 3))
            self.vv_extra_ref_frame.pack(fill=tk.X, padx=10, pady=(0, 4))
        else:
            self.sampling_frame.pack_forget()
            self.vv_extra_ref_frame.pack_forget()

        text_limit = self._get_text_limit()
        self.text_limit_label.config(text=f"Text to speak (max. {text_limit} characters):")
        self._update_char_count()

        # ref_text indicator (only relevant for qwen3)
        self._update_ref_text_indicator()

    def _update_ref_text_indicator(self):
        """Update the ref_text sidecar indicator label."""
        model = self.model_var.get()
        if model != "qwen3":
            self.ref_text_label.config(text="")
            return
        txt = self._get_ref_text_path()
        if txt:
            self.ref_text_label.config(
                text=f"✓ ref_text: {txt.name}", fg="#2a7a2a"
            )
        else:
            self.ref_text_label.config(
                text="⚠ no .txt sidecar – x_vector only (lower quality)", fg="#b05000"
            )

    def _update_language_dropdown_state(self):
        """Legacy shim – delegates to _update_model_ui."""
        self._update_model_ui()

    def _format_label(self, param_name: str):
        """Update slider value label with correct formatting."""
        value = self.param_vars[param_name].get()
        # Check whether this slider is integer
        is_int = next(
            (row[6] for row in _SLIDER_CONFIG if row[0] == param_name), False
        )
        if is_int:
            self.param_labels[param_name].config(text=f"{int(value)}")
        else:
            self.param_labels[param_name].config(text=f"{value:.2f}")

    # ------------------------------------------------------------------ #
    # Settings persistence                                                  #
    # ------------------------------------------------------------------ #

    def _load_settings(self):
        if not self.settings_file.exists():
            self._center_window()
            return
        try:
            import json
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)

            if settings.get('reference_audio') and settings['reference_audio'] in self.reference_audio_files:
                self.ref_audio_var.set(settings['reference_audio'])
            for key, var in (
                ('reference_audio_2', self.ref_audio_var_2),
                ('reference_audio_3', self.ref_audio_var_3),
                ('reference_audio_4', self.ref_audio_var_4),
            ):
                v = settings.get(key, '')
                if not v:
                    var.set('')
                elif v in self.reference_audio_files:
                    var.set(v)
                else:
                    var.set('')
            if settings.get('model_type'):
                self.model_var.set(settings['model_type'])
            if settings.get('language'):
                self.lang_var.set(settings['language'])
            if settings.get('use_sampling') is not None:
                self.use_sampling_var.set(bool(settings['use_sampling']))
            if settings.get('active_state'):
                self.active_state = settings['active_state']

            for key, state_obj in [('state_a', self.state_a), ('state_b', self.state_b)]:
                if settings.get(key):
                    d = settings[key]
                    state_obj['text']         = d.get('text', '')
                    state_obj['seed']         = d.get('seed', '12345')
                    state_obj['params']       = d.get('params', {})
                    state_obj['needs_refresh'] = d.get('needs_refresh', True)

            active_state = self.state_a if self.active_state == 1 else self.state_b
            self.seed_var.set(str(active_state.get('seed', '12345')))

            text = active_state.get('text', '')
            if text:
                self.text_widget.delete("1.0", tk.END)
                self.text_widget.insert("1.0", text)

            params = active_state.get('params', {})
            for param_name, value in params.items():
                if param_name in self.param_vars:
                    try:
                        self.param_vars[param_name].set(float(value))
                        self._format_label(param_name)
                    except Exception:
                        pass

            if self.active_state == 1:
                self.state_a_btn.config(relief=tk.SUNKEN, bg="#ffffff")
                self.state_b_btn.config(relief=tk.RAISED,  bg="#e0e0e0")
            else:
                self.state_a_btn.config(relief=tk.RAISED,  bg="#e0e0e0")
                self.state_b_btn.config(relief=tk.SUNKEN, bg="#ffffff")

            if settings.get('window_x') is not None and settings.get('window_y') is not None:
                x, y = settings['window_x'], settings['window_y']
                if -100 <= x < 3000 and -100 <= y < 3000:
                    self.root.geometry(f"500x980+{x}+{y}")
                else:
                    self._center_window()
            else:
                self._center_window()

        except Exception as e:
            print(f"Could not load settings: {e}")
            self._center_window()

    def _save_settings(self):
        try:
            import json
            current_state = self.state_a if self.active_state == 1 else self.state_b
            current_ui = self._get_current_ui_state()
            current_state['text']   = current_ui['text']
            current_state['seed']   = current_ui['seed']
            current_state['params'] = current_ui['params']

            self.root.update_idletasks()
            geometry = self.root.geometry()
            parts = geometry.split('+')
            window_x = int(parts[1]) if len(parts) > 1 else None
            window_y = int(parts[2]) if len(parts) > 2 else None

            settings = {
                'reference_audio': self.ref_audio_var.get(),
                'reference_audio_2': self.ref_audio_var_2.get(),
                'reference_audio_3': self.ref_audio_var_3.get(),
                'reference_audio_4': self.ref_audio_var_4.get(),
                'model_type':      self.model_var.get(),
                'language':        self.lang_var.get(),
                'use_sampling':    self.use_sampling_var.get(),
                'active_state':    self.active_state,
                'window_x':        window_x,
                'window_y':        window_y,
                'state_a': {
                    'text':         self.state_a.get('text', ''),
                    'seed':         self.state_a.get('seed', '12345'),
                    'params':       self.state_a.get('params', {}),
                    'needs_refresh': self.state_a.get('needs_refresh', True),
                },
                'state_b': {
                    'text':         self.state_b.get('text', ''),
                    'seed':         self.state_b.get('seed', '12345'),
                    'params':       self.state_b.get('params', {}),
                    'needs_refresh': self.state_b.get('needs_refresh', True),
                },
            }
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Could not save settings: {e}")

    def _center_window(self):
        self.root.update_idletasks()
        w, h = 500, 980
        x = (self.root.winfo_screenwidth()  - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    # ------------------------------------------------------------------ #
    # Preset I/O                                                            #
    # ------------------------------------------------------------------ #

    def _load_presets(self):
        if not self.presets_file.exists():
            return
        try:
            import yaml
            with open(self.presets_file, 'r', encoding='utf-8') as f:
                self.presets = yaml.safe_load(f) or {}
            print(f"Loaded {len(self.presets)} preset(s) from {self.presets_file.name}")
        except Exception as e:
            print(f"Could not load presets: {e}")
            self.presets = {}

    def _save_preset(self):
        try:
            import yaml
            ref = self.ref_audio_var.get()
            if not ref:
                return
            key = self._get_preset_key()
            model = self.model_var.get()

            # Collect only params that are visible/applicable for this model
            preset_params = {}
            for name in self.param_vars:
                if model in _SLIDER_MODELS.get(name, []):
                    preset_params[name] = round(float(self.param_vars[name].get()), 4)
            if model in _VIBEVOICE_UI_MODELS:
                preset_params["use_sampling"] = bool(self.use_sampling_var.get())

            self.presets[key] = preset_params
            sorted_presets = dict(sorted(self.presets.items()))
            with open(self.presets_file, 'w', encoding='utf-8') as f:
                yaml.dump(sorted_presets, f, default_flow_style=False, sort_keys=False)

            print(f"Saved preset '{key}'")
            self._update_load_button_state()
        except Exception as e:
            print(f"Could not save preset: {e}")

    def _update_load_button_state(self):
        if self._has_preset():
            self.load_preset_btn.config(state=tk.NORMAL)
        else:
            self.load_preset_btn.config(state=tk.DISABLED)

    # ------------------------------------------------------------------ #
    # Model loading                                                         #
    # ------------------------------------------------------------------ #

    def _load_initial_model(self):
        self._load_model(self.model_var.get())

    def _load_model(self, model_ui_name: str):
        """Load the appropriate TTS model into cache."""
        model_map = {
            "classic":      "standard",
            "multilanguage": "multilingual",
            "turbo":         "turbo",
            "qwen3":         "qwen3",
            "vibevoice":     "vibevoice",
            "vibevoice_1_5b": "vibevoice_1_5b",
            "vibevoice_q4": "vibevoice_q4",
        }
        model_type = model_map.get(model_ui_name, "standard")

        # When switching away from a VibeVoice variant (7B Q8 ≈ 8 GB VRAM) to a
        # non-VV model, free the VV weights first. Otherwise the next load
        # (e.g. Qwen3) competes for VRAM, swaps to system RAM and appears to
        # hang the GUI for minutes.
        prev_type = getattr(self, "current_model_type", None)
        _VV_TYPES = {"vibevoice", "vibevoice_1_5b", "vibevoice_q4"}
        if (
            prev_type in _VV_TYPES
            and model_type not in _VV_TYPES
            and model_ui_name not in _VIBEVOICE_UI_MODELS
        ):
            self.current_model = None
            ChatterboxModelCache.evict_vibevoice(self.device)

        config: Dict[str, Any] = {"generation": {"model_type": model_type}}
        if model_ui_name in _VIBEVOICE_UI_MODELS:
            tester_cfg: Dict[str, Any] = {}
            if _VIBEVOICE_TESTER_ATTN:
                tester_cfg["vibevoice_attn_implementation"] = _VIBEVOICE_TESTER_ATTN
            elif model_ui_name == "vibevoice":
                tester_cfg["vibevoice_attn_implementation"] = "sdpa"
            elif model_ui_name == "vibevoice_q4":
                tester_cfg["vibevoice_attn_implementation"] = "sdpa"
            config["chatterbox_tester"] = tester_cfg
        self.current_model = ChatterboxModelCache.get_model(
            self.device, model_type, config=config
        )
        self.current_model_type = model_type
        if model_ui_name in _VIBEVOICE_UI_MODELS and self.current_model is not None:
            lm = getattr(
                getattr(self.current_model, "model", None), "language_model", None
            )
            impl = (
                getattr(getattr(lm, "config", None), "_attn_implementation", None)
                if lm is not None
                else None
            )
            print(
                "VibeVoice: language_model._attn_implementation="
                f"{impl!r} (tester env CHATTERBOX_TESTER_VIBEVOICE_ATTN="
                f"{_VIBEVOICE_TESTER_ATTN!r})"
            )

    def _on_model_change(self):
        self._load_model(self.model_var.get())
        self._update_model_ui()
        self._update_load_button_state()
        self._on_text_change()
        self._mark_needs_refresh()

    # ------------------------------------------------------------------ #
    # Text / slider change handlers                                         #
    # ------------------------------------------------------------------ #

    def _text_select_all(self, event):
        self.text_widget.tag_add(tk.SEL, "1.0", tk.END)
        self.text_widget.mark_set(tk.INSERT, "1.0")
        self.text_widget.see(tk.INSERT)
        return 'break'

    def _text_paste_replace(self, event):
        try:
            if self.text_widget.tag_ranges(tk.SEL):
                self.text_widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
            self.text_widget.insert(tk.INSERT, self.root.clipboard_get())
            self._on_text_change()
            self._mark_needs_refresh()
        except tk.TclError:
            pass
        return 'break'

    def _on_text_keypress(self):
        self._mark_needs_refresh()
        if hasattr(self, '_text_save_timer') and self._text_save_timer:
            self.root.after_cancel(self._text_save_timer)
        self._text_save_timer = self.root.after(1000, self._save_text_after_idle)

    def _save_text_after_idle(self):
        self._save_state_to_history()

    def _on_text_change(self):
        if hasattr(self, '_text_save_timer') and self._text_save_timer:
            self.root.after_cancel(self._text_save_timer)
        self._text_save_timer = self.root.after(500, self._save_state_to_history)

    def _get_text_limit(self) -> int:
        return (
            _LONGFORM_TEXT_UI_LIMIT
            if self.model_var.get() in _LONGFORM_TEXT_UI_MODELS
            else 500
        )

    def _update_char_count(self):
        text = self.text_widget.get("1.0", tk.END).strip()
        text_limit = self._get_text_limit()
        if len(text) > text_limit:
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.insert("1.0", text[:text_limit])
            text = text[:text_limit]
        self.char_count_label.config(text=f"{len(text)}/{text_limit} characters")

    def _on_slider_change(self, param_name: str, value: str):
        self._format_label(param_name)
        self._mark_needs_refresh()
        if hasattr(self, '_slider_save_timer') and self._slider_save_timer:
            self.root.after_cancel(self._slider_save_timer)
        self._slider_save_timer = self.root.after(300, self._save_state_to_history)

    # ------------------------------------------------------------------ #
    # State / history                                                       #
    # ------------------------------------------------------------------ #

    def _get_all_params(self) -> Dict[str, float]:
        params = {name: float(var.get()) for name, var in self.param_vars.items()}
        params["use_sampling"] = 1.0 if self.use_sampling_var.get() else 0.0
        return params

    def _get_current_state(self) -> Dict[str, Any]:
        return {
            'reference_audio': self.ref_audio_var.get(),
            'reference_audio_2': self.ref_audio_var_2.get(),
            'reference_audio_3': self.ref_audio_var_3.get(),
            'reference_audio_4': self.ref_audio_var_4.get(),
            'model_type':      self.model_var.get(),
            'language':        self.lang_var.get(),
            'seed':            self.seed_var.get(),
            'text':            self.text_widget.get("1.0", tk.END).strip(),
            'params':          self._get_all_params(),
        }

    def _get_current_ui_state(self) -> Dict[str, Any]:
        return {
            'text':   self.text_widget.get("1.0", tk.END).strip(),
            'seed':   self.seed_var.get(),
            'params': self._get_all_params(),
        }

    def _save_state_to_history(self, force: bool = False):
        if self.is_restoring_state:
            return
        current_state = self._get_current_state()
        if not force and self.history and self.history_position >= 0:
            if current_state == self.history[self.history_position]:
                return
        if self.history_position < len(self.history) - 1:
            self.history = self.history[:self.history_position + 1]
        self.history.append(current_state)
        self.history_position = len(self.history) - 1
        if len(self.history) > 50:
            self.history.pop(0)
            self.history_position -= 1
        self._update_history_buttons()

    def _restore_state(self, state: Dict[str, Any]):
        self.is_restoring_state = True
        try:
            if state.get('reference_audio') and state['reference_audio'] in self.reference_audio_files:
                self.ref_audio_var.set(state['reference_audio'])
            for key, var in (
                ('reference_audio_2', self.ref_audio_var_2),
                ('reference_audio_3', self.ref_audio_var_3),
                ('reference_audio_4', self.ref_audio_var_4),
            ):
                v = state.get(key, '')
                if not v:
                    var.set('')
                elif v in self.reference_audio_files:
                    var.set(v)
                else:
                    var.set('')
            if state.get('model_type'):
                old_model = self.model_var.get()
                self.model_var.set(state['model_type'])
                if old_model != state['model_type']:
                    self._load_model(state['model_type'])
                    self._update_model_ui()
            if state.get('language'):
                self.lang_var.set(state['language'])
            if state.get('seed'):
                self.seed_var.set(str(state['seed']))
            if state.get('text') is not None:
                self.text_widget.delete("1.0", tk.END)
                self.text_widget.insert("1.0", state['text'])
            for param_name, value in state.get('params', {}).items():
                if param_name in self.param_vars:
                    try:
                        self.param_vars[param_name].set(float(value))
                        self._format_label(param_name)
                    except Exception:
                        pass
                elif param_name == "use_sampling":
                    self.use_sampling_var.set(bool(value))
        finally:
            self.is_restoring_state = False

    def _undo(self):
        if self.history_position > 0:
            self.history_position -= 1
            self._restore_state(self.history[self.history_position])
            self._update_history_buttons()
            self._mark_needs_refresh()

    def _redo(self):
        if self.history_position < len(self.history) - 1:
            self.history_position += 1
            self._restore_state(self.history[self.history_position])
            self._update_history_buttons()
            self._mark_needs_refresh()

    def _update_history_buttons(self):
        self.undo_btn.config(state=tk.NORMAL if self.history_position > 0 else tk.DISABLED)
        self.redo_btn.config(
            state=tk.NORMAL if self.history_position < len(self.history) - 1 else tk.DISABLED
        )

    def _mark_needs_refresh(self):
        active_state = self.state_a if self.active_state == 1 else self.state_b
        active_state['needs_refresh'] = True
        self._update_refresh_button_color()

    def _update_refresh_button_color(self):
        active_state = self.state_a if self.active_state == 1 else self.state_b
        self.refresh_btn.config(
            bg="#4CAF50" if active_state['needs_refresh'] else "#677867"
        )

    def _switch_state(self, state_num: int):
        if state_num == self.active_state:
            return
        current_state = self.state_a if self.active_state == 1 else self.state_b
        current_ui = self._get_current_ui_state()
        current_state['text']   = current_ui['text']
        current_state['seed']   = current_ui['seed']
        current_state['params'] = current_ui['params']

        self.active_state = state_num
        new_state = self.state_a if state_num == 1 else self.state_b

        if state_num == 1:
            self.state_a_btn.config(relief=tk.SUNKEN, bg="#ffffff")
            self.state_b_btn.config(relief=tk.RAISED,  bg="#e0e0e0")
        else:
            self.state_a_btn.config(relief=tk.RAISED,  bg="#e0e0e0")
            self.state_b_btn.config(relief=tk.SUNKEN, bg="#ffffff")

        self._stop_playback()
        self.is_restoring_state = True
        try:
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.insert("1.0", new_state.get('text', ''))
            self.seed_var.set(new_state.get('seed', '12345'))
            for param_name, value in new_state.get('params', {}).items():
                if param_name in self.param_vars:
                    try:
                        self.param_vars[param_name].set(float(value))
                        self._format_label(param_name)
                    except Exception:
                        pass
            self.current_audio  = new_state.get('audio')
            self.temp_audio_file = new_state.get('temp_file')
            if self.current_audio is not None and self.temp_audio_file:
                if self.update_timer:
                    self.root.after_cancel(self.update_timer)
                    self.update_timer = None
                self.audio_duration = len(self.current_audio) / self.sample_rate
                self.timeline.config(to=self.audio_duration * 1000)
                self.timeline_var.set(0)
                self.playback_position = 0.0
                self.save_btn.config(state=tk.NORMAL)
            else:
                if self.update_timer:
                    self.root.after_cancel(self.update_timer)
                    self.update_timer = None
                self.audio_duration = 0.0
                self.timeline.config(to=100)
                self.timeline_var.set(0)
                self.playback_position = 0.0
                self.save_btn.config(state=tk.DISABLED)
        finally:
            self.is_restoring_state = False

        self._update_refresh_button_color()
        self._update_history_buttons()

    def _copy_state(self, from_state_num: int, to_state_num: int):
        current_state = self.state_a if self.active_state == 1 else self.state_b
        current_ui = self._get_current_ui_state()
        current_state['text']   = current_ui['text']
        current_state['seed']   = current_ui['seed']
        current_state['params'] = current_ui['params']

        if self.active_state == to_state_num:
            self._save_state_to_history(force=True)

        source = self.state_a if from_state_num == 1 else self.state_b
        copied_state = {
            'audio':         source.get('audio'),
            'temp_file':     source.get('temp_file'),
            'text':          source.get('text', ''),
            'seed':          source.get('seed', '12345'),
            'params':        source.get('params', {}).copy(),
            'needs_refresh': source.get('needs_refresh', True),
        }
        if to_state_num == 1:
            self.state_a = copied_state
        else:
            self.state_b = copied_state

        if self.active_state == to_state_num:
            self.is_restoring_state = True
            try:
                self.text_widget.delete("1.0", tk.END)
                self.text_widget.insert("1.0", copied_state.get('text', ''))
                self.seed_var.set(copied_state.get('seed', '12345'))
                for param_name, value in copied_state.get('params', {}).items():
                    if param_name in self.param_vars:
                        try:
                            self.param_vars[param_name].set(float(value))
                            self._format_label(param_name)
                        except Exception:
                            pass
                self._update_refresh_button_color()
            finally:
                self.is_restoring_state = False

    def _copy_a_to_b(self):
        self._copy_state(1, 2)

    def _copy_b_to_a(self):
        self._copy_state(2, 1)

    # ------------------------------------------------------------------ #
    # Generation                                                            #
    # ------------------------------------------------------------------ #

    def _print_generation_summary(
        self,
        *,
        speaker_names: List[str],
        num_unique_speakers: int,
        num_segments: int,
        prefilling_tokens: Optional[int],
        generated_tokens: Optional[int],
        total_tokens: Optional[int],
        generation_seconds: float,
        audio_duration_seconds: float,
    ) -> None:
        """Print a one-shot text summary after a successful generate (all models)."""
        print(f"Speaker names: {speaker_names!r}")
        print(f"Number of unique speakers: {num_unique_speakers}")
        print(f"Number of segments: {num_segments}")
        if prefilling_tokens is not None and generated_tokens is not None and total_tokens is not None:
            print(f"Prefilling tokens: {prefilling_tokens}")
            print(f"Generated tokens: {generated_tokens}")
            print(f"Total tokens: {total_tokens}")
        else:
            print("Prefilling tokens: N/A")
            print("Generated tokens: N/A")
            print("Total tokens: N/A")
        print(f"Generation time: {generation_seconds:.2f} seconds")
        print(f"Audio duration: {audio_duration_seconds:.2f} seconds")
        if audio_duration_seconds > 0:
            rtf = generation_seconds / audio_duration_seconds
            print(f"RTF (Real Time Factor): {rtf:.2f}x")
        else:
            print("RTF (Real Time Factor): N/A")

    def _on_refresh(self):
        if self.is_generating:
            return
        self._update_char_count()
        self._save_settings()
        threading.Thread(target=self._generate_and_play, daemon=True).start()

    def _generate_and_play(self):
        self.is_generating = True
        self.root.after(0, lambda: self.refresh_btn.config(
            state=tk.DISABLED, text="⏳ Generating…"
        ))

        text = self.text_widget.get("1.0", tk.END).strip()
        if not text:
            self.is_generating = False
            self.root.after(0, lambda: self.refresh_btn.config(
                state=tk.NORMAL, text="⟳ REFRESH - Generate Audio"
            ))
            return

        try:
            seed_str = self.seed_var.get().strip()
            seed = int(seed_str) if seed_str.isdigit() else 0
        except Exception:
            seed = 0

        # 0 or empty field → new random seed every generation (same range as 🎲 Random button).
        if seed == 0:
            seed = random.randint(1, 999999)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            print(f"🎲 Seed 0 / empty → random seed for this run: {seed}")
        else:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            print(f"🎲 Seed set to: {seed}")

        ref_audio_file = self.ref_audio_var.get()
        ref_audio_path = str(
            Path(__file__).parent / "data" / "input" / "reference_audio" / ref_audio_file
        )

        try:
            model_ui = self.model_var.get()
            current_model = self.current_model
            if current_model is None:
                raise RuntimeError("No TTS model loaded")

            gen_elapsed = 0.0
            sum_speaker_names: List[str] = []
            sum_unique_speakers = 1
            sum_segments = 1
            sum_prefill: Optional[int] = None
            sum_gen_tok: Optional[int] = None
            sum_total_tok: Optional[int] = None

            # ---- Qwen3 voice cloning ------------------------------------
            if model_ui == "qwen3":
                lang_key  = self.lang_var.get()
                lang_name = self.QWEN3_LANGUAGES.get(lang_key, "English")

                txt_path = self._get_ref_text_path()
                ref_text = txt_path.read_text(encoding="utf-8").strip() if txt_path else None
                x_vec_only = (ref_text is None)

                gen_kwargs = {
                    "temperature":           self.param_vars["temperature"].get(),
                    "top_k":                 int(self.param_vars["top_k"].get()),
                    "top_p":                 self.param_vars["top_p"].get(),
                    "repetition_penalty":    self.param_vars["repetition_penalty"].get(),
                    "do_sample":             True,
                    "subtalker_dosample":    True,
                    "subtalker_top_k":       int(self.param_vars["subtalker_top_k"].get()),
                    "subtalker_top_p":       self.param_vars["subtalker_top_p"].get(),
                    "subtalker_temperature": self.param_vars["subtalker_temperature"].get(),
                }
                print(f"🎙️ Qwen3 | lang={lang_name} | x_vec_only={x_vec_only} | {gen_kwargs}")

                # Wire up a token-level progress bar on the talker LM.
                # The public ``generate_voice_clone`` API does not forward a
                # ``streamer=`` kwarg, but internally it calls
                # ``self.model.talker.generate`` (HF generate). Temporarily wrap
                # that bound method to inject our streamer, then restore.
                talker = getattr(
                    getattr(current_model, "model", None), "talker", None
                )
                streamer = _Qwen3ProgressStreamer(total=2048, label="Qwen3")
                orig_talker_generate = None
                if talker is not None and hasattr(talker, "generate"):
                    orig_talker_generate = talker.generate

                    def _patched_generate(*a, _orig=orig_talker_generate, _s=streamer, **kw):
                        kw.setdefault("streamer", _s)
                        return _orig(*a, **kw)

                    talker.generate = _patched_generate  # type: ignore[assignment]

                _t0 = time.perf_counter()
                try:
                    wavs, sr = current_model.generate_voice_clone(
                        text=text,
                        language=lang_name,
                        ref_audio=ref_audio_path,
                        ref_text=ref_text,
                        x_vector_only_mode=x_vec_only,
                        **gen_kwargs,
                    )
                finally:
                    if talker is not None and orig_talker_generate is not None:
                        talker.generate = orig_talker_generate  # type: ignore[assignment]
                    streamer.end()
                gen_elapsed = time.perf_counter() - _t0
                audio_np = wavs[0].astype(np.float32)
                self.sample_rate = int(sr)
                sum_speaker_names = [Path(ref_audio_file).stem] if ref_audio_file else []
                sum_unique_speakers = 1
                sum_segments = 1

            # ---- VibeVoice-Large-Q8 --------------------------------------
            elif model_ui in _VIBEVOICE_UI_MODELS:
                if not hasattr(current_model, "_vv_processor"):
                    raise RuntimeError("VibeVoice processor not available on loaded model")
                processor = current_model._vv_processor

                ref_vars = (
                    self.ref_audio_var,
                    self.ref_audio_var_2,
                    self.ref_audio_var_3,
                    self.ref_audio_var_4,
                )

                vv_k = 1
                if _vibevoice_text_has_bracket_speaker_lines(text):
                    vv_k = _vibevoice_validate_multispeaker_bracket_text(text)
                    script_for_processor = _vibevoice_bracket_lines_to_speaker_script(text)
                    voice_samples_arg: List[np.ndarray] = []
                    for i in range(vv_k):
                        fn = ref_vars[i].get().strip()
                        if not fn:
                            raise RuntimeError(
                                f"VibeVoice multi-speaker: reference audio for speaker {i + 1} "
                                "is required (select a file in the corresponding dropdown)"
                            )
                        p = self._reference_audio_full_path(fn)
                        if not os.path.exists(p):
                            raise RuntimeError(
                                f"VibeVoice multi-speaker: reference file not found: {fn}"
                            )
                        voice_samples_arg.append(self._load_vibevoice_reference_np(p))
                else:
                    if not ref_audio_file or not os.path.exists(ref_audio_path):
                        raise RuntimeError("VibeVoice requires a valid reference audio file")
                    ref_audio_np = self._load_vibevoice_reference_np(ref_audio_path)
                    script_for_processor = f"Speaker 1: {' '.join(text.split())}"
                    voice_samples_arg = [ref_audio_np]

                vv_batch = processor(
                    [script_for_processor],
                    voice_samples=voice_samples_arg,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                parsed_scripts_batch = vv_batch.get("parsed_scripts")
                model_device = next(current_model.parameters()).device
                inputs = {
                    kk: v.to(model_device) if isinstance(v, torch.Tensor) else v
                    for kk, v in vv_batch.items()
                }
                prefill_tokens = int(inputs["input_ids"].shape[-1])

                sum_speaker_names = [
                    Path(ref_vars[i].get().strip()).stem for i in range(vv_k)
                ]
                sum_unique_speakers = vv_k
                if parsed_scripts_batch and len(parsed_scripts_batch) > 0 and parsed_scripts_batch[0]:
                    sum_segments = len(parsed_scripts_batch[0])
                else:
                    sum_segments = 1

                current_model.set_ddpm_inference_steps(
                    int(self.param_vars["diffusion_steps"].get())
                )

                use_sampling = bool(self.use_sampling_var.get())
                gen_kwargs = {
                    "tokenizer": processor.tokenizer,
                    "cfg_scale": float(self.param_vars["cfg_scale"].get()),
                    "max_new_tokens": None,
                    "do_sample": use_sampling,
                }
                if use_sampling:
                    gen_kwargs["temperature"] = float(self.param_vars["temperature"].get())
                    gen_kwargs["top_p"] = float(self.param_vars["top_p"].get())

                # Log effective LM sampling args (temperature/top_p only affect output when do_sample=True).
                # Transformers omits TemperatureLogitsWarper when temperature==1.0 and TopPLogitsWarper when top_p>=1.0.
                # Omit tokenizer: its repr dumps added_tokens_decoder and floods the console.
                _vv_log = {k: v for k, v in gen_kwargs.items() if k != "tokenizer"}
                print(f"🎙️ VibeVoice | {_vv_log}")

                # User "print(f"🎙️ VibeVoice | {gen_kwargs}")" for detailed logging of the tokenizer parameters

                _t0 = time.perf_counter()
                with torch.no_grad():
                    output = current_model.generate(**inputs, **gen_kwargs)
                gen_elapsed = time.perf_counter() - _t0

                if not hasattr(output, "speech_outputs") or not output.speech_outputs:
                    raise RuntimeError("VibeVoice generation returned no speech outputs")

                total_tok = int(output.sequences.shape[-1])
                sum_prefill = prefill_tokens
                sum_gen_tok = total_tok - prefill_tokens
                sum_total_tok = total_tok

                speech_outputs = output.speech_outputs
                audio_tensor = (
                    torch.cat(speech_outputs, dim=-1)
                    if isinstance(speech_outputs, list)
                    else speech_outputs
                )
                audio_np = audio_tensor.cpu().float().numpy().squeeze().astype(np.float32)
                self.sample_rate = 24000

            # ---- Chatterbox Turbo ----------------------------------------
            elif model_ui == "turbo":
                params = {
                    "audio_prompt_path":  ref_audio_path,
                    "exaggeration":       self.param_vars["exaggeration"].get(),
                    "cfg_weight":         self.param_vars["cfg_weight"].get(),
                    "temperature":        self.param_vars["temperature"].get(),
                    "repetition_penalty": self.param_vars["repetition_penalty"].get(),
                    "min_p":              self.param_vars["min_p"].get(),
                    "top_p":              self.param_vars["top_p"].get(),
                    "top_k":              int(self.param_vars["top_k"].get()),
                }
                print(f"⚡ Turbo | {params}")
                _t0 = time.perf_counter()
                audio_tensor = current_model.generate(text, **params)
                gen_elapsed = time.perf_counter() - _t0
                audio_np = (audio_tensor.cpu().numpy()
                            if isinstance(audio_tensor, torch.Tensor)
                            else audio_tensor).astype(np.float32)
                self.sample_rate = 24000
                sum_speaker_names = [Path(ref_audio_file).stem] if ref_audio_file else []
                sum_unique_speakers = 1
                sum_segments = 1

            # ---- Chatterbox Multilingual ---------------------------------
            elif model_ui == "multilanguage":
                lang_key    = self.lang_var.get()
                language_id = self.LANGUAGES[lang_key]
                params = {
                    "audio_prompt_path":  ref_audio_path,
                    "exaggeration":       self.param_vars["exaggeration"].get(),
                    "cfg_weight":         self.param_vars["cfg_weight"].get(),
                    "temperature":        self.param_vars["temperature"].get(),
                    "repetition_penalty": self.param_vars["repetition_penalty"].get(),
                    "min_p":              self.param_vars["min_p"].get(),
                    "top_p":              self.param_vars["top_p"].get(),
                }
                _t0 = time.perf_counter()
                audio_tensor = current_model.generate(text, language_id=language_id, **params)
                gen_elapsed = time.perf_counter() - _t0
                audio_np = (audio_tensor.cpu().numpy()
                            if isinstance(audio_tensor, torch.Tensor)
                            else audio_tensor).astype(np.float32)
                self.sample_rate = 24000
                sum_speaker_names = [Path(ref_audio_file).stem] if ref_audio_file else []
                sum_unique_speakers = 1
                sum_segments = 1

            # ---- Chatterbox Classic (standard) ---------------------------
            else:
                params = {
                    "audio_prompt_path":  ref_audio_path,
                    "exaggeration":       self.param_vars["exaggeration"].get(),
                    "cfg_weight":         self.param_vars["cfg_weight"].get(),
                    "temperature":        self.param_vars["temperature"].get(),
                    "repetition_penalty": self.param_vars["repetition_penalty"].get(),
                    "min_p":              self.param_vars["min_p"].get(),
                    "top_p":              self.param_vars["top_p"].get(),
                }
                _t0 = time.perf_counter()
                audio_tensor = current_model.generate(text, **params)
                gen_elapsed = time.perf_counter() - _t0
                audio_np = (audio_tensor.cpu().numpy()
                            if isinstance(audio_tensor, torch.Tensor)
                            else audio_tensor).astype(np.float32)
                self.sample_rate = 24000
                sum_speaker_names = [Path(ref_audio_file).stem] if ref_audio_file else []
                sum_unique_speakers = 1
                sum_segments = 1

            # ---- Common post-processing ----------------------------------
            if audio_np.ndim == 2:
                audio_np = audio_np.squeeze()

            max_val = np.abs(audio_np).max()
            if max_val > 0:
                audio_np = audio_np / max_val * 0.95

            audio_int16 = (audio_np * 32767).astype(np.int16)
            self.current_audio = audio_int16

            if self.temp_audio_file and os.path.exists(self.temp_audio_file):
                os.remove(self.temp_audio_file)
            self.temp_audio_file = tempfile.mktemp(suffix='.wav')
            sf.write(self.temp_audio_file, audio_int16, self.sample_rate)

            self.audio_duration = len(audio_int16) / self.sample_rate

            self._print_generation_summary(
                speaker_names=sum_speaker_names if sum_speaker_names else ["(none)"],
                num_unique_speakers=sum_unique_speakers,
                num_segments=sum_segments,
                prefilling_tokens=sum_prefill,
                generated_tokens=sum_gen_tok,
                total_tokens=sum_total_tok,
                generation_seconds=gen_elapsed,
                audio_duration_seconds=self.audio_duration,
            )

            if self.update_timer:
                self.root.after_cancel(self.update_timer)
                self.update_timer = None

            self.root.after(0, lambda: self.timeline.config(to=self.audio_duration * 1000))
            self.timeline_var.set(0)
            self.playback_position = 0.0

            active_state = self.state_a if self.active_state == 1 else self.state_b
            active_state['audio']         = audio_int16
            active_state['temp_file']     = self.temp_audio_file
            active_state['text']          = text
            active_state['seed']          = self.seed_var.get()
            active_state['params']        = self._get_all_params()
            active_state['needs_refresh'] = False

            self.root.after(0, self._update_refresh_button_color)
            self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
            self._stop_playback()
            self._play_audio()

        except Exception as e:
            print(f"Error generating audio: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_generating = False
            self.root.after(0, lambda: self.refresh_btn.config(
                state=tk.NORMAL, text="⟳ REFRESH - Generate Audio"
            ))

    # ------------------------------------------------------------------ #
    # Playback                                                              #
    # ------------------------------------------------------------------ #

    def _on_play(self):
        if self.current_audio is None or not self.temp_audio_file:
            return
        if self.is_playing:
            return
        if self.playback_position >= self.audio_duration:
            self.playback_position = 0.0
            self.timeline_var.set(0)
        self._play_audio()

    def _on_pause(self):
        self._pause_playback()

    def _on_timeline_change(self, value):
        if self.current_audio is None or not self.temp_audio_file:
            return
        self.playback_position = float(value) / 1000.0
        if self.is_playing:
            self._stop_playback()
            self._play_audio()

    def _play_audio(self):
        if self.current_audio is None or not self.temp_audio_file:
            return
        if self.is_playing:
            self._stop_playback()
        self.is_playing = False
        self.external_player_process = None
        time.sleep(0.1)
        self.is_playing = True
        self.play_btn.config(state=tk.DISABLED, bg="#90EE90")
        self.pause_btn.config(state=tk.NORMAL, bg="#f0f0f0")

        try:
            import shutil
            if shutil.which('ffplay'):
                cmd = ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet']
                if self.playback_position > 0:
                    cmd.extend(['-ss', str(self.playback_position)])
                cmd.append(self.temp_audio_file)
                self.external_player_process = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                self._update_position()
                return
            elif shutil.which('aplay'):
                start_sample = int(self.playback_position * self.sample_rate)
                audio_from_pos = self.current_audio[start_sample:]
                partial_file = tempfile.mktemp(suffix='.wav')
                sf.write(partial_file, audio_from_pos, self.sample_rate)
                self.external_player_process = subprocess.Popen(
                    ['aplay', '-q', partial_file],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                self._update_position()
                return
        except Exception as e:
            print(f"External player error: {e}")

        self.is_playing = False
        self.play_btn.config(state=tk.NORMAL, bg="#f0f0f0")
        self.pause_btn.config(state=tk.DISABLED, bg="#f0f0f0")

    def _update_position(self):
        if not self.is_playing:
            return
        if self.external_player_process and self.external_player_process.poll() is not None:
            self.root.after(0, self._on_playback_finished)
            return
        self.playback_position += 0.1
        if self.playback_position >= self.audio_duration:
            self.root.after(0, self._on_playback_finished)
            return
        self.timeline_var.set(self.playback_position * 1000)
        self.update_timer = self.root.after(100, self._update_position)

    def _pause_playback(self):
        if not self.is_playing:
            return
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
            self.update_timer = None
        self.is_playing = False
        if self.external_player_process:
            try:
                self.external_player_process.terminate()
                self.external_player_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    self.external_player_process.kill()
                    self.external_player_process.wait(timeout=2)
                except Exception:
                    pass
            except Exception:
                try:
                    self.external_player_process.kill()
                except Exception:
                    pass
            finally:
                self.external_player_process = None
        self.play_btn.config(state=tk.NORMAL, bg="#f0f0f0")
        self.pause_btn.config(state=tk.DISABLED, bg="#f0f0f0")

    def _stop_playback(self):
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
            self.update_timer = None
        self.is_playing = False
        if self.external_player_process:
            try:
                self.external_player_process.terminate()
                self.external_player_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    self.external_player_process.kill()
                    self.external_player_process.wait(timeout=2)
                except Exception:
                    pass
            except Exception:
                try:
                    self.external_player_process.kill()
                except Exception:
                    pass
            finally:
                self.external_player_process = None
        self.play_btn.config(state=tk.NORMAL, bg="#f0f0f0")
        self.pause_btn.config(state=tk.DISABLED, bg="#f0f0f0")

    def _on_playback_finished(self):
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
            self.update_timer = None
        self.is_playing = False
        self.play_btn.config(state=tk.NORMAL, bg="#f0f0f0")
        self.pause_btn.config(state=tk.DISABLED, bg="#f0f0f0")
        if self.external_player_process:
            try:
                self.external_player_process.terminate()
                self.external_player_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    self.external_player_process.kill()
                    self.external_player_process.wait(timeout=1)
                except Exception:
                    pass
            except Exception:
                try:
                    self.external_player_process.kill()
                except Exception:
                    pass
            finally:
                self.external_player_process = None
        self.playback_position = self.audio_duration
        self.timeline_var.set(self.audio_duration * 1000)

    # ------------------------------------------------------------------ #
    # YAML export helpers                                                   #
    # ------------------------------------------------------------------ #

    def _build_yaml_params(self) -> str:
        """Return model-specific tts_params YAML block."""
        model_ui = self.model_var.get()

        if model_ui in _VIBEVOICE_UI_MODELS:
            use_sampling = "true" if self.use_sampling_var.get() else "false"
            return (
                f"tts_params:\n"
                f"        temperature: {self.param_vars['temperature'].get():.2f}\n"
                f"        temperature_max_deviation: 0\n"
                f"        top_p: {self.param_vars['top_p'].get():.2f}\n"
                f"        cfg_scale: {self.param_vars['cfg_scale'].get():.2f}\n"
                f"        cfg_scale_max_deviation: 0\n"
                f"        diffusion_steps: {int(self.param_vars['diffusion_steps'].get())}\n"
                f"        voice_speed_factor: {self.param_vars['voice_speed_factor'].get():.2f}\n"
                f"        use_sampling: {use_sampling}"
            )
        elif model_ui == "qwen3":
            return (
                f"tts_params:\n"
                f"        temperature: {self.param_vars['temperature'].get():.2f}\n"
                f"        temperature_max_deviation: 0.0\n"
                f"        repetition_penalty: {self.param_vars['repetition_penalty'].get():.2f}\n"
                f"        top_k: {int(self.param_vars['top_k'].get())}\n"
                f"        top_k_max_deviation: 0\n"
                f"        top_p: {self.param_vars['top_p'].get():.2f}\n"
                f"        subtalker_temperature: {self.param_vars['subtalker_temperature'].get():.2f}\n"
                f"        subtalker_temperature_max_deviation: 0.0\n"
                f"        subtalker_top_k: {int(self.param_vars['subtalker_top_k'].get())}\n"
                f"        subtalker_top_k_max_deviation: 0\n"
                f"        subtalker_top_p: {self.param_vars['subtalker_top_p'].get():.2f}"
            )
        elif model_ui == "turbo":
            return (
                f"tts_params:\n"
                f"        exaggeration: {self.param_vars['exaggeration'].get():.2f}\n"
                f"        exaggeration_max_deviation: 0.0\n"
                f"        cfg_weight: {self.param_vars['cfg_weight'].get():.2f}\n"
                f"        cfg_weight_max_deviation: 0.0\n"
                f"        temperature: {self.param_vars['temperature'].get():.2f}\n"
                f"        temperature_max_deviation: 0.0\n"
                f"        repetition_penalty: {self.param_vars['repetition_penalty'].get():.2f}\n"
                f"        min_p: {self.param_vars['min_p'].get():.2f}\n"
                f"        top_k: {int(self.param_vars['top_k'].get())}\n"
                f"        top_k_max_deviation: 0\n"
                f"        top_p: {self.param_vars['top_p'].get():.2f}"
            )
        else:
            # classic / multilanguage
            return (
                f"tts_params:\n"
                f"        exaggeration: {self.param_vars['exaggeration'].get():.2f}\n"
                f"        exaggeration_max_deviation: 0.0\n"
                f"        cfg_weight: {self.param_vars['cfg_weight'].get():.2f}\n"
                f"        cfg_weight_max_deviation: 0.0\n"
                f"        temperature: {self.param_vars['temperature'].get():.2f}\n"
                f"        temperature_max_deviation: 0.0\n"
                f"        repetition_penalty: {self.param_vars['repetition_penalty'].get():.2f}\n"
                f"        min_p: {self.param_vars['min_p'].get():.2f}\n"
                f"        top_p: {self.param_vars['top_p'].get():.2f}"
            )

    def _copy_settings(self):
        yaml_text = self._build_yaml_params()

        import shutil
        success = False
        if shutil.which('xclip'):
            try:
                process = subprocess.Popen(
                    ['xclip', '-selection', 'clipboard'],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                process.communicate(input=yaml_text.encode('utf-8'))
                success = process.returncode in (0, None)
                if success:
                    print("Copied using xclip")
            except Exception:
                pass

        if not success and shutil.which('xsel'):
            try:
                process = subprocess.Popen(
                    ['xsel', '--clipboard', '--input'],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                process.communicate(input=yaml_text.encode('utf-8'))
                success = process.returncode in (0, None)
                if success:
                    print("Copied using xsel")
            except Exception:
                pass

        if not success:
            try:
                self.root.clipboard_clear()
                self.root.update()
                self.root.clipboard_append(yaml_text)
                self.root.update()
                success = True
                print("Copied using Tkinter clipboard")
            except Exception:
                pass

        print(f"Copied YAML ({self.model_var.get()}):\n{yaml_text}")
        if not success:
            print("WARNING: Failed to copy. Install xclip: sudo apt-get install xclip")

        original_text = "📋 Copy YAML to Clipboard"
        self.copy_btn.config(text="✓ Copied!" if success else "✗ Clipboard error - see console")
        self.root.after(2000, lambda: self.copy_btn.config(text=original_text))

    # ------------------------------------------------------------------ #
    # Insert from Clipboard                                                 #
    # ------------------------------------------------------------------ #

    def _insert_from_clipboard(self):
        try:
            clipboard_text = self.root.clipboard_get()
            lines = clipboard_text.split('\n')
            cleaned_text = '\n'.join(line.lstrip() for line in lines if line.strip())

            import yaml
            data = yaml.safe_load(cleaned_text)
            if not isinstance(data, dict):
                raise ValueError("Not a valid YAML dict")

            tts_params = data.get('tts_params', data)
            if not isinstance(tts_params, dict):
                tts_params = data

            # All known params (all sliders)
            param_mapping = {row[0]: row[0] for row in _SLIDER_CONFIG}
            updated_items = []

            for yaml_key, ui_key in param_mapping.items():
                if yaml_key in tts_params:
                    try:
                        value = float(tts_params[yaml_key])
                        self.param_vars[ui_key].set(value)
                        self._format_label(ui_key)
                        updated_items.append(f"{ui_key}={value}")
                    except (ValueError, TypeError) as e:
                        print(f"[CLIPBOARD] Skipping {yaml_key}: {e}")

            if "use_sampling" in tts_params:
                self.use_sampling_var.set(bool(tts_params["use_sampling"]))
                updated_items.append(f"use_sampling={self.use_sampling_var.get()}")

            if 'reference_audio' in data:
                try:
                    ref = str(data['reference_audio'])
                    if '/' in ref:
                        ref = ref.split('/')[-1]
                    if ref in self.reference_audio_files:
                        self.ref_audio_var.set(ref)
                        updated_items.append(f"ref_audio={ref}")
                except Exception as e:
                    print(f"[CLIPBOARD] reference_audio error: {e}")

            if 'language' in data:
                try:
                    lang_code = str(data['language'])
                    model_ui  = self.model_var.get()
                    lang_table = self.QWEN3_LANGUAGES if model_ui == "qwen3" else self.LANGUAGES
                    # Match by code (value) or display name (key)
                    for display, val in lang_table.items():
                        code = val if model_ui != "qwen3" else display.split("(")[-1].rstrip(")")
                        if lang_code in (code, display, val):
                            self.lang_var.set(display)
                            updated_items.append(f"language={display}")
                            break
                except Exception as e:
                    print(f"[CLIPBOARD] language error: {e}")

            if updated_items:
                original_text = "📥 Insert from Clipboard"
                self.insert_btn.config(text=f"✓ Loaded {len(updated_items)} items!")
                self._on_text_change()
                self._mark_needs_refresh()
                self.root.after(2000, lambda: self.insert_btn.config(text=original_text))
            else:
                raise ValueError("No valid parameters found")

        except tk.TclError:
            self.insert_btn.config(text="✗ Clipboard is empty")
            self.root.after(2000, lambda: self.insert_btn.config(text="📥 Insert from Clipboard"))
        except Exception as e:
            self.insert_btn.config(text="✗ Error loading parameters")
            self.root.after(2000, lambda: self.insert_btn.config(text="📥 Insert from Clipboard"))
            print(f"[CLIPBOARD] Error: {e}")

    # ------------------------------------------------------------------ #
    # Save audio                                                            #
    # ------------------------------------------------------------------ #

    def _save_audio(self):
        if self.current_audio is None or not self.temp_audio_file:
            return

        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ref_name  = Path(self.ref_audio_var.get()).stem
        base_name = f"test_{ref_name}_{timestamp}"
        output_path = output_dir / f"{base_name}.wav"
        yaml_path   = output_dir / f"{base_name}.yaml"

        import shutil as _shutil
        _shutil.copy2(self.temp_audio_file, output_path)

        seed = int(self.seed_var.get()) if self.seed_var.get().isdigit() else 12345
        model_ui = self.model_var.get()

        header = (
            f"# Chatterbox TTS Test - {timestamp}\n"
            f"# Reference Audio: {self.ref_audio_var.get()}\n"
            f"# Model: {model_ui}\n"
        )
        if model_ui == "multilanguage":
            lang_code = self.LANGUAGES.get(self.lang_var.get(), 'en')
            header += f"# Language: {self.lang_var.get()} ({lang_code})\n"
        elif model_ui == "qwen3":
            lang_name = self.QWEN3_LANGUAGES.get(self.lang_var.get(), 'English')
            header += f"# Language: {self.lang_var.get()} → {lang_name}\n"
            txt_path  = self._get_ref_text_path()
            header += f"# ref_text: {txt_path.name if txt_path else 'none (x_vector only)'}\n"
        elif model_ui in _VIBEVOICE_UI_MODELS:
            header += "# Language: auto-detected by VibeVoice\n"
            header += f"# use_sampling: {self.use_sampling_var.get()}\n"
            for spk_i, var in (
                (2, self.ref_audio_var_2),
                (3, self.ref_audio_var_3),
                (4, self.ref_audio_var_4),
            ):
                extra = var.get().strip()
                if extra:
                    header += f"# Reference Audio (speaker {spk_i}): {extra}\n"

        header += f"# Seed: {seed}\n\n"
        header += f"# Text:\n# {self.text_widget.get('1.0', tk.END).strip()}\n\n"
        header += self._build_yaml_params() + "\n"

        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(header)

        self.save_btn.config(text="✓ Saved!")
        print(f"Audio saved to: {output_path}")
        print(f"Metadata saved to: {yaml_path}")
        self.root.after(2000, lambda: self.save_btn.config(text="💾 Save Audio"))

    # ------------------------------------------------------------------ #
    # Cleanup                                                               #
    # ------------------------------------------------------------------ #

    def cleanup(self):
        self._stop_playback()
        if self.external_player_process:
            try:
                self.external_player_process.terminate()
                self.external_player_process.wait(timeout=2)
            except Exception:
                try:
                    self.external_player_process.kill()
                except Exception:
                    pass
            self.external_player_process = None
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
            self.update_timer = None
        self._save_settings()
        if self.temp_audio_file and os.path.exists(self.temp_audio_file):
            try:
                os.remove(self.temp_audio_file)
            except Exception:
                pass


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("""
Chatterbox TTS Parameter Testing Tool

Usage: python chatterbox_tester.py

A GUI tool for quickly testing different TTS parameters, models,
and reference audio files to find optimal voice settings.

Supported models:
  classic        ChatterboxTTS (English only)
  multilanguage  ChatterboxMultilingualTTS (24 languages)
  turbo          ChatterboxTurboTTS (English only; adds top_k 0-2000)
  qwen3          Qwen3-TTS-12Hz-1.7B-Base voice cloning (10 languages)
                 Place a .txt transcript next to your .wav reference audio
                 for best cloning quality (ICL mode).
  vibevoice      VibeVoice-Large-Q8 (long-form + voice cloning)

For better audio playback in WSL, install ffmpeg:
  sudo apt-get install ffmpeg

For detailed documentation, see CHATTERBOX_TESTER.md
        """)
        sys.exit(0)

    try:
        root = tk.Tk()
        app  = ChatterboxTester(root)

        def on_closing():
            app.cleanup()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
    except tk.TclError as e:
        print(f"Error: Cannot start GUI - {e}")
        print("This tool requires a graphical display environment.")
        print("If running via SSH, use X11 forwarding: ssh -X")
        sys.exit(1)


if __name__ == "__main__":
    main()
