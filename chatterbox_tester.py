#!/usr/bin/env python3
"""
Chatterbox TTS Parameter Testing Tool (pygame version)

A lightweight GUI tool for quickly testing different Chatterbox parameters,
models, and reference audio files to find optimal voice settings.

This version uses pygame for audio playback (better WSL compatibility).
"""

import tkinter as tk
from tkinter import ttk
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
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


class ChatterboxTester:
    """Main application class for Chatterbox parameter testing."""
    
    # Available languages for multilingual model
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
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Chatterbox TTS Tester")
        
        # Set initial size (position will be set after loading settings)
        self.root.geometry("500x900")
        
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
            'audio': None,
            'temp_file': None,
            'text': '',
            'seed': '12345',
            'params': {},
            'needs_refresh': True
        }
        self.state_b: Dict[str, Any] = {
            'audio': None,
            'temp_file': None,
            'text': '',
            'seed': '12345',
            'params': {},
            'needs_refresh': True
        }
        
        # Track if current UI matches rendered audio
        self.needs_refresh: bool = True
        
        # Model and generation state
        self.device = self._detect_device()
        self.current_model = None
        self.current_model_type = "standard"
        
        # Generation state
        self.is_generating: bool = False
        
        # UI state
        self.reference_audio_files = self._get_reference_audio_files()
        
        # Settings file path
        self.settings_file = Path(__file__).parent / ".chatterbox_tester_settings.json"
        
        # Presets file path and storage
        self.presets_file = Path(__file__).parent / ".chatterbox_tester_presets.yaml"
        self.presets: Dict[str, Dict[str, float]] = {}  # reference_audio -> params
        
        # History for undo/redo
        self.history: list[Dict[str, Any]] = []
        self.history_position: int = -1
        self.is_restoring_state: bool = False  # Flag to prevent history recording during restore
        self._slider_save_timer: Optional[str] = None  # Timer for delayed slider history save
        self._text_save_timer: Optional[str] = None  # Timer for delayed text history save
        
        self._build_ui()
        self._load_settings()  # Load saved settings first
        self._load_presets()  # Load TTS parameter presets
        self._load_initial_model()
        self._update_language_dropdown_state()  # Set initial state of language dropdown
        self._update_load_button_state()  # Set initial state of Load preset button
        
        # Keyboard shortcuts for undo/redo
        self.root.bind('<Control-z>', lambda e: self._undo())
        self.root.bind('<Control-y>', lambda e: self._redo())
        self.root.bind('<Control-Z>', lambda e: self._undo())  # Also support Shift variant
        self.root.bind('<Control-Y>', lambda e: self._redo())
        
        # Save initial state to history after UI is built and settings loaded
        self._save_state_to_history()
        
        # Set initial refresh button color
        self._update_refresh_button_color()
        
    def _detect_device(self) -> str:
        """Detect available device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _get_reference_audio_files(self) -> list:
        """Get list of WAV files from reference audio directory."""
        ref_audio_dir = Path(__file__).parent / "data" / "input" / "reference_audio"
        if not ref_audio_dir.exists():
            return []
        
        wav_files = sorted([f.name for f in ref_audio_dir.glob("*.wav")])
        return wav_files
    
    def _refresh_reference_audio_list(self):
        """Refresh the reference audio file list."""
        # Get current selection
        current_selection = self.ref_audio_var.get()
        
        # Reload the file list
        self.reference_audio_files = self._get_reference_audio_files()
        
        # Update dropdown values
        self.ref_dropdown.config(values=self.reference_audio_files)
        
        # Restore selection if still valid, otherwise select first item
        if current_selection in self.reference_audio_files:
            self.ref_audio_var.set(current_selection)
        elif self.reference_audio_files:
            self.ref_audio_var.set(self.reference_audio_files[0])
        
        # Update Load button state
        self._update_load_button_state()
        
        print(f"Reference audio list refreshed: {len(self.reference_audio_files)} files found")
    
    def _on_load_preset(self):
        """Load preset parameters for current reference audio."""
        current_ref_audio = self.ref_audio_var.get()
        if not current_ref_audio or current_ref_audio not in self.presets:
            return
        
        try:
            preset = self.presets[current_ref_audio]
            
            # Temporarily disable history recording during preset load
            self.is_restoring_state = True
            
            # Load parameters into sliders
            for param_name, value in preset.items():
                if param_name in self.param_vars:
                    self.param_vars[param_name].set(value)
                    self.param_labels[param_name].config(text=f"{value:.2f}")
            
            self.is_restoring_state = False
            
            # Mark as needs refresh since parameters changed
            self._mark_needs_refresh()
            
            # Save to history after loading preset
            self._save_state_to_history()
            
            # Visual feedback
            original_text = "Load"
            self.load_preset_btn.config(text="✓ Loaded")
            print(f"Loaded preset for '{current_ref_audio}'")
            self.root.after(1500, lambda: self.load_preset_btn.config(text=original_text))
            
        except Exception as e:
            print(f"Error loading preset: {e}")
    
    def _on_save_preset(self):
        """Save current parameters as preset."""
        self._save_preset()
        
        # Visual feedback
        original_text = "Save"
        self.save_preset_btn.config(text="✓ Saved")
        self.root.after(1500, lambda: self.save_preset_btn.config(text=original_text))
    
    def _build_ui(self):
        """Build the user interface."""
        # Reference Audio File Dropdown with Load/Save/Refresh Buttons
        ref_frame = tk.Frame(self.root)
        ref_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.ref_audio_var = tk.StringVar(
            value=self.reference_audio_files[0] if self.reference_audio_files else ""
        )
        self.ref_dropdown = ttk.Combobox(
            ref_frame,
            textvariable=self.ref_audio_var,
            values=self.reference_audio_files,
            state="readonly",
            font=("Arial", 12)
        )
        self.ref_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.ref_dropdown.bind('<<ComboboxSelected>>', lambda e: (self._on_text_change(), self._mark_needs_refresh(), self._update_load_button_state()))
        
        # Refresh button for reference audio files
        ref_refresh_btn = tk.Button(
            ref_frame,
            text="⭯",
            font=("Arial", 9),
            command=self._refresh_reference_audio_list,
            bg="#f0f0f0",
            width=3,
            cursor="hand2"
        )
        ref_refresh_btn.pack(side=tk.LEFT, padx=2)

        # Load preset button
        self.load_preset_btn = tk.Button(
            ref_frame,
            text="Load",
            font=("Arial", 9),
            command=self._on_load_preset,
            bg="#f0f0f0",
            state=tk.DISABLED,
            cursor="hand2"
        )
        self.load_preset_btn.pack(side=tk.LEFT, padx=2)
        
        # Save preset button
        self.save_preset_btn = tk.Button(
            ref_frame,
            text="Save",
            font=("Arial", 9),
            command=self._on_save_preset,
            bg="#f0f0f0",
            cursor="hand2"
        )
        self.save_preset_btn.pack(side=tk.LEFT, padx=2)
        
        
        # Model Selection (Radio Buttons)
        model_frame = tk.Frame(self.root)
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.model_var = tk.StringVar(value="multilanguage")
        
        tk.Radiobutton(
            model_frame,
            text="classic",
            variable=self.model_var,
            value="classic",
            font=("Arial", 12),
            command=self._on_model_change
        ).pack(side=tk.LEFT, padx=20)
        
        tk.Radiobutton(
            model_frame,
            text="multilanguage",
            variable=self.model_var,
            value="multilanguage",
            font=("Arial", 12),
            command=self._on_model_change
        ).pack(side=tk.LEFT, padx=20)
        
        # Language Dropdown (for multilanguage model)
        self.lang_var = tk.StringVar(value="English (en)")
        self.lang_dropdown = ttk.Combobox(
            self.root,
            textvariable=self.lang_var,
            values=list(self.LANGUAGES.keys()),
            state="readonly",
            font=("Arial", 12)
        )
        self.lang_dropdown.pack(fill=tk.X, padx=10, pady=5)
        self.lang_dropdown.bind('<<ComboboxSelected>>', lambda e: (self._on_text_change(), self._mark_needs_refresh()))
        
        # Seed Input
        seed_frame = tk.Frame(self.root)
        seed_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(seed_frame, text="Seed:", font=("Arial", 10)).pack(side=tk.LEFT)
        
        self.seed_var = tk.StringVar(value="12345")
        seed_entry = tk.Entry(
            seed_frame,
            textvariable=self.seed_var,
            font=("Arial", 11),
            width=15
        )
        seed_entry.pack(side=tk.LEFT, padx=5)
        seed_entry.bind('<FocusOut>', lambda e: (self._on_text_change(), self._mark_needs_refresh()))
        seed_entry.bind('<Return>', lambda e: (self._on_text_change(), self._mark_needs_refresh()))
        
        # Random seed button
        def randomize_seed():
            import random
            self.seed_var.set(str(random.randint(1, 999999)))
            self._on_text_change()
            self._mark_needs_refresh()
        
        tk.Button(
            seed_frame,
            text="🎲 Random",
            command=randomize_seed,
            font=("Arial", 9)
        ).pack(side=tk.LEFT, padx=5)
        
        # Text Input
        text_label = tk.Label(self.root, text="Text to speak (max. 500 characters):", font=("Arial", 10))
        text_label.pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        text_frame = tk.Frame(self.root)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Arial", 11), height=6)
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        text_scrollbar = tk.Scrollbar(text_frame, command=self.text_widget.yview)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget.config(yscrollcommand=text_scrollbar.set)
        
        # Bind text changes to history (with delayed save to batch with other changes)
        self.text_widget.bind('<FocusOut>', lambda e: self._on_text_change())
        # Track text changes in real-time for needs_refresh AND idle-based history save
        self.text_widget.bind('<KeyRelease>', lambda e: self._on_text_keypress())
        
        # Bind Ctrl+A to select all text
        self.text_widget.bind('<Control-a>', self._text_select_all)
        self.text_widget.bind('<Control-A>', self._text_select_all)
        
        # Bind Ctrl+V to paste with replace behavior
        self.text_widget.bind('<Control-v>', self._text_paste_replace)
        self.text_widget.bind('<Control-V>', self._text_paste_replace)
        
        # Default text
        default_text = "Ibus que re endundaectis aut que pro blaborios mil is plab illam, voluptur arum fugias aut eos ditae idendi sed ut faccull oruptas ne net fuga. Omnimus voluptae nonsectur se liquas nestia esti dicia cuptatquia seque plis pere eos"
        self.text_widget.insert("1.0", default_text)
        
        char_count_frame = tk.Frame(self.root)
        char_count_frame.pack(fill=tk.X, padx=10)
        self.char_count_label = tk.Label(
            char_count_frame,
            text=f"max. 500 characters",
            font=("Arial", 9),
            fg="gray"
        )
        self.char_count_label.pack(anchor=tk.W)
        
        # A/B State Tabs (moved here, above refresh button)
        state_tabs_frame = tk.Frame(self.root, bg="#e0e0e0")
        state_tabs_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        
        self.state_a_btn = tk.Button(
            state_tabs_frame,
            text="1",
            font=("Arial", 14, "bold"),
            command=lambda: self._switch_state(1),
            bg="#ffffff",
            fg="#000000",
            relief=tk.SUNKEN,
            borderwidth=2,
            width=8,
            cursor="hand2"
        )
        self.state_a_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Copy A to B button
        self.copy_a_to_b_btn = tk.Button(
            state_tabs_frame,
            text="→",
            font=("Arial", 12, "bold"),
            command=self._copy_a_to_b,
            bg="#d0d0d0",
            fg="#000000",
            relief=tk.RAISED,
            borderwidth=1,
            width=2,
            cursor="hand2"
        )
        self.copy_a_to_b_btn.pack(side=tk.LEFT, padx=2)
        
        # Copy B to A button
        self.copy_b_to_a_btn = tk.Button(
            state_tabs_frame,
            text="←",
            font=("Arial", 12, "bold"),
            command=self._copy_b_to_a,
            bg="#d0d0d0",
            fg="#000000",
            relief=tk.RAISED,
            borderwidth=1,
            width=2,
            cursor="hand2"
        )
        self.copy_b_to_a_btn.pack(side=tk.LEFT, padx=2)
        
        self.state_b_btn = tk.Button(
            state_tabs_frame,
            text="2",
            font=("Arial", 14, "bold"),
            command=lambda: self._switch_state(2),
            bg="#e0e0e0",
            fg="#000000",
            relief=tk.RAISED,
            borderwidth=2,
            width=8,
            cursor="hand2"
        )
        self.state_b_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Refresh Button
        refresh_frame = tk.Frame(self.root)
        refresh_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.refresh_btn = tk.Button(
            refresh_frame,
            text="⟳ REFRESH - Generate Audio",
            font=("", 14, "bold"),
            command=self._on_refresh,
            bg="#4CAF50",  # Bright green by default (needs refresh)
            fg="white",
            activebackground="#45a049",
            relief=tk.RAISED,
            borderwidth=2,
            cursor="hand2"
        )
        self.refresh_btn.pack(fill=tk.X, ipady=8)
        
        # Parameters Section
        params_frame = tk.Frame(self.root)
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Slider configurations: (name, min, max, default, resolution)
        sliders_config = [
            ("exaggeration", 0.0, 2.0, 0.7, 0.01),
            ("cfg_weight", 0.0, 1.5, 0.45, 0.01),
            ("temperature", 0.0, 2.0, 0.8, 0.01),
            ("repetition_penalty", 1.0, 3.0, 2.2, 0.01),
            ("min_p", 0.01, 0.5, 0.05, 0.01),
            ("top_p", 0.5, 1.0, 0.98, 0.01),
        ]
        
        self.param_vars = {}
        self.param_labels = {}
        
        for name, min_val, max_val, default, resolution in sliders_config:
            slider_frame = tk.Frame(params_frame)
            slider_frame.pack(fill=tk.X, pady=3)
            
            # Label with current value
            label_text = tk.Label(
                slider_frame,
                text=name,
                font=("Arial", 10),
                width=20,
                anchor=tk.W
            )
            label_text.pack(side=tk.LEFT)
            
            var = tk.DoubleVar(value=default)
            self.param_vars[name] = var
            
            slider = tk.Scale(
                slider_frame,
                from_=min_val,
                to=max_val,
                resolution=resolution,
                orient=tk.HORIZONTAL,
                variable=var,
                showvalue=False,
                command=lambda v, n=name: self._on_slider_change(n, v)
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            value_label = tk.Label(
                slider_frame,
                text=f"{default:.2f}",
                font=("Arial", 10),
                width=6,
                anchor=tk.E,
                relief=tk.SUNKEN,
                borderwidth=1
            )
            value_label.pack(side=tk.RIGHT)
            self.param_labels[name] = value_label
        
        # Timeline / Scrubber
        timeline_frame = tk.Frame(self.root)
        timeline_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.timeline_var = tk.DoubleVar(value=0)
        self.timeline = tk.Scale(
            timeline_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.timeline_var,
            showvalue=False,
            command=self._on_timeline_change
        )
        self.timeline.pack(fill=tk.X)
        
        # Playback Controls
        controls_frame = tk.Frame(self.root)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.play_btn = tk.Button(
            controls_frame,
            text="▶",
            font=("Arial", 20),
            width=3,
            command=self._on_play,
            bg="#f0f0f0"
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.pause_btn = tk.Button(
            controls_frame,
            text="⏸",
            font=("Arial", 20),
            width=3,
            command=self._on_pause,
            state=tk.DISABLED,
            bg="#f0f0f0"
        )
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        # Undo/Redo buttons (right-aligned)
        self.redo_btn = tk.Button(
            controls_frame,
            text="↷",
            font=("Arial", 16),
            width=3,
            command=self._redo,
            bg="#f0f0f0",
            state=tk.DISABLED
        )
        self.redo_btn.pack(side=tk.RIGHT, padx=5)
        
        self.undo_btn = tk.Button(
            controls_frame,
            text="↶",
            font=("Arial", 16),
            width=3,
            command=self._undo,
            bg="#f0f0f0",
            state=tk.DISABLED
        )
        self.undo_btn.pack(side=tk.RIGHT)
        
        # Action Buttons Frame
        action_frame = tk.Frame(self.root)
        action_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Copy Settings Button
        self.copy_btn = tk.Button(
            action_frame,
            text="📋 Copy YAML to Clipboard",
            font=("Arial", 11),
            command=self._copy_settings,
            bg="#f0f0f0"
        )
        self.copy_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Save Audio Button
        self.save_btn = tk.Button(
            action_frame,
            text="💾 Save Audio",
            font=("Arial", 11),
            command=self._save_audio,
            bg="#f0f0f0",
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # Insert from Clipboard Button
        insert_frame = tk.Frame(self.root)
        insert_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.insert_btn = tk.Button(
            insert_frame,
            text="📥 Insert from Clipboard",
            font=("Arial", 11),
            command=self._insert_from_clipboard,
            bg="#f0f0f0"
        )
        self.insert_btn.pack(fill=tk.X)
    
    def _load_settings(self):
        """Load saved settings from file."""
        if not self.settings_file.exists():
            self._center_window()
            return
        
        try:
            import json
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            # Restore reference audio
            if settings.get('reference_audio') and settings['reference_audio'] in self.reference_audio_files:
                self.ref_audio_var.set(settings['reference_audio'])
            
            # Restore model type
            if settings.get('model_type'):
                self.model_var.set(settings['model_type'])
                # Update language dropdown state when model is restored
                self._update_language_dropdown_state()
            
            # Restore language
            if settings.get('language'):
                self.lang_var.set(settings['language'])
            
            # Restore active state
            if settings.get('active_state'):
                self.active_state = settings['active_state']
            
            # Restore state A
            if settings.get('state_a'):
                state_a_data = settings['state_a']
                self.state_a['text'] = state_a_data.get('text', '')
                self.state_a['seed'] = state_a_data.get('seed', '12345')
                self.state_a['params'] = state_a_data.get('params', {})
                self.state_a['needs_refresh'] = state_a_data.get('needs_refresh', True)
                # Note: audio and temp_file are not persisted
            
            # Restore state B
            if settings.get('state_b'):
                state_b_data = settings['state_b']
                self.state_b['text'] = state_b_data.get('text', '')
                self.state_b['seed'] = state_b_data.get('seed', '12345')
                self.state_b['params'] = state_b_data.get('params', {})
                self.state_b['needs_refresh'] = state_b_data.get('needs_refresh', True)
                # Note: audio and temp_file are not persisted
            
            # Load active state into UI
            active_state = self.state_a if self.active_state == 1 else self.state_b
            
            # Restore seed
            self.seed_var.set(str(active_state.get('seed', '12345')))
            
            # Restore text
            text = active_state.get('text', '')
            if text:
                self.text_widget.delete("1.0", tk.END)
                self.text_widget.insert("1.0", text)
            
            # Restore parameters
            params = active_state.get('params', {})
            for param_name, value in params.items():
                if param_name in self.param_vars:
                    self.param_vars[param_name].set(value)
                    self.param_labels[param_name].config(text=f"{value:.2f}")
            
            # Update state buttons
            if self.active_state == 1:
                self.state_a_btn.config(relief=tk.SUNKEN, bg="#ffffff")
                self.state_b_btn.config(relief=tk.RAISED, bg="#e0e0e0")
            else:
                self.state_a_btn.config(relief=tk.RAISED, bg="#e0e0e0")
                self.state_b_btn.config(relief=tk.SUNKEN, bg="#ffffff")
            
            # Restore window position
            if settings.get('window_x') is not None and settings.get('window_y') is not None:
                x = settings['window_x']
                y = settings['window_y']
                # Ensure window is on screen (basic validation)
                if x >= -100 and y >= -100 and x < 3000 and y < 3000:
                    self.root.geometry(f"500x900+{x}+{y}")
                    print(f"Settings loaded successfully (Active state: {self.active_state}, Position: {x},{y})")
                else:
                    self._center_window()
                    print(f"Settings loaded successfully (Active state: {self.active_state}, Position: centered)")
            else:
                self._center_window()
                print(f"Settings loaded successfully (Active state: {self.active_state}, Position: centered)")
        except Exception as e:
            print(f"Could not load settings: {e}")
            self._center_window()
    
    def _save_settings(self):
        """Save current settings to file."""
        try:
            import json
            
            # Save current UI to active state first
            current_state = self.state_a if self.active_state == 1 else self.state_b
            current_ui = self._get_current_ui_state()
            current_state['text'] = current_ui['text']
            current_state['seed'] = current_ui['seed']
            current_state['params'] = current_ui['params']
            
            # Get current window position
            self.root.update_idletasks()  # Ensure geometry is current
            geometry = self.root.geometry()
            # Parse geometry string: "widthxheight+x+y"
            parts = geometry.split('+')
            window_x = int(parts[1]) if len(parts) > 1 else None
            window_y = int(parts[2]) if len(parts) > 2 else None
            
            settings = {
                'reference_audio': self.ref_audio_var.get(),
                'model_type': self.model_var.get(),
                'language': self.lang_var.get(),
                'active_state': self.active_state,
                'window_x': window_x,
                'window_y': window_y,
                'state_a': {
                    'text': self.state_a.get('text', ''),
                    'seed': self.state_a.get('seed', '12345'),
                    'params': self.state_a.get('params', {}),
                    'needs_refresh': self.state_a.get('needs_refresh', True)
                    # Note: audio and temp_file are not persisted
                },
                'state_b': {
                    'text': self.state_b.get('text', ''),
                    'seed': self.state_b.get('seed', '12345'),
                    'params': self.state_b.get('params', {}),
                    'needs_refresh': self.state_b.get('needs_refresh', True)
                    # Note: audio and temp_file are not persisted
                }
            }
            
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"Could not save settings: {e}")
    
    def _center_window(self):
        """Center the window on the screen."""
        self.root.update_idletasks()
        
        # Get window size
        window_width = 500
        window_height = 900
        
        # Get screen size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate center position
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        # Set geometry
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    def _load_presets(self):
        """Load TTS parameter presets from YAML file."""
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
        """Save current TTS parameters as preset for current reference audio."""
        try:
            import yaml
            
            # Get current reference audio filename
            current_ref_audio = self.ref_audio_var.get()
            if not current_ref_audio:
                return
            
            # Get current TTS parameters (only the 6 preset params)
            preset_params = {
                'exaggeration': float(self.param_vars['exaggeration'].get()),
                'cfg_weight': float(self.param_vars['cfg_weight'].get()),
                'temperature': float(self.param_vars['temperature'].get()),
                'repetition_penalty': float(self.param_vars['repetition_penalty'].get()),
                'min_p': float(self.param_vars['min_p'].get()),
                'top_p': float(self.param_vars['top_p'].get()),
            }
            
            # Update presets dictionary
            self.presets[current_ref_audio] = preset_params
            
            # Save to file (sorted by key for better readability)
            sorted_presets = dict(sorted(self.presets.items()))
            with open(self.presets_file, 'w', encoding='utf-8') as f:
                yaml.dump(sorted_presets, f, default_flow_style=False, sort_keys=False)
            
            print(f"Saved preset for '{current_ref_audio}'")
            
            # Update Load button state
            self._update_load_button_state()
            
        except Exception as e:
            print(f"Could not save preset: {e}")
    
    def _update_load_button_state(self):
        """Enable/disable Load button based on whether preset exists for current audio."""
        current_ref_audio = self.ref_audio_var.get()
        if current_ref_audio and current_ref_audio in self.presets:
            self.load_preset_btn.config(state=tk.NORMAL)
        else:
            self.load_preset_btn.config(state=tk.DISABLED)
    
    def _load_initial_model(self):
        """Load the initial model."""
        model_type = self.model_var.get()
        self._load_model("multilingual" if model_type == "multilanguage" else "classic")
    
    def _load_model(self, model_type: str):
        """Load a Chatterbox model."""
        model_name = "standard" if model_type == "classic" else "multilingual"
        self.current_model = ChatterboxModelCache.get_model(self.device, model_name)
        self.current_model_type = model_name
    
    def _update_language_dropdown_state(self):
        """Update language dropdown state based on model type."""
        model_type = self.model_var.get()
        if model_type == "multilanguage":
            self.lang_dropdown.config(state="readonly")
        else:
            self.lang_dropdown.config(state="disabled")
    
    def _on_model_change(self):
        """Handle model selection change."""
        model_type = self.model_var.get()
        
        # Enable/disable language dropdown
        self._update_language_dropdown_state()
        
        # Load new model
        self._load_model(model_type)
        
        # Save to history and mark needs refresh (with delay)
        self._on_text_change()
        self._mark_needs_refresh()
    
    def _text_select_all(self, event):
        """Select all text in the text widget (Ctrl+A)."""
        self.text_widget.tag_add(tk.SEL, "1.0", tk.END)
        self.text_widget.mark_set(tk.INSERT, "1.0")
        self.text_widget.see(tk.INSERT)
        return 'break'  # Prevent default behavior
    
    def _text_paste_replace(self, event):
        """Paste clipboard content, replacing selected text (Ctrl+V)."""
        try:
            # Check if there's a selection
            if self.text_widget.tag_ranges(tk.SEL):
                # Delete selected text
                self.text_widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
            
            # Get clipboard content
            clipboard_text = self.root.clipboard_get()
            
            # Insert at current cursor position
            self.text_widget.insert(tk.INSERT, clipboard_text)
            
            # Save to history after paste (with delay)
            self._on_text_change()
            self._mark_needs_refresh()
            
        except tk.TclError:
            # Clipboard is empty or unavailable
            pass
        
        return 'break'  # Prevent default behavior
    
    def _on_text_keypress(self):
        """Handle text keypress - mark needs refresh and schedule history save after idle."""
        # Mark needs refresh immediately for button color update
        self._mark_needs_refresh()
        
        # Cancel any existing text save timer
        if hasattr(self, '_text_save_timer') and self._text_save_timer:
            self.root.after_cancel(self._text_save_timer)
        
        # Schedule save after 1000ms idle time (1 second after last keypress)
        self._text_save_timer = self.root.after(1000, self._save_text_after_idle)
    
    def _save_text_after_idle(self):
        """Save history after text typing idle period."""
        self._save_state_to_history()
    
    def _on_text_change(self):
        """Handle text change with delayed history save to batch with other UI changes."""
        # Cancel any existing text save timer
        if hasattr(self, '_text_save_timer') and self._text_save_timer:
            self.root.after_cancel(self._text_save_timer)
        
        # Schedule save after 500ms delay (allows batching with slider changes)
        self._text_save_timer = self.root.after(500, self._save_state_to_history)
    
    def _update_char_count(self):
        """Update character count display."""
        text = self.text_widget.get("1.0", tk.END).strip()
        
        # Enforce 500 character limit
        if len(text) > 500:
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.insert("1.0", text[:500])
            text = text[:500]
        
        # Update character count
        char_count = len(text)
        self.char_count_label.config(text=f"{char_count}/500 characters")
    
    def _update_slider_label(self, param_name: str):
        """Update slider value label."""
        value = self.param_vars[param_name].get()
        self.param_labels[param_name].config(text=f"{value:.2f}")
    
    def _on_slider_change(self, param_name: str, value: str):
        """Handle slider change with history tracking."""
        self._update_slider_label(param_name)
        # Mark needs refresh immediately
        self._mark_needs_refresh()
        # Only save to history when slider is released (on every change would be too much)
        # We'll use a delayed save approach
        if hasattr(self, '_slider_save_timer') and self._slider_save_timer:
            self.root.after_cancel(self._slider_save_timer)
        self._slider_save_timer = self.root.after(300, self._save_state_to_history)
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get current UI state as dictionary."""
        return {
            'reference_audio': self.ref_audio_var.get(),
            'model_type': self.model_var.get(),
            'language': self.lang_var.get(),
            'seed': self.seed_var.get(),
            'text': self.text_widget.get("1.0", tk.END).strip(),
            'params': {
                'exaggeration': self.param_vars['exaggeration'].get(),
                'cfg_weight': self.param_vars['cfg_weight'].get(),
                'temperature': self.param_vars['temperature'].get(),
                'repetition_penalty': self.param_vars['repetition_penalty'].get(),
                'min_p': self.param_vars['min_p'].get(),
                'top_p': self.param_vars['top_p'].get(),
            }
        }
    
    def _save_state_to_history(self, force: bool = False):
        """Save current state to history.
        
        Args:
            force: If True, save even if state hasn't changed
        """
        # Don't save during restore operation
        if self.is_restoring_state:
            return
        
        current_state = self._get_current_state()
        
        # Check if state is different from last saved state (unless forced)
        if not force and self.history and self.history_position >= 0:
            last_state = self.history[self.history_position]
            if current_state == last_state:
                return  # No change, don't save
        
        # Remove any states after current position (when user made changes after undo)
        if self.history_position < len(self.history) - 1:
            removed_count = len(self.history) - self.history_position - 1
            self.history = self.history[:self.history_position + 1]
        
        # Add new state
        self.history.append(current_state)
        self.history_position = len(self.history) - 1
        
        # Limit history size to 50 states
        if len(self.history) > 50:
            self.history.pop(0)
            self.history_position -= 1
        
        self._update_history_buttons()
    
    def _restore_state(self, state: Dict[str, Any]):
        """Restore UI state from dictionary."""
        self.is_restoring_state = True
        
        try:
            # Restore reference audio
            if state.get('reference_audio') and state['reference_audio'] in self.reference_audio_files:
                self.ref_audio_var.set(state['reference_audio'])
            
            # Restore model type
            if state.get('model_type'):
                old_model = self.model_var.get()
                self.model_var.set(state['model_type'])
                # Only reload model if it changed
                if old_model != state['model_type']:
                    self._load_model(state['model_type'])
                    self._update_language_dropdown_state()
            
            # Restore language
            if state.get('language'):
                self.lang_var.set(state['language'])
            
            # Restore seed
            if state.get('seed'):
                self.seed_var.set(str(state['seed']))
            
            # Restore text
            if state.get('text') is not None:
                self.text_widget.delete("1.0", tk.END)
                self.text_widget.insert("1.0", state['text'])
            
            # Restore parameters
            params = state.get('params', {})
            for param_name, value in params.items():
                if param_name in self.param_vars:
                    self.param_vars[param_name].set(value)
                    self.param_labels[param_name].config(text=f"{value:.2f}")
        
        finally:
            self.is_restoring_state = False
    
    def _undo(self):
        """Undo to previous state."""
        if self.history_position > 0:
            self.history_position -= 1
            state = self.history[self.history_position]
            self._restore_state(state)
            self._update_history_buttons()
            self._mark_needs_refresh()
        else:
            pass
    def _redo(self):
        """Redo to next state."""
        if self.history_position < len(self.history) - 1:
            self.history_position += 1
            state = self.history[self.history_position]
            self._restore_state(state)
            self._update_history_buttons()
            self._mark_needs_refresh()
    
    def _update_history_buttons(self):
        """Update undo/redo button states."""
        # Enable undo if we can go back
        can_undo = self.history_position > 0
        can_redo = self.history_position < len(self.history) - 1
        
        if can_undo:
            self.undo_btn.config(state=tk.NORMAL)
        else:
            self.undo_btn.config(state=tk.DISABLED)
        
        # Enable redo if we can go forward
        if can_redo:
            self.redo_btn.config(state=tk.NORMAL)
        else:
            self.redo_btn.config(state=tk.DISABLED)
        
    
    def _get_current_ui_state(self) -> Dict[str, Any]:
        """Get current UI state for comparison (only state-specific params)."""
        return {
            'text': self.text_widget.get("1.0", tk.END).strip(),
            'seed': self.seed_var.get(),
            'params': {
                'exaggeration': self.param_vars['exaggeration'].get(),
                'cfg_weight': self.param_vars['cfg_weight'].get(),
                'temperature': self.param_vars['temperature'].get(),
                'repetition_penalty': self.param_vars['repetition_penalty'].get(),
                'min_p': self.param_vars['min_p'].get(),
                'top_p': self.param_vars['top_p'].get(),
            }
        }
    
    def _mark_needs_refresh(self):
        """Mark that UI has changed and needs refresh."""
        active_state = self.state_a if self.active_state == 1 else self.state_b
        active_state['needs_refresh'] = True
        self._update_refresh_button_color()
    
    def _update_refresh_button_color(self):
        """Update refresh button color based on whether refresh is needed."""
        active_state = self.state_a if self.active_state == 1 else self.state_b
        
        if active_state['needs_refresh']:
            # Bright green - needs refresh
            self.refresh_btn.config(bg="#4CAF50")
        else:
            # Pale green - audio matches UI
            self.refresh_btn.config(bg="#677867")
    
    def _switch_state(self, state_num: int):
        """Switch between state A and state B."""
        if state_num == self.active_state:
            return  # Already in this state
        
        
        # Save current UI to current state before switching
        current_state = self.state_a if self.active_state == 1 else self.state_b
        current_ui = self._get_current_ui_state()
        current_state['text'] = current_ui['text']
        current_state['seed'] = current_ui['seed']
        current_state['params'] = current_ui['params']
        
        # Switch active state
        self.active_state = state_num
        new_state = self.state_a if state_num == 1 else self.state_b
        
        
        # Update UI button states
        if state_num == 1:
            self.state_a_btn.config(relief=tk.SUNKEN, bg="#ffffff")
            self.state_b_btn.config(relief=tk.RAISED, bg="#e0e0e0")
        else:
            self.state_a_btn.config(relief=tk.RAISED, bg="#e0e0e0")
            self.state_b_btn.config(relief=tk.SUNKEN, bg="#ffffff")
        
        # Stop current playback
        self._stop_playback()
        
        # Load new state into UI (without triggering history save)
        self.is_restoring_state = True
        try:
            # Restore text
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.insert("1.0", new_state.get('text', ''))
            
            # Restore seed
            self.seed_var.set(new_state.get('seed', '12345'))
            
            # Restore parameters
            params = new_state.get('params', {})
            for param_name, value in params.items():
                if param_name in self.param_vars:
                    self.param_vars[param_name].set(value)
                    self.param_labels[param_name].config(text=f"{value:.2f}")
            
            # Load audio for this state
            self.current_audio = new_state.get('audio')
            self.temp_audio_file = new_state.get('temp_file')
            
            if self.current_audio is not None and self.temp_audio_file:
                # Stop any running position updates before resetting timeline
                if self.update_timer:
                    self.root.after_cancel(self.update_timer)
                    self.update_timer = None
                
                # Calculate duration
                self.audio_duration = len(self.current_audio) / self.sample_rate
                self.timeline.config(to=self.audio_duration * 1000)
                self.timeline_var.set(0)
                self.playback_position = 0.0
                self.save_btn.config(state=tk.NORMAL)
            else:
                # Stop any running position updates before resetting timeline
                if self.update_timer:
                    self.root.after_cancel(self.update_timer)
                    self.update_timer = None
                
                # No audio in this state
                self.audio_duration = 0.0
                self.timeline.config(to=100)
                self.timeline_var.set(0)
                self.playback_position = 0.0
                self.save_btn.config(state=tk.DISABLED)
        
        finally:
            self.is_restoring_state = False
        
        # Update refresh button color
        self._update_refresh_button_color()
        
        # IMPORTANT: Update history buttons after switching
        self._update_history_buttons()
    
    def _copy_state(self, from_state_num: int, to_state_num: int):
        """Copy state from one to another."""
        
        # Save current UI to active state first
        current_state = self.state_a if self.active_state == 1 else self.state_b
        current_ui = self._get_current_ui_state()
        current_state['text'] = current_ui['text']
        current_state['seed'] = current_ui['seed']
        current_state['params'] = current_ui['params']
        
        # If currently in target state, save current state BEFORE copying (for undo)
        if self.active_state == to_state_num:
            self._save_state_to_history(force=True)
        else:
            pass
        # Get source and target states
        source = self.state_a if from_state_num == 1 else self.state_b
        
        # Copy source to target
        copied_state = {
            'audio': source.get('audio'),
            'temp_file': source.get('temp_file'),
            'text': source.get('text', ''),
            'seed': source.get('seed', '12345'),
            'params': source.get('params', {}).copy(),
            'needs_refresh': source.get('needs_refresh', True)
        }
        
        if to_state_num == 1:
            self.state_a = copied_state
        else:
            self.state_b = copied_state
        
        
        # If currently in target state, update UI (switch_state will restore from state)
        if self.active_state == to_state_num:
            # Disable history saving during UI update
            self.is_restoring_state = True
            try:
                # Restore text
                self.text_widget.delete("1.0", tk.END)
                self.text_widget.insert("1.0", copied_state.get('text', ''))
                
                # Restore seed
                self.seed_var.set(copied_state.get('seed', '12345'))
                
                # Restore parameters
                params = copied_state.get('params', {})
                for param_name, value in params.items():
                    if param_name in self.param_vars:
                        self.param_vars[param_name].set(value)
                        self.param_labels[param_name].config(text=f"{value:.2f}")
                
                # Update refresh button
                self._update_refresh_button_color()
            finally:
                self.is_restoring_state = False
        else:
            pass
    
    def _copy_a_to_b(self):
        """Copy state A to state B."""
        self._copy_state(1, 2)
    
    def _copy_b_to_a(self):
        """Copy state B to state A."""
        self._copy_state(2, 1)
    
    def _on_refresh(self):
        """Handle refresh button click."""
        if self.is_generating:
            return
        
        # Update char count
        self._update_char_count()
        
        # Save current settings
        self._save_settings()
        
        # Start generation in thread to keep UI responsive
        generation_thread = threading.Thread(target=self._generate_and_play, daemon=True)
        generation_thread.start()
    
    def _generate_and_play(self):
        """Generate audio with current parameters and play it."""
        self.is_generating = True
        
        # Disable refresh button
        self.root.after(0, lambda: self.refresh_btn.config(state=tk.DISABLED, text="⏳ Generating..."))
        
        # Don't stop current playback immediately - let it continue during generation
        # Audio will be stopped only when new audio is ready to play
        
        # Get current parameters
        text = self.text_widget.get("1.0", tk.END).strip()
        if not text:
            return
        
        # Set seed for reproducibility
        # Convention: seed > 0 => deterministic; seed == 0 => use random (do NOT set torch seed)
        try:
            seed_str = self.seed_var.get().strip()
            seed = int(seed_str) if seed_str.isdigit() else 0
        except Exception:
            seed = 0

        if seed > 0:
            torch.manual_seed(seed)
            print(f"🎲 Seed set to: {seed}")
        else:
            # Intentionally do not call torch.manual_seed so the model uses randomness internally
            print("🎲 Using random seed (0)")
        
        # Get reference audio path
        ref_audio_file = self.ref_audio_var.get()
        ref_audio_path = str(Path(__file__).parent / "data" / "input" / "reference_audio" / ref_audio_file)
        
        # Prepare generation parameters
        # Build params dict (without language_id - it's positional!)
        params = {
            "audio_prompt_path": ref_audio_path,
            "exaggeration": self.param_vars["exaggeration"].get(),
            "cfg_weight": self.param_vars["cfg_weight"].get(),
            "temperature": self.param_vars["temperature"].get(),
            "repetition_penalty": self.param_vars["repetition_penalty"].get(),
            "min_p": self.param_vars["min_p"].get(),
            "top_p": self.param_vars["top_p"].get(),
        }
        
        try:
            # Generate audio with current parameters
            if self.current_model_type == "multilingual":
                lang_key = self.lang_var.get()
                language_id = self.LANGUAGES[lang_key]
                audio_tensor = self.current_model.generate(text, language_id=language_id, **params)
            else:
                # Classic model doesn't need language_id
                audio_tensor = self.current_model.generate(text, **params)
            
            # Convert to numpy array
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.cpu().numpy()
            else:
                audio_np = audio_tensor
            
            # Ensure 1D array
            if audio_np.ndim == 2:
                audio_np = audio_np.squeeze()
            
            # Normalize audio to prevent clipping and distortion
            # Find peak value
            max_val = np.abs(audio_np).max()
            if max_val > 0:
                # Normalize to -1.0 to 1.0 range with some headroom
                audio_np = audio_np / max_val * 0.95
            
            # Convert to int16 range
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            self.current_audio = audio_int16
            self.sample_rate = 24000  # ChatterboxTTS native sample rate
            
            # Save to temporary file for pygame
            if self.temp_audio_file and os.path.exists(self.temp_audio_file):
                os.remove(self.temp_audio_file)
            
            self.temp_audio_file = tempfile.mktemp(suffix='.wav')
            sf.write(self.temp_audio_file, audio_int16, self.sample_rate)
            
            # Calculate duration
            self.audio_duration = len(audio_int16) / self.sample_rate
            
            # Stop any running position updates before resetting timeline
            if self.update_timer:
                self.root.after_cancel(self.update_timer)
                self.update_timer = None
            
            # Reset timeline
            self.timeline.config(to=self.audio_duration * 1000)  # Convert to ms
            self.timeline_var.set(0)
            self.playback_position = 0.0
            
            # Save audio to active state
            active_state = self.state_a if self.active_state == 1 else self.state_b
            active_state['audio'] = audio_int16
            active_state['temp_file'] = self.temp_audio_file
            active_state['text'] = text
            active_state['seed'] = self.seed_var.get()
            active_state['params'] = {
                'exaggeration': self.param_vars['exaggeration'].get(),
                'cfg_weight': self.param_vars['cfg_weight'].get(),
                'temperature': self.param_vars['temperature'].get(),
                'repetition_penalty': self.param_vars['repetition_penalty'].get(),
                'min_p': self.param_vars['min_p'].get(),
                'top_p': self.param_vars['top_p'].get(),
            }
            active_state['needs_refresh'] = False
            
            # Update refresh button color to pale green
            self.root.after(0, self._update_refresh_button_color)
            
            # Enable save button
            self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
            
            # Stop current playback and play new audio
            self._stop_playback()
            self._play_audio()
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_generating = False
            # Re-enable refresh button
            self.root.after(0, lambda: self.refresh_btn.config(state=tk.NORMAL, text="⟳ REFRESH - Generate Audio"))
    
    def _on_play(self):
        """Handle play button click."""
        if self.current_audio is None or not self.temp_audio_file:
            return
        
        # If already playing, do nothing (prevent multiple instances)
        if self.is_playing:
            return
        
        # If at end, restart from beginning
        if self.playback_position >= self.audio_duration:
            self.playback_position = 0.0
            self.timeline_var.set(0)
        
        self._play_audio()
    
    def _on_pause(self):
        """Handle pause button click."""
        self._pause_playback()
    
    def _on_timeline_change(self, value):
        """Handle timeline scrubbing."""
        if self.current_audio is None or not self.temp_audio_file:
            return
        
        # Convert timeline value (ms) to seconds
        new_position = float(value) / 1000.0
        self.playback_position = new_position
        
        # If currently playing, restart from new position
        if self.is_playing:
            # Stop current playback and restart from new position
            self._stop_playback()
            self._play_audio()
    
    def _play_audio(self):
        """Play audio from current position."""
        if self.current_audio is None or not self.temp_audio_file:
            return
        
        # Stop any currently playing audio first
        if self.is_playing:
            self._stop_playback()
        
        # Ensure clean state before starting
        self.is_playing = False
        self.external_player_process = None
        
        # Wait a moment to ensure previous process is fully cleaned up
        time.sleep(0.1)
        
        self.is_playing = True
        self.play_btn.config(state=tk.DISABLED, bg="#90EE90")  # Light green when playing
        self.pause_btn.config(state=tk.NORMAL, bg="#f0f0f0")
        
        try:
            # Try to use external player for better quality in WSL
            # Check if ffplay or aplay is available
            import subprocess
            import shutil
            
            if shutil.which('ffplay'):
                # Use ffplay (from ffmpeg) - best quality with seeking support
                ffplay_cmd = ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet']
                
                # Add seeking parameter if playback_position > 0
                if self.playback_position > 0:
                    ffplay_cmd.extend(['-ss', str(self.playback_position)])
                
                ffplay_cmd.append(self.temp_audio_file)
                
                self.external_player_process = subprocess.Popen(
                    ffplay_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                # Start position update loop
                self._update_position()
                return
            elif shutil.which('aplay'):
                # Use aplay (ALSA player)
                self.external_player_process = subprocess.Popen(
                    ['aplay', '-q', self.temp_audio_file],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                # Start position update loop
                self._update_position()
                return
        except Exception as e:
            print(f"Error starting audio player: {e}")
        
        # Fallback to pygame (disabled for WSL2)
        print("Warning: No external audio player available. Audio playback disabled.")
        print("Install ffmpeg for audio support: sudo apt install ffmpeg")
        self.is_playing = False
        self.play_btn.config(state=tk.NORMAL, bg="#f0f0f0")
        self.pause_btn.config(state=tk.DISABLED, bg="#f0f0f0")
    
    def _update_position(self):
        """Update timeline position during playback."""
        # Double-check is_playing state to prevent race conditions
        if not self.is_playing:
            return
        
        # Cancel any existing timer first to prevent accumulation
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
            self.update_timer = None
        
        # Check if external player process exists and is running
        if self.external_player_process:
            poll_result = self.external_player_process.poll()
            if poll_result is None:
                # Process still running, estimate position more accurately
                self.playback_position += 0.05  # 50ms increments
                
                # Check if we've reached or exceeded the duration
                if self.playback_position >= self.audio_duration:
                    self._on_playback_finished()
                    return
                
                # Update timeline position
                position_ms = self.playback_position * 1000
                self.timeline_var.set(position_ms)
                
                # Schedule next update only if still playing
                if self.is_playing:
                    self.update_timer = self.root.after(50, self._update_position)
            else:
                # Process finished - check if we're close to end
                if self.playback_position >= self.audio_duration - 0.1:  # Close to end
                    self._on_playback_finished()
                else:
                    # Process finished early - force finish
                    print(f"Warning: ffplay process finished early at {self.playback_position:.2f}s of {self.audio_duration:.2f}s")
                    self._on_playback_finished()
        else:
            # No external player process - finish playback
            self._on_playback_finished()
    
    def _pause_playback(self):
        """Pause audio playback."""
        if not self.is_playing:
            return
        
        # For pygame, we can pause (disabled for WSL2)
        if not self.external_player_process:
            # mixer.music.pause() disabled for WSL2
            self._stop_playback()
            return
        else:
            # For external players, we need to stop (no pause support)
            self._stop_playback()
            return
        
        self.is_playing = False
        self.play_btn.config(state=tk.NORMAL, bg="#f0f0f0")
        self.pause_btn.config(state=tk.DISABLED, bg="#f0f0f0")
        
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
            self.update_timer = None
    
    def _stop_playback(self):
        """Stop audio playback."""
        # Stop position updates first to prevent race conditions
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
            self.update_timer = None
        
        # Set playing state to False
        self.is_playing = False
        
        # Stop external player process if running
        if self.external_player_process:
            try:
                # Try graceful termination first
                self.external_player_process.terminate()
                # Wait longer for graceful shutdown (especially important for ffplay)
                self.external_player_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # Force kill if graceful termination fails
                try:
                    print("Warning: ffplay process did not terminate gracefully, forcing kill")
                    self.external_player_process.kill()
                    self.external_player_process.wait(timeout=2)
                except:
                    pass
            except:
                # Handle other exceptions
                try:
                    self.external_player_process.kill()
                except:
                    pass
            finally:
                self.external_player_process = None
        
        # Stop pygame music playback (disabled for WSL2)
        try:
            # mixer.music.stop() disabled for WSL2
            pass
        except:
            pass
        
        # Reset button states and colors
        self.play_btn.config(state=tk.NORMAL, bg="#f0f0f0")
        self.pause_btn.config(state=tk.DISABLED, bg="#f0f0f0")
    
    def _on_playback_finished(self):
        """Handle playback finished."""
        # Stop position updates first to prevent race conditions
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
            self.update_timer = None
        
        # Set playing state to False
        self.is_playing = False
        
        # Update UI
        self.play_btn.config(state=tk.NORMAL, bg="#f0f0f0")
        self.pause_btn.config(state=tk.DISABLED, bg="#f0f0f0")
        
        # Properly clean up external player process
        if self.external_player_process:
            try:
                # Try graceful termination first
                self.external_player_process.terminate()
                self.external_player_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # Force kill if graceful termination fails
                try:
                    self.external_player_process.kill()
                    self.external_player_process.wait(timeout=1)
                except:
                    pass
            except:
                # Handle other exceptions
                try:
                    self.external_player_process.kill()
                except:
                    pass
            finally:
                self.external_player_process = None
        
        # Set position to end
        self.playback_position = self.audio_duration
        self.timeline_var.set(self.audio_duration * 1000)
    
    def _copy_settings(self):
        """Copy current settings as YAML to clipboard."""
        params = {
            "exaggeration": self.param_vars["exaggeration"].get(),
            "cfg_weight": self.param_vars["cfg_weight"].get(),
            "temperature": self.param_vars["temperature"].get(),
            "repetition_penalty": self.param_vars["repetition_penalty"].get(),
            "min_p": self.param_vars["min_p"].get(),
            "top_p": self.param_vars["top_p"].get(),
        }
        
        yaml_text = f"""tts_params:
        exaggeration: {params['exaggeration']:.2f}
        exaggeration_max_deviation: 0.0
        cfg_weight: {params['cfg_weight']:.2f}
        cfg_weight_max_deviation: 0.0
        temperature: {params['temperature']:.2f}
        temperature_max_deviation: 0.0
        repetition_penalty: {params['repetition_penalty']:.2f}
        min_p: {params['min_p']:.2f}
        top_p: {params['top_p']:.2f}"""
        
        # Try multiple clipboard methods for better compatibility
        success = False
        
        # Method 1: Try xclip (WSL/Linux)
        import subprocess
        import shutil
        
        if shutil.which('xclip'):
            try:
                process = subprocess.Popen(
                    ['xclip', '-selection', 'clipboard'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                process.communicate(input=yaml_text.encode('utf-8'))
                if process.returncode == 0 or process.returncode is None:
                    success = True
                    print("Copied using xclip")
            except:
                pass
        
        # Method 2: Try xsel (alternative Linux clipboard tool)
        if not success and shutil.which('xsel'):
            try:
                process = subprocess.Popen(
                    ['xsel', '--clipboard', '--input'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                process.communicate(input=yaml_text.encode('utf-8'))
                if process.returncode == 0 or process.returncode is None:
                    success = True
                    print("Copied using xsel")
            except:
                pass
        
        # Method 3: Fallback to Tkinter (may not work in WSL)
        if not success:
            try:
                self.root.clipboard_clear()
                self.root.update()
                self.root.clipboard_append(yaml_text)
                self.root.update()
                success = True
                print("Copied using Tkinter clipboard")
            except:
                pass
        
        # Debug output to verify what was copied
        print(f"Copied to clipboard:")
        print(f"  exaggeration: {params['exaggeration']:.2f}")
        print(f"  cfg_weight: {params['cfg_weight']:.2f}")
        print(f"  temperature: {params['temperature']:.2f}")
        print(f"  repetition_penalty: {params['repetition_penalty']:.2f}")
        print(f"  min_p: {params['min_p']:.2f}")
        print(f"  top_p: {params['top_p']:.2f}")
        
        if not success:
            print("WARNING: Failed to copy to clipboard. Install xclip: sudo apt-get install xclip")
        
        # Visual feedback
        original_text = "📋 Copy YAML to Clipboard"
        if success:
            self.copy_btn.config(text="✓ Copied to clipboard!")
        else:
            self.copy_btn.config(text="✗ Clipboard error - see console")
        self.root.after(2000, lambda: self.copy_btn.config(text=original_text))
    
    def _insert_from_clipboard(self):
        """Insert parameters from YAML clipboard content (flexible parsing)."""
        try:
            # Get clipboard content
            clipboard_text = self.root.clipboard_get()
            
            # Clean up clipboard text: remove leading whitespace from each line
            # This handles cases where user copies indented YAML from config files
            lines = clipboard_text.split('\n')
            cleaned_lines = []
            for line in lines:
                # Remove leading spaces/tabs but preserve the line structure
                cleaned_line = line.lstrip()
                if cleaned_line:  # Skip empty lines
                    cleaned_lines.append(cleaned_line)
            
            cleaned_text = '\n'.join(cleaned_lines)
            print(f"[CLIPBOARD] Cleaned text:\n{cleaned_text}")
            
            # Parse YAML
            import yaml
            data = yaml.safe_load(cleaned_text)
            
            if not isinstance(data, dict):
                raise ValueError("Clipboard content is not valid YAML dict")
            
            updated_items = []
            
            # Try to get tts_params (either as section or directly in root)
            # If tts_params exists as a key, use it; otherwise search in root
            if 'tts_params' in data and isinstance(data['tts_params'], dict):
                tts_params = data['tts_params']
            else:
                # No tts_params section, use root dict
                tts_params = data
            
            # Map of expected parameters
            param_mapping = {
                'exaggeration': 'exaggeration',
                'cfg_weight': 'cfg_weight',
                'temperature': 'temperature',
                'repetition_penalty': 'repetition_penalty',
                'min_p': 'min_p',
                'top_p': 'top_p'
            }
            
            # Update parameters if they exist in clipboard YAML
            for yaml_key, ui_key in param_mapping.items():
                if yaml_key in tts_params:
                    try:
                        # Only process if value is a number (not a dict or other structure)
                        value = float(tts_params[yaml_key])
                        self.param_vars[ui_key].set(value)
                        self.param_labels[ui_key].config(text=f"{value:.2f}")
                        updated_items.append(f"{ui_key}={value:.2f}")
                        print(f"[CLIPBOARD] Loaded {ui_key} = {value:.2f}")
                    except (ValueError, TypeError) as e:
                        print(f"[CLIPBOARD] Skipping {yaml_key} (not a valid number): {e}")
            
            # Check for reference_audio (with error handling)
            if 'reference_audio' in data:
                try:
                    ref_audio = str(data['reference_audio'])
                    # Extract just the filename if it's a path
                    if '/' in ref_audio:
                        ref_audio = ref_audio.split('/')[-1]
                    
                    # Check if this audio file exists in our list
                    if ref_audio in self.reference_audio_files:
                        self.ref_audio_var.set(ref_audio)
                        updated_items.append(f"ref_audio={ref_audio}")
                        print(f"[CLIPBOARD] Loaded reference_audio = {ref_audio}")
                    else:
                        print(f"[CLIPBOARD] Warning: reference_audio '{ref_audio}' not found in available files")
                except Exception as e:
                    print(f"[CLIPBOARD] Error processing reference_audio: {e}")
            
            # Check for language (with error handling)
            if 'language' in data:
                try:
                    lang_code = str(data['language'])
                    # Find the language by code
                    lang_found = False
                    for lang_display, code in self.LANGUAGES.items():
                        if code == lang_code:
                            self.lang_var.set(lang_display)
                            updated_items.append(f"language={lang_display}")
                            print(f"[CLIPBOARD] Loaded language = {lang_display} ({lang_code})")
                            lang_found = True
                            break
                    
                    if not lang_found:
                        print(f"[CLIPBOARD] Warning: language '{lang_code}' not found in available languages")
                except Exception as e:
                    print(f"[CLIPBOARD] Error processing language: {e}")
            
            # Visual feedback
            if updated_items:
                original_text = "📥 Insert from Clipboard"
                self.insert_btn.config(text=f"✓ Loaded {len(updated_items)} items!")
                print(f"[CLIPBOARD] Successfully loaded: {', '.join(updated_items)}")
                self.root.after(2000, lambda: self.insert_btn.config(text=original_text))
                # Save to history after clipboard insert (with delay)
                self._on_text_change()
                self._mark_needs_refresh()
            else:
                raise ValueError("No valid parameters found in YAML")
                
        except tk.TclError:
            # Clipboard empty or not accessible
            original_text = "📥 Insert from Clipboard"
            self.insert_btn.config(text="✗ Clipboard is empty")
            self.root.after(2000, lambda: self.insert_btn.config(text=original_text))
            print("[CLIPBOARD] Error: Clipboard is empty")
        except yaml.YAMLError as e:
            # YAML parsing error
            original_text = "📥 Insert from Clipboard"
            self.insert_btn.config(text="✗ Invalid YAML format")
            self.root.after(2000, lambda: self.insert_btn.config(text=original_text))
            print(f"[CLIPBOARD] Error: Invalid YAML format - {e}")
        except Exception as e:
            # Other errors
            original_text = "📥 Insert from Clipboard"
            self.insert_btn.config(text="✗ Error loading parameters")
            self.root.after(2000, lambda: self.insert_btn.config(text=original_text))
            print(f"[CLIPBOARD] Error loading parameters: {e}")
    
    def _save_audio(self):
        """Save current audio to file with metadata YAML."""
        if self.current_audio is None or not self.temp_audio_file:
            return
        
        # Create output directory if needed
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ref_name = Path(self.ref_audio_var.get()).stem
        base_name = f"test_{ref_name}_{timestamp}"
        output_path = output_dir / f"{base_name}.wav"
        yaml_path = output_dir / f"{base_name}.yaml"
        
        # Copy temp file to output
        import shutil
        shutil.copy2(self.temp_audio_file, output_path)
        
        # Get seed
        seed = int(self.seed_var.get()) if self.seed_var.get().isdigit() else 12345
        
        # Create YAML metadata in cbpipe format
        yaml_content = f"""# Chatterbox TTS Test - {timestamp}
# Reference Audio: {self.ref_audio_var.get()}
# Model: {self.model_var.get()}"""
        
        if self.model_var.get() == 'multilanguage':
            lang_code = self.LANGUAGES.get(self.lang_var.get(), 'en')
            yaml_content += f"\n# Language: {self.lang_var.get()} ({lang_code})"
        
        yaml_content += f"""
# Seed: {seed}

# Text:
# {self.text_widget.get("1.0", tk.END).strip()}

tts_params:
        exaggeration: {self.param_vars['exaggeration'].get():.2f}
        exaggeration_max_deviation: 0.0
        cfg_weight: {self.param_vars['cfg_weight'].get():.2f}
        cfg_weight_max_deviation: 0.0
        temperature: {self.param_vars['temperature'].get():.2f}
        temperature_max_deviation: 0.0
        repetition_penalty: {self.param_vars['repetition_penalty'].get():.2f}
        min_p: {self.param_vars['min_p'].get():.2f}
        top_p: {self.param_vars['top_p'].get():.2f}
"""
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        # Visual feedback
        original_text = "💾 Save Audio"
        self.save_btn.config(text=f"✓ Saved!")
        print(f"Audio saved to: {output_path}")
        print(f"Metadata saved to: {yaml_path}")
        self.root.after(2000, lambda: self.save_btn.config(text=original_text))
    
    def cleanup(self):
        """Cleanup temporary files and save settings."""
        # Stop any playing audio
        self._stop_playback()
        
        # Additional cleanup for external player processes
        if self.external_player_process:
            try:
                self.external_player_process.terminate()
                self.external_player_process.wait(timeout=2)
            except:
                try:
                    self.external_player_process.kill()
                except:
                    pass
            self.external_player_process = None
        
        # Cancel any pending timers
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
            self.update_timer = None
        
        # Save settings before closing
        self._save_settings()
        
        # Cleanup temp audio file
        if self.temp_audio_file and os.path.exists(self.temp_audio_file):
            try:
                os.remove(self.temp_audio_file)
            except:
                pass


def main():
    """Main entry point."""
    import sys
    
    # Show help if requested
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("""
Chatterbox TTS Parameter Testing Tool

Usage: python chatterbox_tester.py

A GUI tool for quickly testing different Chatterbox parameters, models,
and reference audio files to find optimal voice settings.

Features:
  - Select reference audio from data/input/reference_audio/
  - Switch between classic and multilanguage models (24 languages)
  - Parameter adjustment (exaggeration, cfg_weight, temperature, etc.)
  - Manual REFRESH button for generation
  - Audio playback (uses ffplay/aplay if available for better quality)
  - Save generated audio to output/ directory
  - Export settings as YAML for cbpipe configurations

For better audio playback in WSL, install ffmpeg:
  sudo apt-get install ffmpeg

For detailed documentation, see CHATTERBOX_TESTER.md
        """)
        sys.exit(0)
    
    # Check if running in a display environment
    try:
        root = tk.Tk()
        app = ChatterboxTester(root)
        
        # Cleanup on exit
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
