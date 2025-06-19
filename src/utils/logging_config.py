"""
Simplified logging configuration utility.
Provides dual logging (console + rotating file) with optional colorized console output.
Uses standard Python logging levels: INFO for normal messages, DEBUG for verbose messages.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Optional color support for console output (falls back silently)
try:
    from colorama import Fore, Style
    from colorama import init as colorama_init  # type: ignore

    colorama_init()

    class _ColorFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG: Fore.CYAN,
            logging.INFO: Fore.GREEN,
            logging.WARNING: Fore.YELLOW,
            logging.ERROR: Fore.RED,
            logging.CRITICAL: Fore.MAGENTA,
        }

        def format(self, record: logging.LogRecord) -> str:  # noqa: D401
            color = self.COLORS.get(record.levelno, "")
            reset = Style.RESET_ALL if color else ""
            message = super().format(record)
            return f"{color}{message}{reset}"

except ImportError:  # pragma: no cover – colorama optional

    class _ColorFormatter(logging.Formatter):
        """Fallback formatter without colors."""

        def format(self, record: logging.LogRecord) -> str:  # noqa: D401
            return super().format(record)


class StructuredFormatter(logging.Formatter):
    """Simplified formatter that adds contextual icons to logging messages."""

    # Consolidated icon mapping (context overrides level)
    ICONS = {
        # Level defaults
        logging.DEBUG: "ℹ️ ",
        logging.INFO: "",
        logging.WARNING: "⚠️ ",
        logging.ERROR: "❌ ",
        logging.CRITICAL: "🚨 ",
        # Context overrides (checked first)
        "starting": "▶️ ",
        "preprocessing": "📝",
        "generation": "⚡",
        "validation": "🚦",
        "assembly": "🔧",
        "complete": "✅",
        "model": "🤖",
        "config": "⚙️ ",
        "audio": "〰️",
        "file": "📁",
    }

    def __init__(self, fmt: str, use_icons: bool = True):
        super().__init__(fmt)
        self.use_icons = use_icons

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        if not self.use_icons:
            return super().format(record)

        message = record.getMessage()
        icon_list = ['🎯', '✅', '❌', '⚠️', '🔍', 'ℹ️', '🚨', '📝', '⚡', '🚦', '🔧', '▶️', '⏳', '📁', '🤖', '〰️', '⚙️', '🚀', '💾', '🎵', '📊', '🎙️', '🔄', '⏭️', '♻️', '🗑️', '✓', '🎛️', '💻', '🏁']

        # Multi-codepoint-safe Icon-Erkennung am Zeilenanfang
        stripped = message.lstrip()
        for icon in icon_list:
            if stripped.startswith(icon):
                # Alles nach dem Icon (inkl. Whitespaces) entfernen, exakt ein Leerzeichen einfügen
                rest = stripped[len(icon):].lstrip()
                record.msg = f"{icon} {rest}" if rest else icon
                record.args = None
                return super().format(record)

        # Skip icon addition for structural messages
        if self._skip_icon_addition(message):
            return super().format(record)

        # Add icon if none exists
        icon = self._get_icon(message, record.levelno)
        if icon:
            record.msg = f"{icon} {message}"
            record.args = None

        return super().format(record)

    def _skip_icon_addition(self, message: str) -> bool:
        """Check if we should skip adding an icon (for structural messages)."""
        return (
            # Has emoji (simple Unicode check) - improved to catch more cases
            any(ord(c) > 0x1F300 for c in message[:30])  # Extended range and check more characters
            or set(message.strip()) <= {"=", "-", " "}
            or message.count(":") > 2
            or "TTS PIPELINE" in message
            or message.startswith("- ")
        )

    def _get_icon(self, message: str, level: int) -> str:
        """Get best icon (context-aware)."""
        msg_lower = message.lower()

        # Check context keywords first
        for keyword, icon in self.ICONS.items():
            if isinstance(keyword, str) and keyword in msg_lower:
                return icon

        # Fallback to level icon
        return self.ICONS.get(level, "")


class LoggingConfigurator:  # pylint: disable=too-few-public-methods
    """Central logging configuration helper with dual logging support.

    Usage:
        from utils.logging_config import LoggingConfigurator
        LoggingConfigurator.configure(log_file, verbose_mode=True)
    """

    @staticmethod
    def configure(
        log_file: Path,
        *,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 3,
        append: bool = False,
        verbose_mode: bool = False,
        use_icons: bool = True,
    ) -> None:
        """Configure root logger with enhanced console and rotating file handler.

        Args:
            log_file: Target file path for detailed logs.
            console_level: Verbosity for console handler (default: INFO).
            file_level: Verbosity for file handler (default: DEBUG).
            max_bytes: Maximum size of each rotating log file.
            backup_count: Number of backup log files to keep.
            append: Append to existing log file instead of truncating.
            verbose_mode: Enable verbose mode (show DEBUG level on console).
            use_icons: Enable icons in console output.
        """
        # Ensure directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        root_logger = logging.getLogger()
        root_logger.setLevel(min(console_level, file_level))

        # Remove pre-existing handlers to avoid duplicate logs in interactive sessions
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Set console level based on verbose mode
        if verbose_mode:
            actual_console_level = logging.DEBUG
            # console_fmt = "%(levelname)s - %(message)s"
        else:
            actual_console_level = console_level  # Usually INFO
            # console_fmt = "%(message)s"  # Cleaner format for non-verbose mode

        console_fmt = "%(message)s"

        # Console handler with structured formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(actual_console_level)

        # Create structured formatter with color support
        class StructuredColorFormatter(_ColorFormatter, StructuredFormatter):
            def __init__(self, fmt: str):
                _ColorFormatter.__init__(self, fmt)
                StructuredFormatter.__init__(self, fmt, use_icons=use_icons)

            def format(self, record: logging.LogRecord) -> str:
                # Apply structure first, then color
                record = StructuredFormatter.format(self, record)
                return (
                    _ColorFormatter.format(self, record)
                    if isinstance(record, logging.LogRecord)
                    else record
                )

        console_formatter = StructuredColorFormatter(console_fmt)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # Structured log record factory: ensures optional context fields exist
        old_factory = logging.getLogRecordFactory()

        def _record_factory(*fa, **kwa):  # type: ignore
            record = old_factory(*fa, **kwa)
            # Ensure custom attributes exist to avoid formatting errors
            for attr in ("stage", "task", "job"):
                if not hasattr(record, attr):
                    setattr(record, attr, "-")
            return record

        logging.setLogRecordFactory(_record_factory)

        # Rotating file handler – detailed, structured format (always includes DEBUG)
        file_fmt = "%(asctime)s | %(levelname)-8s | %(job)s | %(task)s | %(stage)s | %(name)s | %(message)s"
        file_mode = "a" if append else "w"
        file_handler: Optional[logging.Handler]
        try:
            file_handler = RotatingFileHandler(
                log_file,
                mode=file_mode,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        except Exception:  # pragma: no cover – cannot instantiate
            # Fallback to simple FileHandler if rotating fails for platform reasons
            file_handler = logging.FileHandler(
                log_file, mode=file_mode, encoding="utf-8"
            )

        file_handler.setLevel(file_level)  # Usually DEBUG - logs everything to file
        file_handler.setFormatter(logging.Formatter(file_fmt))
        root_logger.addHandler(file_handler)

        # Configure third-party library logging
        LoggingConfigurator._configure_third_party_logging(verbose_mode)

    @staticmethod
    def _configure_third_party_logging(verbose_mode: bool = False) -> None:
        """Configure third-party library logging levels."""

        # Always reduce noisy third-party libraries
        noisy_libs = [
            "urllib3",
            "matplotlib",
            "PIL",
            "requests",
            "transformers.tokenization_utils",
            "transformers.configuration_utils",
            "transformers.modeling_utils",
        ]

        for lib in noisy_libs:
            logging.getLogger(lib).setLevel(logging.WARNING)

        # Completely silence numba bytecode dumps (as requested)
        logging.getLogger("numba.core.byteflow").setLevel(logging.CRITICAL)
        logging.getLogger("numba").setLevel(logging.WARNING)

        # Additional ChatterboxTTS/HuggingFace noise reduction
        if not verbose_mode:
            logging.getLogger("transformers").setLevel(logging.WARNING)
            logging.getLogger("torch").setLevel(logging.WARNING)
            logging.getLogger("torchaudio").setLevel(logging.WARNING)
