"""
Utilities for number-aware text normalization used during validation and scoring.

This module provides language-aware, low-dependency normalization routines
to reduce mismatch impact between numeric digits and written-out number words.

Modes:
- off:         No normalization
- placeholder: Replace digits and recognized number words with a placeholder token
- digits:      Convert recognized number words to digits (best-effort)
- words:       Convert digit sequences to written-out words (best-effort)

Dependencies are optional. If unavailable, functions fall back to 'placeholder'.

All functions are designed to be robust and safe for production use.
"""

from __future__ import annotations

import re
from typing import Optional

_PLACEHOLDER_TOKEN = "<NUM>"


def _safe_import_word2num_de():
    try:
        from word2num_de import word_to_number  # type: ignore

        return word_to_number
    except Exception:
        return None


def _safe_import_num2words():
    try:
        from num2words import num2words  # type: ignore

        return num2words
    except Exception:
        return None


_WORD_TO_NUMBER_DE = _safe_import_word2num_de()
_NUM2WORDS = _safe_import_num2words()


_DIGITS_RE = re.compile(r"\d+")

# Conservative German number-word detector for placeholder mode as a fallback
# Covers common components and compositions; complex phrases remain best-effort
_DE_NUMBER_WORD_PARTS = (
    "null|eins|ein|eine|einem|einen|einer|zwei|drei|vier|fuenf|fünf|sechs|sieben|acht|neun|zehn|"
    "elf|zwoelf|zwölf|dreizehn|vierzehn|fuenfzehn|fünfzehn|sechzehn|siebzehn|achtzehn|neunzehn|"
    "zwanzig|dreissig|dreißig|vierzig|fuenfzig|fünfzig|sechzig|siebzig|achtzig|neunzig|"
    "hundert|tausend|million|millionen|milliarde|milliarden|billion|billionen|"
    "erst|zweit|dritt|viert|fuenft|fünft|sechst|siebt|acht|neunt|zehnt"
)
_DE_NUMBER_WORD_RE = re.compile(rf"^(?:{_DE_NUMBER_WORD_PARTS})(?:[a-zäöüß-])*\Z", re.IGNORECASE)


def _replace_digits_with_placeholder(text: str) -> str:
    return _DIGITS_RE.sub(_PLACEHOLDER_TOKEN, text)


def _de_word_is_number(token: str) -> bool:
    if not token:
        return False
    t = token.strip().lower()
    # Strip punctuation around token
    t = t.strip(".,;:!?()[]{}\"'")

    # Try strict parser first if available
    if _WORD_TO_NUMBER_DE is not None:
        try:
            value = _WORD_TO_NUMBER_DE(t)
            if isinstance(value, int):
                return True
        except Exception:
            pass

    # Fallback heuristic: pattern of common german number components
    return bool(_DE_NUMBER_WORD_RE.match(t))


def _replace_de_number_words_with_placeholder(text: str) -> str:
    # Tokenize crudely on whitespace to keep complexity low
    tokens = text.split()
    out_tokens = []
    for tok in tokens:
        if _de_word_is_number(tok):
            out_tokens.append(_PLACEHOLDER_TOKEN)
        else:
            out_tokens.append(tok)
    return " ".join(out_tokens)


def _de_number_words_to_digits(text: str) -> Optional[str]:
    if _WORD_TO_NUMBER_DE is None:
        return None

    def convert_token(tok: str) -> str:
        stripped = tok.strip(".,;:!?()[]{}\"'")
        try:
            value = _WORD_TO_NUMBER_DE(stripped.lower())
            if isinstance(value, int):
                # Preserve surrounding punctuation
                return tok.replace(stripped, str(value))
        except Exception:
            pass
        return tok

    tokens = text.split()
    converted = [convert_token(t) for t in tokens]
    return " ".join(converted)


def _digits_to_words(text: str, language: str) -> Optional[str]:
    if _NUM2WORDS is None:
        return None

    def repl(m: re.Match[str]) -> str:
        try:
            number = int(m.group(0))
            # num2words language codes largely match provided ones (e.g., 'de', 'fr', 'es')
            return _NUM2WORDS(number, lang=language)
        except Exception:
            return _PLACEHOLDER_TOKEN

    return _DIGITS_RE.sub(repl, text)


def normalize_text_for_numbers(text: str, language: str, mode: str) -> str:
    """
    Normalize a text for numeric content according to the given mode.

    Args:
        text: Input text
        language: ISO-ish language code (e.g., 'en', 'de')
        mode: 'off' | 'placeholder' | 'digits' | 'words'

    Returns:
        Normalized text (best-effort, safe fallbacks)
    """
    if not text or not mode or mode == "off":
        return text

    language = (language or "en").lower()
    mode = mode.lower()

    # Always replace raw digits for placeholder mode
    if mode == "placeholder":
        text = _replace_digits_with_placeholder(text)
        if language == "de":
            text = _replace_de_number_words_with_placeholder(text)
        return text

    # digits: words -> digits; always also normalize raw digits shape via no-op
    if mode == "digits":
        if language == "de":
            converted = _de_number_words_to_digits(text)
            if converted is not None:
                return converted
        # Fallback
        return normalize_text_for_numbers(text, language, mode="placeholder")

    # words: digits -> words
    if mode == "words":
        converted = _digits_to_words(text, language)
        if converted is not None:
            return converted
        return normalize_text_for_numbers(text, language, mode="placeholder")

    # Unknown mode, be safe
    return text


