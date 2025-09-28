#!/usr/bin/env python3
"""
TaskRunLogger - Persistente, robuste Laufzeitprotokollierung pro Task.

Speichert fortlaufend:
- total_execution_seconds: kumulierte Laufzeit über alle Sessions
- last_session_seconds: Laufzeit der aktuellen/letzten Session
- runs_count: Anzahl abgeschlossener Sessions

Schreibt zusätzlich spärliche Ereignisse in eine JSONL-Datei.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RuntimeState:
    total_execution_seconds: float = 0.0
    last_session_seconds: float = 0.0
    runs_count: int = 0
    last_updated: str = ""
    last_run_info: Optional[Dict[str, Any]] = None
    last_status: Optional[str] = None  # "success" | "failed" | None
    last_final_audio_path: Optional[str] = None


class TaskRunLogger:
    """Persistenter Laufzeit-Logger pro Task-Verzeichnis."""

    def __init__(self, task_directory: Path):
        self.task_directory = task_directory
        self.runtime_file = self.task_directory / "task_runtime.json"
        self.events_file = self.task_directory / "task_events.log.jsonl"

        self._state = self._load_state()
        self._session_start_time: Optional[float] = None
        self._last_flush_time: Optional[float] = None
        self._session_active: bool = False

    # --- Public API ---
    @property
    def total_execution_seconds(self) -> float:
        return float(self._state.total_execution_seconds)

    @property
    def last_session_seconds(self) -> float:
        return float(self._state.last_session_seconds)

    def start_session(self, run_info: Dict[str, Any]) -> None:
        """
        Beginnt eine Session und protokolliert Start-Ereignis.
        run_info kann Felder enthalten wie: run_type, force_final, rerender_all, gap_filling, etc.
        """
        if self._session_active:
            # Idempotent: nichts tun, falls bereits aktiv
            logger.debug("TaskRunLogger.start_session called while session active; ignoring")
            return

        now = time.time()
        self._session_start_time = now
        self._last_flush_time = now
        self._session_active = True
        self._state.last_session_seconds = 0.0
        self._state.last_run_info = dict(run_info) if run_info else {}
        self._state.last_status = None
        self._state.last_final_audio_path = None
        self._state.last_updated = _iso_now()
        self._append_event("session_start", {"run_info": self._state.last_run_info})
        self._persist_state()

    def tick(self) -> None:
        """
        Aktualisiert die Laufzeiten und persistiert sie. Kann gefahrlos oft aufgerufen werden.
        """
        if not self._session_active or self._last_flush_time is None:
            return
        now = time.time()
        delta = max(0.0, now - self._last_flush_time)

        # Schutz gegen unrealistische Deltas (z.B. Systemuhränderungen)
        if delta > 60 * 60 * 12:  # >12h Schritt ignorieren
            logger.warning(f"Ignoring anomalous time delta in TaskRunLogger.tick: {delta:.2f}s")
            delta = 0.0

        self._state.last_session_seconds += delta
        self._state.total_execution_seconds += delta
        self._state.last_updated = _iso_now()
        self._last_flush_time = now
        self._persist_state()

    def add_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        self._append_event(event_type, data or {})

    def end_session(
        self,
        success: bool,
        final_audio_path: Optional[Path] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Beendet die Session, finalisiert die Laufzeiten und persistiert den Status.
        """
        if not self._session_active:
            return

        # finaler Tick
        self.tick()

        self._state.runs_count += 1
        self._state.last_status = "success" if success else "failed"
        self._state.last_final_audio_path = (
            str(final_audio_path) if final_audio_path else None
        )
        payload: Dict[str, Any] = {
            "success": success,
            "session_seconds": self._state.last_session_seconds,
        }
        if final_audio_path:
            payload["final_audio_path"] = str(final_audio_path)
        if error_message:
            payload["error_message"] = error_message
        self._append_event("session_end", payload)

        self._persist_state()
        # Session schließen
        self._session_start_time = None
        self._last_flush_time = None
        self._session_active = False

    # --- Internals ---
    def _load_state(self) -> RuntimeState:
        try:
            if self.runtime_file.exists():
                with self.runtime_file.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                return RuntimeState(
                    total_execution_seconds=float(raw.get("total_execution_seconds", 0.0)),
                    last_session_seconds=float(raw.get("last_session_seconds", 0.0)),
                    runs_count=int(raw.get("runs_count", 0)),
                    last_updated=str(raw.get("last_updated", "")),
                    last_run_info=raw.get("last_run_info"),
                    last_status=raw.get("last_status"),
                    last_final_audio_path=raw.get("last_final_audio_path"),
                )
        except Exception as e:
            logger.warning(f"Failed to load runtime state: {e}")
        return RuntimeState()

    def _persist_state(self) -> None:
        try:
            tmp = self.runtime_file.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(asdict(self._state), f, ensure_ascii=False, indent=2)
            tmp.replace(self.runtime_file)
        except Exception as e:
            logger.error(f"Failed to persist runtime state: {e}")

    def _append_event(self, event_type: str, data: Dict[str, Any]) -> None:
        try:
            self.events_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {"ts": _iso_now(), "event": event_type, **(data or {})}
            with self.events_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to append event to {self.events_file}: {e}")


