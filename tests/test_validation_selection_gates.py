from types import SimpleNamespace

from src.pipeline.task_executor.stage_handlers.validation_handler import ValidationHandler


class _FakeAudio:
    def __init__(self, samples: int) -> None:
        self.shape = (samples,)

    def numel(self) -> int:
        return int(self.shape[-1])


def _handler() -> ValidationHandler:
    handler = ValidationHandler.__new__(ValidationHandler)
    handler.config = {
        "audio": {"sample_rate": 24000},
        "validation": {
            "prosody": {
                "targets": {
                    "wpm_min": 115,
                    "wpm_max": 155,
                }
            },
            "selection": {
                "gating": {
                    "require_similarity": True,
                    "require_mos": False,
                    "duration": {
                        "enabled": True,
                        "min_word_count": 12,
                        "max_duration_ratio": 2.0,
                    },
                }
            },
            "mos": {"min_mos": 1.5},
        },
    }
    return handler


def test_duration_gate_rejects_token_cap_loop() -> None:
    """A long token-cap loop must fail even when similarity passed."""
    handler = _handler()
    samples = int(327.6 * 24000)
    candidate = SimpleNamespace(audio_tensor=_FakeAudio(samples))
    original_text = " ".join(["word"] * 180)

    passes_mos, passes_similarity, passes_duration, duration_gate, passes_mos_threshold, final_valid = (
        handler._compute_selection_gates(
            similarity_valid=True,
            prosody_details={"enabled": True, "raw_mos": 0.4, "wpm": 33.0},
            candidate=candidate,
            original_text=original_text,
        )
    )

    assert passes_mos is True
    assert passes_mos_threshold is False
    assert passes_similarity is True
    assert passes_duration is False
    assert final_valid is False
    assert "max_duration_ratio" in duration_gate["reasons"]


def test_duration_gate_accepts_target_length_audio() -> None:
    """A candidate near the configured WPM target remains selectable."""
    handler = _handler()
    samples = int(80.0 * 24000)
    candidate = SimpleNamespace(audio_tensor=_FakeAudio(samples))
    original_text = " ".join(["word"] * 180)

    passes_mos, passes_similarity, passes_duration, duration_gate, passes_mos_threshold, final_valid = (
        handler._compute_selection_gates(
            similarity_valid=True,
            prosody_details={"enabled": True, "raw_mos": 0.4, "wpm": 135.0},
            candidate=candidate,
            original_text=original_text,
        )
    )

    assert passes_mos is True
    assert passes_mos_threshold is False
    assert passes_similarity is True
    assert passes_duration is True
    assert final_valid is True
    assert duration_gate["reasons"] == []
