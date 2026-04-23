#!/usr/bin/env python3
"""
Schema-level tests for TaskMetricsGenerator.generate_analysis_metrics().

Only field names and types are checked – no concrete numeric values –
so the test runs without any real audio or model data.

Run with:
    PYTHONPATH=src python -m pytest tests/test_analysis_metrics_generator.py -v
"""
import json
from pathlib import Path
from typing import Any, Dict

import pytest


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

def _make_candidate(idx_0: int, include_prosody: bool = True) -> Dict[str, Any]:
    """Return a minimal whisper-metrics candidate entry (0-based idx)."""
    prosody: Any = {
        "enabled": True,
        "subscores": {
            "semantic_alignment": 1.0,
            "flow": 0.5,
            "liveliness": 0.7,
            "intelligibility": 0.8,
            "mos": 0.4,
        },
        "prosody_score": 0.55,
        "raw_mos": 3.5,
        "liveliness_raw": 0.45,
        "wpm": 130.0,
    } if include_prosody else {"enabled": False}

    return {
        "transcription": "Some transcription text.",
        "quality_details": {
            "candidate_id": f"chunk_0_candidate_{idx_0}",
            "audio_duration": 4.0 + idx_0 * 0.5,
            "expected_duration": None,
            "validation_passed": True,
            "individual_scores": {
                "similarity_score": 0.90,
                "length_score": 0.93,
                "penalty_score": 0.0,
                "overall_score": 0.88,
            },
            "validation_metrics": {
                "asr_backend": "whisper",
                "whisper_similarity": 0.90,
                "whisper_quality": 0.87,
                "transcription_length": 25,
                "original_text_length": 28,
                "normalized_transcription": "Some transcription text.",
                "normalization_language": "de",
                "numbers_normalization_mode": "placeholder",
                "base_similarity_threshold": 0.7,
                "effective_similarity_threshold": 0.68,
            },
        },
        "overall_quality_score": 0.88,
        "final_selection_score": 0.75,
        "prosody": prosody,
        "is_valid": True,
        "passes_mos_gate": True,
        "passes_similarity_gate": True,
    }


def _make_whisper_metrics(num_chunks: int, num_candidates: int) -> Dict[str, Any]:
    chunks = {}
    for chunk_idx in range(num_chunks):
        candidates = {
            str(ci): _make_candidate(ci) for ci in range(num_candidates)
        }
        chunks[str(chunk_idx)] = {
            "text": "Chunk text that should not appear in analysis_metrics.",
            "chunk_text": "Chunk text.",
            "speaker_id": "spk_A",
            "language_id": "de",
            "candidates": candidates,
        }
    return {"timestamp": 1000000.0, "total_chunks": num_chunks, "chunks": chunks}


def _make_chunks_metadata(num_chunks: int) -> Dict[str, Any]:
    chunks = [
        {
            "idx": i,
            "text_length": 200 + i * 10,
            "excerpt": "Some excerpt …",
            "is_paragraph_break": False,
            "paragraph_break_type": None,
            "filename": f"chunk_{i + 1:03d}.txt",
            "speaker_id": "spk_A",
            "speaker_transition": False,
            "original_markup": None,
            "speaker_transition_context": None,
        }
        for i in range(num_chunks)
    ]
    return {"total_chunks": num_chunks, "chunks": chunks}


def _make_task_metrics(num_chunks: int) -> Dict[str, Any]:
    """Minimal task_metrics.json so selected_candidates can be read back."""
    return {
        "job_name": "test_job",
        "task_name": "test_task_20260101_120000",
        "run_label": "unit-test",
        "timestamp": "20260101_120000",
        "selected_candidates": {str(i + 1): 1 for i in range(num_chunks)},
        "user_selected_chunks": [],
        "chunks": [],
        "summary": {},
        "task_runtime": {},
    }


def _make_task_runtime() -> Dict[str, Any]:
    return {
        "total_execution_seconds": 3600.0,
        "last_status": "success",
    }


def _make_config() -> Dict[str, Any]:
    return {
        "job": {"name": "test_job", "run_label": "unit-test"},
        "generation": {
            "model_type": "qwen3",
            "global_seed": 0,
            "seed_fixed": False,
            "num_candidates": 2,
            "default_speaker": "spk_A",
            "speakers": [
                {
                    "id": "spk_A",
                    "reference_audio": "speaker_a.wav",
                    "language": "de",
                    "tts_params": {
                        "temperature": 1.15,
                        "top_k": 120,
                        "top_p": 0.99,
                    },
                }
            ],
        },
    }


def _make_candidates_metadata(num_candidates: int) -> Dict[str, Any]:
    candidates = [
        {
            "candidate_idx": ci,
            "audio_filename": f"candidate_{ci + 1:02d}.wav",
            "audio_duration": 4.0 + ci * 0.5,
            "generation_params": {
                "temperature": 1.15 + ci * 0.05,
                "top_k": 120 + ci * 6,
                "top_p": 0.99,
                "repetition_penalty": 1.07,
                "subtalker_temperature": 1.4 - ci * 0.04,
                "subtalker_top_k": 170 + ci * 4,
                "subtalker_top_p": 0.97,
                "language_id": "de",     # must be excluded
                "type": "EXPRESSIVE",
                "seed": 9584 + ci * 1000,  # must be excluded
            },
        }
        for ci in range(num_candidates)
    ]
    return {"chunk_idx": 0, "total_candidates": num_candidates, "candidates": candidates}


# ---------------------------------------------------------------------------
# Fixture: minimal task directory with all required sub-files
# ---------------------------------------------------------------------------

@pytest.fixture
def task_dir(tmp_path: Path) -> Path:
    num_chunks = 3
    num_candidates = 2

    whisper_dir = tmp_path / "whisper"
    whisper_dir.mkdir()
    (whisper_dir / "whisper_metrics.json").write_text(
        json.dumps(_make_whisper_metrics(num_chunks, num_candidates)), encoding="utf-8"
    )

    texts_dir = tmp_path / "texts"
    texts_dir.mkdir()
    (texts_dir / "chunks_metadata.json").write_text(
        json.dumps(_make_chunks_metadata(num_chunks)), encoding="utf-8"
    )

    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    for chunk_idx in range(num_chunks):
        chunk_dir = candidates_dir / f"chunk_{chunk_idx + 1:03d}"
        chunk_dir.mkdir()
        (chunk_dir / "candidates_metadata.json").write_text(
            json.dumps(_make_candidates_metadata(num_candidates)), encoding="utf-8"
        )

    (tmp_path / "task_runtime.json").write_text(
        json.dumps(_make_task_runtime()), encoding="utf-8"
    )
    (tmp_path / "task_metrics.json").write_text(
        json.dumps(_make_task_metrics(num_chunks)), encoding="utf-8"
    )

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_generate_analysis_metrics_returns_true(task_dir: Path) -> None:
    """generate_analysis_metrics() must succeed and return True."""
    from src.utils.file_manager.task_metrics_generator import TaskMetricsGenerator

    tmg = TaskMetricsGenerator(task_dir, _make_config())
    result = tmg.generate_analysis_metrics()
    assert result is True


def test_analysis_metrics_file_is_created(task_dir: Path) -> None:
    """analysis_metrics.json must be written to the task directory."""
    from src.utils.file_manager.task_metrics_generator import TaskMetricsGenerator

    tmg = TaskMetricsGenerator(task_dir, _make_config())
    tmg.generate_analysis_metrics()
    assert (task_dir / "analysis_metrics.json").exists()


def test_top_level_schema_fields(task_dir: Path) -> None:
    """Top-level keys and their types must match the schema definition."""
    from src.utils.file_manager.task_metrics_generator import (
        ANALYSIS_METRICS_SCHEMA_VERSION,
        TaskMetricsGenerator,
    )

    TaskMetricsGenerator(task_dir, _make_config()).generate_analysis_metrics()
    data = json.loads((task_dir / "analysis_metrics.json").read_text(encoding="utf-8"))

    assert data["schema_version"] == ANALYSIS_METRICS_SCHEMA_VERSION
    assert isinstance(data["task"], dict)
    assert isinstance(data["speakers"], dict)
    assert isinstance(data["chunks"], list)


def test_task_section_field_types(task_dir: Path) -> None:
    """All fields in the ``task`` section must have the expected types."""
    from src.utils.file_manager.task_metrics_generator import TaskMetricsGenerator

    TaskMetricsGenerator(task_dir, _make_config()).generate_analysis_metrics()
    task = json.loads(
        (task_dir / "analysis_metrics.json").read_text(encoding="utf-8")
    )["task"]

    assert isinstance(task["job_name"], str)
    assert isinstance(task["task_name"], str)
    assert isinstance(task["run_label"], str)
    assert isinstance(task["timestamp"], str)
    assert isinstance(task["model_type"], str)
    assert task["global_seed"] is None or isinstance(task["global_seed"], int)
    assert isinstance(task["seed_fixed"], bool)
    assert isinstance(task["total_chunks"], int)
    assert isinstance(task["total_candidates_generated"], int)
    # task_runtime_seconds: int or None
    assert task["task_runtime_seconds"] is None or isinstance(task["task_runtime_seconds"], int)
    assert task["global_seed"] == 0
    assert task["seed_fixed"] is False


def test_task_global_seed_and_seed_fixed_from_config(task_dir: Path) -> None:
    """Task section reflects generation.global_seed and generation.seed_fixed."""
    from src.utils.file_manager.task_metrics_generator import TaskMetricsGenerator

    cfg = _make_config()
    cfg["generation"]["global_seed"] = 12345
    cfg["generation"]["seed_fixed"] = True

    TaskMetricsGenerator(task_dir, cfg).generate_analysis_metrics()
    task = json.loads(
        (task_dir / "analysis_metrics.json").read_text(encoding="utf-8")
    )["task"]
    assert task["global_seed"] == 12345
    assert task["seed_fixed"] is True


def test_candidate_torch_seed_from_metadata(task_dir: Path) -> None:
    """Each analysis candidate carries the effective torch seed (not in generation_params)."""
    from src.utils.file_manager.task_metrics_generator import TaskMetricsGenerator

    TaskMetricsGenerator(task_dir, _make_config()).generate_analysis_metrics()
    chunks = json.loads(
        (task_dir / "analysis_metrics.json").read_text(encoding="utf-8")
    )["chunks"]

    for chunk in chunks:
        for ci, cand in enumerate(chunk["candidates"]):
            assert "torch_seed" in cand
            assert cand["torch_seed"] == 9584 + ci * 1000
            assert "seed" not in cand["generation_params"]


def test_speakers_section(task_dir: Path) -> None:
    """Speakers dict must contain one entry per distinct speaker_id."""
    from src.utils.file_manager.task_metrics_generator import TaskMetricsGenerator

    TaskMetricsGenerator(task_dir, _make_config()).generate_analysis_metrics()
    data = json.loads((task_dir / "analysis_metrics.json").read_text(encoding="utf-8"))

    speakers = data["speakers"]
    assert "spk_A" in speakers
    spk = speakers["spk_A"]
    assert isinstance(spk["chunk_count"], int)
    assert isinstance(spk["num_candidates_per_chunk"], int)
    assert isinstance(spk["reference_audio"], str)
    assert isinstance(spk["language"], str)
    assert "ramp_spec" in spk
    assert spk["ramp_spec"] == {}


def test_ramp_spec_negative_subtalker_temperature(task_dir: Path) -> None:
    """ramp_spec preserves sign of max_deviation; end = base + max_deviation."""
    from src.utils.file_manager.task_metrics_generator import TaskMetricsGenerator

    cfg = _make_config()
    cfg["generation"]["speakers"][0]["tts_params"] = {
        "subtalker_temperature": 1.45,
        "subtalker_temperature_max_deviation": -0.30,
    }

    TaskMetricsGenerator(task_dir, cfg).generate_analysis_metrics()
    spk = json.loads(
        (task_dir / "analysis_metrics.json").read_text(encoding="utf-8")
    )["speakers"]["spk_A"]

    assert spk["ramp_spec"]["subtalker_temperature"] == {
        "base": 1.45,
        "max_deviation": -0.3,
        "end": 1.15,
    }


def test_ramp_spec_qwen3_omits_inherited_chatterbox_axes(task_dir: Path) -> None:
    """Qwen3 ramp_spec must not list exaggeration/cfg_weight (not used by the model)."""
    from src.utils.file_manager.task_metrics_generator import TaskMetricsGenerator

    cfg = _make_config()
    cfg["generation"]["speakers"][0]["tts_params"] = {
        "temperature": 1.1,
        "temperature_max_deviation": 0.2,
        "exaggeration": 0.45,
        "exaggeration_max_deviation": -0.1,
        "cfg_weight": 0.15,
        "cfg_weight_max_deviation": 0.2,
    }

    TaskMetricsGenerator(task_dir, cfg).generate_analysis_metrics()
    spk = json.loads(
        (task_dir / "analysis_metrics.json").read_text(encoding="utf-8")
    )["speakers"]["spk_A"]

    rs = spk["ramp_spec"]
    assert "exaggeration" not in rs
    assert "cfg_weight" not in rs
    assert "temperature" in rs


def test_ramp_spec_chatterbox_includes_exaggeration(task_dir: Path) -> None:
    """Chatterbox family: exaggeration appears when it has a non-zero deviation."""
    from src.utils.file_manager.task_metrics_generator import TaskMetricsGenerator

    cfg = _make_config()
    cfg["generation"]["model_type"] = "standard"
    cfg["generation"]["speakers"][0]["tts_params"] = {
        "exaggeration": 0.45,
        "exaggeration_max_deviation": -0.1,
        "temperature": 1.0,
        "temperature_max_deviation": 0,
    }

    TaskMetricsGenerator(task_dir, cfg).generate_analysis_metrics()
    spk = json.loads(
        (task_dir / "analysis_metrics.json").read_text(encoding="utf-8")
    )["speakers"]["spk_A"]

    assert "exaggeration" in spk["ramp_spec"]
    assert "temperature" not in spk["ramp_spec"]


def test_ramp_spec_empty_when_all_max_deviation_zero(task_dir: Path) -> None:
    """A speaker with no ramp axes (all deviations 0) gets ramp_spec: {} not null."""
    from src.utils.file_manager.task_metrics_generator import TaskMetricsGenerator

    cfg = _make_config()
    cfg["generation"]["speakers"][0]["tts_params"] = {
        "temperature": 1.1,
        "temperature_max_deviation": 0,
        "subtalker_top_k": 100,
        "subtalker_top_k_max_deviation": 0,
    }

    TaskMetricsGenerator(task_dir, cfg).generate_analysis_metrics()
    spk = json.loads(
        (task_dir / "analysis_metrics.json").read_text(encoding="utf-8")
    )["speakers"]["spk_A"]

    assert spk["ramp_spec"] is not None
    assert spk["ramp_spec"] == {}


def test_chunks_section_structure(task_dir: Path) -> None:
    """Each chunk entry must have the required keys with correct types."""
    from src.utils.file_manager.task_metrics_generator import TaskMetricsGenerator

    TaskMetricsGenerator(task_dir, _make_config()).generate_analysis_metrics()
    chunks = json.loads(
        (task_dir / "analysis_metrics.json").read_text(encoding="utf-8")
    )["chunks"]

    assert len(chunks) == 3
    for chunk in chunks:
        assert isinstance(chunk["chunk_idx"], int)
        assert isinstance(chunk["speaker_id"], str)
        assert isinstance(chunk["text_length"], int)
        assert isinstance(chunk["selected_candidate"], int)
        assert isinstance(chunk["candidates"], list)
        for cand in chunk["candidates"]:
            assert "torch_seed" in cand
            assert cand["torch_seed"] is None or isinstance(cand["torch_seed"], int)


def test_chunk_idx_is_1based_and_ascending(task_dir: Path) -> None:
    """chunk_idx values must be 1-based and in ascending order."""
    from src.utils.file_manager.task_metrics_generator import TaskMetricsGenerator

    TaskMetricsGenerator(task_dir, _make_config()).generate_analysis_metrics()
    chunks = json.loads(
        (task_dir / "analysis_metrics.json").read_text(encoding="utf-8")
    )["chunks"]

    indices = [c["chunk_idx"] for c in chunks]
    assert indices == sorted(indices)
    assert indices[0] >= 1


def test_candidate_idx_is_1based_and_ascending(task_dir: Path) -> None:
    """Candidate idx values inside each chunk must be 1-based and ascending."""
    from src.utils.file_manager.task_metrics_generator import TaskMetricsGenerator

    TaskMetricsGenerator(task_dir, _make_config()).generate_analysis_metrics()
    chunks = json.loads(
        (task_dir / "analysis_metrics.json").read_text(encoding="utf-8")
    )["chunks"]

    for chunk in chunks:
        cand_indices = [c["idx"] for c in chunk["candidates"]]
        assert cand_indices == sorted(cand_indices)
        assert cand_indices[0] >= 1


def test_candidate_scores_field_types(task_dir: Path) -> None:
    """All score fields must be float or None; prosody fields present (not missing)."""
    from src.utils.file_manager.task_metrics_generator import TaskMetricsGenerator

    TaskMetricsGenerator(task_dir, _make_config()).generate_analysis_metrics()
    chunks = json.loads(
        (task_dir / "analysis_metrics.json").read_text(encoding="utf-8")
    )["chunks"]

    score_keys = [
        "final_selection_score",
        "overall_quality_score",
        "whisper_similarity",
        "whisper_quality",
        "length_score",
        "penalty_score",
        "prosody_score",
        "prosody_flow",
        "prosody_liveliness",
        "prosody_intelligibility",
        "prosody_mos",
        "raw_mos",
        "wpm",
    ]
    for chunk in chunks:
        for cand in chunk["candidates"]:
            scores = cand["scores"]
            for key in score_keys:
                assert key in scores, f"Missing score key: {key}"
                val = scores[key]
                assert val is None or isinstance(val, (int, float)), (
                    f"Score '{key}' has unexpected type {type(val)}"
                )


def test_candidate_gates_field_types(task_dir: Path) -> None:
    """Gate fields must be bool or None; all three gate keys must be present."""
    from src.utils.file_manager.task_metrics_generator import TaskMetricsGenerator

    TaskMetricsGenerator(task_dir, _make_config()).generate_analysis_metrics()
    chunks = json.loads(
        (task_dir / "analysis_metrics.json").read_text(encoding="utf-8")
    )["chunks"]

    gate_keys = ["is_valid", "passes_mos_gate", "passes_similarity_gate"]
    for chunk in chunks:
        for cand in chunk["candidates"]:
            gates = cand["gates"]
            for key in gate_keys:
                assert key in gates
                assert gates[key] is None or isinstance(gates[key], bool)


def test_generation_params_excludes_seed_and_language_id(task_dir: Path) -> None:
    """generation_params must not contain 'seed' or 'language_id'."""
    from src.utils.file_manager.task_metrics_generator import TaskMetricsGenerator

    TaskMetricsGenerator(task_dir, _make_config()).generate_analysis_metrics()
    chunks = json.loads(
        (task_dir / "analysis_metrics.json").read_text(encoding="utf-8")
    )["chunks"]

    for chunk in chunks:
        for cand in chunk["candidates"]:
            params = cand["generation_params"]
            assert "seed" not in params
            assert "language_id" not in params


def test_prosody_fields_are_null_when_disabled(task_dir: Path) -> None:
    """When prosody is disabled, prosody_* fields must be null (not missing)."""
    import copy
    from src.utils.file_manager.task_metrics_generator import TaskMetricsGenerator

    # Overwrite whisper_metrics with prosody disabled
    num_chunks, num_cands = 2, 2
    wm = _make_whisper_metrics(num_chunks, num_cands)
    for chunk_key in wm["chunks"]:
        for cand_key in wm["chunks"][chunk_key]["candidates"]:
            wm["chunks"][chunk_key]["candidates"][cand_key]["prosody"] = {"enabled": False}

    (task_dir / "whisper" / "whisper_metrics.json").write_text(
        json.dumps(wm), encoding="utf-8"
    )
    # Update task_metrics.json chunk count
    (task_dir / "task_metrics.json").write_text(
        json.dumps(_make_task_metrics(num_chunks)), encoding="utf-8"
    )

    TaskMetricsGenerator(task_dir, _make_config()).generate_analysis_metrics()
    chunks = json.loads(
        (task_dir / "analysis_metrics.json").read_text(encoding="utf-8")
    )["chunks"]

    prosody_keys = [
        "prosody_score", "prosody_flow", "prosody_liveliness",
        "prosody_intelligibility", "prosody_mos", "raw_mos", "wpm",
    ]
    for chunk in chunks:
        for cand in chunk["candidates"]:
            scores = cand["scores"]
            for key in prosody_keys:
                assert key in scores, f"Key '{key}' missing when prosody disabled"
                assert scores[key] is None, (
                    f"Expected null for '{key}' when prosody disabled, got {scores[key]}"
                )


def test_task_metrics_json_unchanged(task_dir: Path) -> None:
    """generate_analysis_metrics() must not modify task_metrics.json."""
    task_metrics_path = task_dir / "task_metrics.json"
    original_bytes = task_metrics_path.read_bytes()

    from src.utils.file_manager.task_metrics_generator import TaskMetricsGenerator

    TaskMetricsGenerator(task_dir, _make_config()).generate_analysis_metrics()

    assert task_metrics_path.read_bytes() == original_bytes


def test_generate_analysis_metrics_without_whisper_returns_false(
    tmp_path: Path,
) -> None:
    """Returns False gracefully when whisper_metrics.json is absent."""
    from src.utils.file_manager.task_metrics_generator import TaskMetricsGenerator

    (tmp_path / "whisper").mkdir()
    (tmp_path / "texts").mkdir()
    (tmp_path / "candidates").mkdir()

    tmg = TaskMetricsGenerator(tmp_path, _make_config())
    result = tmg.generate_analysis_metrics()
    assert result is False
    assert not (tmp_path / "analysis_metrics.json").exists()
