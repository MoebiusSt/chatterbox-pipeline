#!/usr/bin/env python3
"""
Tests for speaker-specific cascading configuration merging with explicit default_speaker key.
Tests all scenarios mentioned in the user's question about speaker config merging.
"""

import pytest
from pathlib import Path
from src.utils.config_manager import ConfigManager
import tempfile
import yaml
import logging

# Set up logging for test debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestSpeakerConfigMerging:
    """Test cases for speaker-specific cascading configuration merging with explicit default_speaker key."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            config_dir.mkdir()
            yield config_dir
    
    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create ConfigManager with temporary directory."""
        project_root = temp_config_dir.parent
        return ConfigManager(project_root)
    
    def create_default_config(self, config_dir: Path, speakers_config: list, default_speaker: str):
        """Helper to create default_config.yaml with specified speakers and default_speaker."""
        default_config = {
            "job": {"name": "default"},
            "input": {"text_file": "test.txt"},
            "chunking": {"target_chunk_limit": 180},
            "generation": {
                "num_candidates": 3,
                "max_retries": 1,
                "default_speaker": default_speaker,
                "speakers": speakers_config
            },
            "validation": {"similarity_threshold": 0.8},
            "audio": {"silence_duration": {"normal": 0.4}}
        }
        
        with open(config_dir / "default_config.yaml", "w") as f:
            yaml.dump(default_config, f)
        
        return default_config
    
    def create_job_config(self, config_dir: Path, filename: str, config_data: dict):
        """Helper to create job config file."""
        with open(config_dir / filename, "w") as f:
            yaml.dump(config_data, f)
    
    def test_scenario_1_only_reference_audio_override(self, config_manager, temp_config_dir):
        """
        Scenario 1: Job config only overrides reference_audio, tts_params should be inherited.
        """
        # Create default config with full speaker configuration
        default_speakers = [
            {
                "id": "david",
                "reference_audio": "david_barnes_1.wav",
                "tts_params": {
                    "exaggeration": 0.55,
                    "cfg_weight": 0.2,
                    "temperature": 0.9,
                    "repetition_penalty": 1.3
                },
                "conservative_candidate": {
                    "enabled": True,
                    "exaggeration": 0.4,
                    "cfg_weight": 0.5,
                    "temperature": 0.7
                }
            }
        ]
        
        self.create_default_config(temp_config_dir, default_speakers, "david")
        
        # Create job config that only overrides reference_audio for default speaker
        job_config = {
            "generation": {
                "speakers": [
                    {
                        "id": "default",  # Using alias for default speaker
                        "reference_audio": "cori_samuel_1.wav"
                        # tts_params intentionally missing!
                    }
                ]
            }
        }
        
        self.create_job_config(temp_config_dir, "test_job.yaml", job_config)
        
        # Load and merge configs
        merged_config = config_manager.load_cascading_config(temp_config_dir / "test_job.yaml")
        
        # Verify results
        speakers = merged_config["generation"]["speakers"]
        assert len(speakers) == 2  # Both david (original) and default (alias) available
        
        # Find both speakers
        default_speaker = next(s for s in speakers if s["id"] == "default")
        david_speaker = next(s for s in speakers if s["id"] == "david")
        
        # Check default (alias) speaker
        assert default_speaker["reference_audio"] == "cori_samuel_1.wav"  # Overridden
        
        # tts_params should be inherited from david
        assert default_speaker["tts_params"]["exaggeration"] == 0.55
        assert default_speaker["tts_params"]["cfg_weight"] == 0.2
        assert default_speaker["tts_params"]["temperature"] == 0.9
        assert default_speaker["tts_params"]["repetition_penalty"] == 1.3
        
        # conservative_candidate should be inherited from david
        assert default_speaker["conservative_candidate"]["enabled"] is True
        assert default_speaker["conservative_candidate"]["exaggeration"] == 0.4
        
        # Check david (original) speaker is still available
        assert david_speaker["reference_audio"] == "david_barnes_1.wav"  # Original
        assert david_speaker["tts_params"]["exaggeration"] == 0.55  # Original
        
        # Both speakers should have identical tts_params (inherited)
        assert default_speaker["tts_params"] == david_speaker["tts_params"]
        
        # Default speaker should still be correct
        assert merged_config["generation"]["default_speaker"] == "david"
        
        logger.info("✅ Scenario 1 passed: reference_audio overridden, tts_params inherited")
    
    def test_scenario_2_change_default_speaker(self, config_manager, temp_config_dir):
        """
        Scenario 2: Job config changes the default_speaker from david to narrator.
        """
        # Create default config with multiple speakers, david as default
        default_speakers = [
            {
                "id": "david",
                "reference_audio": "david.wav",
                "tts_params": {
                    "exaggeration": 0.5,
                    "cfg_weight": 0.2,
                    "temperature": 0.9
                }
            },
            {
                "id": "narrator",
                "reference_audio": "narrator.wav",
                "tts_params": {
                    "exaggeration": 0.65,
                    "cfg_weight": 0.3,
                    "temperature": 1.0
                }
            }
        ]
        
        self.create_default_config(temp_config_dir, default_speakers, "david")
        
        # Create job config that changes default_speaker to narrator
        job_config = {
            "generation": {
                "default_speaker": "narrator"
            }
        }
        
        self.create_job_config(temp_config_dir, "test_job.yaml", job_config)
        
        # Load and merge configs
        merged_config = config_manager.load_cascading_config(temp_config_dir / "test_job.yaml")
        
        # Verify results
        assert merged_config["generation"]["default_speaker"] == "narrator"
        
        # All speakers should still be present
        speakers = merged_config["generation"]["speakers"]
        assert len(speakers) == 2
        
        speaker_ids = [s["id"] for s in speakers]
        assert "david" in speaker_ids
        assert "narrator" in speaker_ids
        
        logger.info("✅ Scenario 2 passed: default_speaker changed to narrator")
    
    def test_scenario_3_override_non_default_speaker(self, config_manager, temp_config_dir):
        """
        Scenario 3: Job config overrides a non-default speaker (narrator) while keeping default.
        """
        # Create default config with multiple speakers
        default_speakers = [
            {
                "id": "david",
                "reference_audio": "david.wav",
                "tts_params": {
                    "exaggeration": 0.5,
                    "cfg_weight": 0.2,
                    "temperature": 0.9
                }
            },
            {
                "id": "narrator",
                "reference_audio": "narrator.wav",
                "tts_params": {
                    "exaggeration": 0.65,
                    "cfg_weight": 0.3,
                    "temperature": 1.0
                }
            }
        ]
        
        self.create_default_config(temp_config_dir, default_speakers, "david")
        
        # Create job config that overrides narrator only
        job_config = {
            "generation": {
                "speakers": [
                    {
                        "id": "narrator",
                        "reference_audio": "new_narrator.wav"
                    }
                ]
            }
        }
        
        self.create_job_config(temp_config_dir, "test_job.yaml", job_config)
        
        # Load and merge configs
        merged_config = config_manager.load_cascading_config(temp_config_dir / "test_job.yaml")
        
        # Verify results
        speakers = merged_config["generation"]["speakers"]
        assert len(speakers) == 2  # Both speakers should be present
        
        # Find narrator speaker
        narrator_speaker = next(s for s in speakers if s["id"] == "narrator")
        assert narrator_speaker["reference_audio"] == "new_narrator.wav"  # Overridden
        
        # Should inherit tts_params from original narrator config
        assert narrator_speaker["tts_params"]["exaggeration"] == 0.65
        assert narrator_speaker["tts_params"]["cfg_weight"] == 0.3
        assert narrator_speaker["tts_params"]["temperature"] == 1.0
        
        # David should be unchanged
        david_speaker = next(s for s in speakers if s["id"] == "david")
        assert david_speaker["reference_audio"] == "david.wav"
        
        # Default speaker should still be david
        assert merged_config["generation"]["default_speaker"] == "david"
        
        logger.info("✅ Scenario 3 passed: Non-default speaker overridden correctly")
    
    def test_scenario_4_complex_speaker_and_default_change(self, config_manager, temp_config_dir):
        """
        Scenario 4: Complex case - change default_speaker and override multiple speakers.
        """
        # Create default config with multiple speakers
        default_speakers = [
            {
                "id": "david",
                "reference_audio": "david.wav",
                "tts_params": {
                    "exaggeration": 0.5,
                    "cfg_weight": 0.2,
                    "temperature": 0.9
                }
            },
            {
                "id": "narrator",
                "reference_audio": "narrator.wav",
                "tts_params": {
                    "exaggeration": 0.65,
                    "cfg_weight": 0.3,
                    "temperature": 1.0
                }
            },
            {
                "id": "character",
                "reference_audio": "character.wav",
                "tts_params": {
                    "exaggeration": 0.7,
                    "cfg_weight": 0.4,
                    "temperature": 1.1
                }
            }
        ]
        
        self.create_default_config(temp_config_dir, default_speakers, "david")
        
        # Create job config that changes default speaker and overrides multiple speakers
        job_config = {
            "generation": {
                "default_speaker": "narrator",
                "speakers": [
                    {
                        "id": "david",
                        "reference_audio": "new_david.wav"
                    },
                    {
                        "id": "character",
                        "reference_audio": "new_character.wav",
                        "tts_params": {
                            "exaggeration": 0.8  # Partial override
                        }
                    }
                ]
            }
        }
        
        self.create_job_config(temp_config_dir, "test_job.yaml", job_config)
        
        # Load and merge configs
        merged_config = config_manager.load_cascading_config(temp_config_dir / "test_job.yaml")
        
        # Verify results
        assert merged_config["generation"]["default_speaker"] == "narrator"
        
        speakers = merged_config["generation"]["speakers"]
        assert len(speakers) == 3  # All speakers should be present
        
        # Check david - overridden reference_audio, inherited tts_params
        david = next(s for s in speakers if s["id"] == "david")
        assert david["reference_audio"] == "new_david.wav"
        assert david["tts_params"]["exaggeration"] == 0.5  # Inherited
        
        # Check narrator - unchanged (not overridden in job config)
        narrator = next(s for s in speakers if s["id"] == "narrator")
        assert narrator["reference_audio"] == "narrator.wav"
        assert narrator["tts_params"]["exaggeration"] == 0.65
        
        # Check character - partial override
        character = next(s for s in speakers if s["id"] == "character")
        assert character["reference_audio"] == "new_character.wav"
        assert character["tts_params"]["exaggeration"] == 0.8  # Overridden
        assert character["tts_params"]["cfg_weight"] == 0.4  # Inherited
        assert character["tts_params"]["temperature"] == 1.1  # Inherited
        
        logger.info("✅ Scenario 4 passed: Complex multi-speaker and default change worked correctly")
    
    def test_new_speaker_with_default_fallback(self, config_manager, temp_config_dir):
        """
        Test that new speakers inherit from the configured default speaker, not first speaker.
        """
        # Create default config with multiple speakers, narrator as default
        default_speakers = [
            {
                "id": "david",
                "reference_audio": "david.wav",
                "tts_params": {
                    "exaggeration": 0.5,
                    "cfg_weight": 0.2,
                    "temperature": 0.9
                }
            },
            {
                "id": "narrator",
                "reference_audio": "narrator.wav",
                "tts_params": {
                    "exaggeration": 0.65,
                    "cfg_weight": 0.3,
                    "temperature": 1.0
                }
            }
        ]
        
        self.create_default_config(temp_config_dir, default_speakers, "narrator")  # narrator is default
        
        # Create job config that adds a new speaker
        job_config = {
            "generation": {
                "speakers": [
                    {
                        "id": "new_speaker",
                        "reference_audio": "new_speaker.wav"
                        # Missing tts_params - should inherit from narrator (default), not david (first)
                    }
                ]
            }
        }
        
        self.create_job_config(temp_config_dir, "test_job.yaml", job_config)
        
        # Load and merge configs
        merged_config = config_manager.load_cascading_config(temp_config_dir / "test_job.yaml")
        
        # Verify results
        speakers = merged_config["generation"]["speakers"]
        assert len(speakers) == 3  # All speakers present
        
        # Check new speaker
        new_speaker = next(s for s in speakers if s["id"] == "new_speaker")
        assert new_speaker["reference_audio"] == "new_speaker.wav"
        
        # Should inherit tts_params from narrator (default speaker), not david (first speaker)
        assert new_speaker["tts_params"]["exaggeration"] == 0.65  # From narrator
        assert new_speaker["tts_params"]["cfg_weight"] == 0.3     # From narrator
        assert new_speaker["tts_params"]["temperature"] == 1.0    # From narrator
        
        logger.info("✅ New speaker inheritance test passed: inherited from default speaker, not first speaker")
    
    def test_invalid_default_speaker_fallback(self, config_manager, temp_config_dir):
        """
        Test that invalid default_speaker falls back to first speaker gracefully.
        """
        # Create default config with invalid default_speaker
        default_speakers = [
            {
                "id": "david",
                "reference_audio": "david.wav",
                "tts_params": {
                    "exaggeration": 0.5,
                    "cfg_weight": 0.2,
                    "temperature": 0.9
                }
            }
        ]
        
        self.create_default_config(temp_config_dir, default_speakers, "nonexistent")  # Invalid default
        
        # Load and merge configs
        merged_config = config_manager.load_cascading_config(temp_config_dir / "default_config.yaml")
        
        # Should automatically fix the invalid default_speaker
        assert merged_config["generation"]["default_speaker"] == "david"  # Fallback to first speaker
        
        logger.info("✅ Invalid default_speaker fallback test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 