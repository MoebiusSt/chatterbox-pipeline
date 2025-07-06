#!/usr/bin/env python3
"""
Tests for speaker-specific cascading configuration merging.
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
    """Test cases for speaker-specific cascading configuration merging."""
    
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
    
    def create_default_config(self, config_dir: Path, speakers_config: list):
        """Helper to create default_config.yaml with specified speakers."""
        default_config = {
            "job": {"name": "default"},
            "input": {"text_file": "test.txt"},
            "chunking": {"target_chunk_limit": 180},
            "generation": {
                "num_candidates": 3,
                "max_retries": 1,
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
                "id": "default",
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
        
        self.create_default_config(temp_config_dir, default_speakers)
        
        # Create job config that only overrides reference_audio
        job_config = {
            "generation": {
                "speakers": [
                    {
                        "id": "default",
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
        assert len(speakers) == 1
        
        speaker = speakers[0]
        assert speaker["id"] == "default"
        assert speaker["reference_audio"] == "cori_samuel_1.wav"  # Overridden
        
        # tts_params should be inherited from default
        assert "tts_params" in speaker
        assert speaker["tts_params"]["exaggeration"] == 0.55
        assert speaker["tts_params"]["cfg_weight"] == 0.2
        assert speaker["tts_params"]["temperature"] == 0.9
        assert speaker["tts_params"]["repetition_penalty"] == 1.3
        
        # conservative_candidate should be inherited from default
        assert "conservative_candidate" in speaker
        assert speaker["conservative_candidate"]["enabled"] is True
        assert speaker["conservative_candidate"]["exaggeration"] == 0.4
        
        logger.info("✅ Scenario 1 passed: reference_audio overridden, tts_params inherited")
    
    def test_scenario_2_default_speaker_with_different_id(self, config_manager, temp_config_dir):
        """
        Scenario 2: Default config has 'mike' as default speaker, job config has 'default'.
        """
        # Create default config with 'mike' as first speaker
        default_speakers = [
            {
                "id": "mike",
                "reference_audio": "mike.wav",
                "tts_params": {
                    "exaggeration": 0.6,
                    "cfg_weight": 0.3,
                    "temperature": 0.8
                }
            }
        ]
        
        self.create_default_config(temp_config_dir, default_speakers)
        
        # Create job config that uses 'default' as ID
        job_config = {
            "generation": {
                "speakers": [
                    {
                        "id": "default",
                        "reference_audio": "cori_samuel_1.wav"
                    }
                ]
            }
        }
        
        self.create_job_config(temp_config_dir, "test_job.yaml", job_config)
        
        # Load and merge configs
        merged_config = config_manager.load_cascading_config(temp_config_dir / "test_job.yaml")
        
        # Verify results
        speakers = merged_config["generation"]["speakers"]
        assert len(speakers) == 1
        
        speaker = speakers[0]
        assert speaker["id"] == "default"
        assert speaker["reference_audio"] == "cori_samuel_1.wav"
        
        # Should inherit tts_params from 'mike' (the default speaker)
        assert "tts_params" in speaker
        assert speaker["tts_params"]["exaggeration"] == 0.6
        assert speaker["tts_params"]["cfg_weight"] == 0.3
        assert speaker["tts_params"]["temperature"] == 0.8
        
        logger.info("✅ Scenario 2 passed: 'default' ID merged with 'mike' speaker")
    
    def test_scenario_3_named_speaker_override(self, config_manager, temp_config_dir):
        """
        Scenario 3: Job config has 'narrator' as speaker 0, default config has 'narrator' as speaker 1.
        """
        # Create default config with multiple speakers
        default_speakers = [
            {
                "id": "mike",
                "reference_audio": "mike.wav",
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
        
        self.create_default_config(temp_config_dir, default_speakers)
        
        # Create job config that puts 'narrator' as first speaker
        job_config = {
            "generation": {
                "speakers": [
                    {
                        "id": "narrator",
                        "reference_audio": "cori_samuel_1.wav"
                    }
                ]
            }
        }
        
        self.create_job_config(temp_config_dir, "test_job.yaml", job_config)
        
        # Load and merge configs
        merged_config = config_manager.load_cascading_config(temp_config_dir / "test_job.yaml")
        
        # Verify results
        speakers = merged_config["generation"]["speakers"]
        # Should have both speakers: narrator (overridden) + mike (remaining)
        assert len(speakers) == 2
        
        # First speaker should be narrator with overridden reference_audio
        narrator_speaker = speakers[0]
        assert narrator_speaker["id"] == "narrator"
        assert narrator_speaker["reference_audio"] == "cori_samuel_1.wav"
        
        # Should inherit tts_params from original narrator config
        assert "tts_params" in narrator_speaker
        assert narrator_speaker["tts_params"]["exaggeration"] == 0.65
        assert narrator_speaker["tts_params"]["cfg_weight"] == 0.3
        assert narrator_speaker["tts_params"]["temperature"] == 1.0
        
        # Second speaker should be mike (unchanged)
        mike_speaker = speakers[1]
        assert mike_speaker["id"] == "mike"
        assert mike_speaker["reference_audio"] == "mike.wav"
        
        logger.info("✅ Scenario 3 passed: 'narrator' found by ID and merged correctly")
    
    def test_scenario_4_multiple_speakers_complex_merge(self, config_manager, temp_config_dir):
        """
        Scenario 4: Complex merging with multiple speakers, reordering, and partial overrides.
        """
        # Create default config with multiple speakers
        default_speakers = [
            {
                "id": "mike",
                "reference_audio": "mike.wav",
                "tts_params": {
                    "exaggeration": 0.5,
                    "cfg_weight": 0.2,
                    "temperature": 0.9
                }
            },
            {
                "id": "cory",
                "reference_audio": "cory.wav",
                "tts_params": {
                    "exaggeration": 0.6,
                    "cfg_weight": 0.3,
                    "temperature": 1.0
                }
            },
            {
                "id": "tom",
                "reference_audio": "tom.wav",
                "tts_params": {
                    "exaggeration": 0.7,
                    "cfg_weight": 0.4,
                    "temperature": 1.1
                }
            },
            {
                "id": "andre",
                "reference_audio": "andre.wav",
                "tts_params": {
                    "exaggeration": 0.8,
                    "cfg_weight": 0.5,
                    "temperature": 1.2
                }
            }
        ]
        
        self.create_default_config(temp_config_dir, default_speakers)
        
        # Create job config that reorders speakers and overrides some
        job_config = {
            "generation": {
                "speakers": [
                    {
                        "id": "tom",
                        "reference_audio": "tom_new.wav"
                    },
                    {
                        "id": "cory",
                        "reference_audio": "cory_new.wav"
                    }
                ]
            }
        }
        
        self.create_job_config(temp_config_dir, "test_job.yaml", job_config)
        
        # Load and merge configs
        merged_config = config_manager.load_cascading_config(temp_config_dir / "test_job.yaml")
        
        # Verify results
        speakers = merged_config["generation"]["speakers"]
        # Should have 4 speakers: tom (first, overridden), cory (overridden), mike (remaining), andre (remaining)
        assert len(speakers) == 4
        
        # Check speaker order and content
        speaker_ids = [s["id"] for s in speakers]
        assert speaker_ids == ["tom", "cory", "mike", "andre"]
        
        # tom should be first with overridden reference_audio
        tom_speaker = speakers[0]
        assert tom_speaker["id"] == "tom"
        assert tom_speaker["reference_audio"] == "tom_new.wav"
        assert tom_speaker["tts_params"]["exaggeration"] == 0.7  # Inherited from default
        
        # cory should be second with overridden reference_audio
        cory_speaker = speakers[1]
        assert cory_speaker["id"] == "cory"
        assert cory_speaker["reference_audio"] == "cory_new.wav"
        assert cory_speaker["tts_params"]["exaggeration"] == 0.6  # Inherited from default
        
        # mike should be third, unchanged
        mike_speaker = speakers[2]
        assert mike_speaker["id"] == "mike"
        assert mike_speaker["reference_audio"] == "mike.wav"
        
        # andre should be fourth, unchanged
        andre_speaker = speakers[3]
        assert andre_speaker["id"] == "andre"
        assert andre_speaker["reference_audio"] == "andre.wav"
        
        logger.info("✅ Scenario 4 passed: Complex multi-speaker merging worked correctly")
    
    def test_partial_tts_params_merge(self, config_manager, temp_config_dir):
        """
        Test that partial tts_params overrides work correctly.
        """
        # Create default config with full tts_params
        default_speakers = [
            {
                "id": "default",
                "reference_audio": "default.wav",
                "tts_params": {
                    "exaggeration": 0.55,
                    "cfg_weight": 0.2,
                    "temperature": 0.9,
                    "repetition_penalty": 1.3
                }
            }
        ]
        
        self.create_default_config(temp_config_dir, default_speakers)
        
        # Create job config that only overrides some tts_params
        job_config = {
            "generation": {
                "speakers": [
                    {
                        "id": "default",
                        "reference_audio": "new.wav",
                        "tts_params": {
                            "exaggeration": 0.75,  # Override
                            "temperature": 1.1      # Override
                            # cfg_weight and repetition_penalty should be inherited
                        }
                    }
                ]
            }
        }
        
        self.create_job_config(temp_config_dir, "test_job.yaml", job_config)
        
        # Load and merge configs
        merged_config = config_manager.load_cascading_config(temp_config_dir / "test_job.yaml")
        
        # Verify results
        speakers = merged_config["generation"]["speakers"]
        speaker = speakers[0]
        
        assert speaker["reference_audio"] == "new.wav"
        
        # Check tts_params merging
        tts_params = speaker["tts_params"]
        assert tts_params["exaggeration"] == 0.75        # Overridden
        assert tts_params["temperature"] == 1.1          # Overridden
        assert tts_params["cfg_weight"] == 0.2           # Inherited
        assert tts_params["repetition_penalty"] == 1.3   # Inherited
        
        logger.info("✅ Partial tts_params merge test passed")
    
    def test_new_speaker_addition(self, config_manager, temp_config_dir):
        """
        Test that new speakers are added properly with defaults.
        """
        # Create default config with one speaker
        default_speakers = [
            {
                "id": "default",
                "reference_audio": "default.wav",
                "tts_params": {
                    "exaggeration": 0.55,
                    "cfg_weight": 0.2,
                    "temperature": 0.9
                }
            }
        ]
        
        self.create_default_config(temp_config_dir, default_speakers)
        
        # Create job config that adds a new speaker
        job_config = {
            "generation": {
                "speakers": [
                    {
                        "id": "default",
                        "reference_audio": "new_default.wav"
                    },
                    {
                        "id": "new_speaker",
                        "reference_audio": "new_speaker.wav"
                        # Missing tts_params - should be filled from default
                    }
                ]
            }
        }
        
        self.create_job_config(temp_config_dir, "test_job.yaml", job_config)
        
        # Load and merge configs
        merged_config = config_manager.load_cascading_config(temp_config_dir / "test_job.yaml")
        
        # Verify results
        speakers = merged_config["generation"]["speakers"]
        assert len(speakers) == 2
        
        # Check new speaker
        new_speaker = speakers[1]
        assert new_speaker["id"] == "new_speaker"
        assert new_speaker["reference_audio"] == "new_speaker.wav"
        
        # Should inherit tts_params from default speaker
        assert "tts_params" in new_speaker
        assert new_speaker["tts_params"]["exaggeration"] == 0.55
        assert new_speaker["tts_params"]["cfg_weight"] == 0.2
        assert new_speaker["tts_params"]["temperature"] == 0.9
        
        logger.info("✅ New speaker addition test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 