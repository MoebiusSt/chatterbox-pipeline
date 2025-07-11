#!/usr/bin/env python3
"""
Test script to validate the improved speaker validation functionality.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config_manager import ConfigManager
from utils.file_manager.file_manager import FileManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_speaker_validation():
    """Test the improved speaker validation functionality."""
    try:
        # Initialize config manager
        config_manager = ConfigManager(Path.cwd())
        
        # Load default config
        config_path = Path.cwd() / "config" / "default_config.yaml"
        config = config_manager.load_cascading_config(config_path)
        
        # Create a test task config
        task_config = config_manager.create_task_config(config)
        
        # Create file manager
        file_manager = FileManager(task_config, config)
        
        # Test the improved validation
        logger.info("Testing improved speaker validation...")
        validation_results = file_manager.validate_speakers_reference_audio()
        
        logger.info(f"Validation results: {validation_results}")
        
        # Test the detailed error output
        if not validation_results["valid"]:
            logger.info("\nDetailed validation information:")
            logger.info(f"Failed speakers: {validation_results['failed_speakers']}")
            logger.info(f"Missing files: {validation_results['missing_files']}")
            logger.info(f"Available files: {validation_results['available_files']}")
            logger.info(f"Configured speakers: {validation_results['configured_speakers']}")
            
            # Test the generation handler validation
            logger.info("\nTesting GenerationHandler validation...")
            from pipeline.task_executor.stage_handlers.generation_handler import GenerationHandler
            from generation.tts_generator import TTSGenerator
            from generation.candidate_manager import CandidateManager
            
            # Create dummy instances for testing
            tts_generator = TTSGenerator(config, logger)
            candidate_manager = CandidateManager(config, file_manager, tts_generator)
            
            generation_handler = GenerationHandler(
                file_manager, config, tts_generator, candidate_manager
            )
            
            # Test the validation method
            is_valid = generation_handler._validate_speakers()
            logger.info(f"Generation handler validation result: {is_valid}")
        else:
            logger.info("All speakers are valid!")
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_speaker_validation() 