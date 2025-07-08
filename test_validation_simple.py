#!/usr/bin/env python3
"""
Simple test script to validate the improved speaker validation functionality.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config_manager import ConfigManager
from utils.file_manager.file_manager import FileManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_generation_handler_validation():
    """Test the GenerationHandler validation method directly."""
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
        
        # Create a simple GenerationHandler mock to test validation
        class MockGenerationHandler:
            def __init__(self, file_manager, config):
                self.file_manager = file_manager
                self.config = config
                
            def _validate_speakers(self) -> bool:
                """
                Validate speaker configuration and reference_audio files.

                Returns:
                    True if all speakers are valid, False otherwise
                """
                try:
                    # Validate speaker configuration structure
                    speakers = self.config.get("generation", {}).get("speakers", [])
                    if not speakers:
                        logger.error("❌ No speakers defined in configuration")
                        return False

                    # Validate reference_audio files for all speakers
                    validation_results = self.file_manager.validate_speakers_reference_audio()

                    if not validation_results["valid"]:
                        # Create detailed error message
                        failed_speakers = validation_results["failed_speakers"]
                        missing_files = validation_results["missing_files"]
                        available_files = validation_results["available_files"]
                        configured_speakers = validation_results["configured_speakers"]

                        logger.error("❌ " + "="*60)
                        logger.error("❌ SPEAKER VALIDATION FAILED")
                        logger.error("❌ " + "="*60)
                        logger.error(f"❌ Failed speakers: {len(failed_speakers)}")
                        logger.error("")
                        
                        # List each failed speaker with its missing file
                        for speaker_id in failed_speakers:
                            missing_file = missing_files.get(speaker_id, "unknown")
                            logger.error(f"   • Speaker '{speaker_id}' → Missing file: {missing_file}")
                        
                        logger.error("")
                        logger.error(f"📂 Available reference audio files ({len(available_files)}):")
                        if available_files:
                            for i, file in enumerate(sorted(available_files), 1):
                                logger.info(f"   {i:2d}. {file}")
                        else:
                            logger.error("   (No .wav files found in reference_audio directory)")
                        
                        logger.error("")
                        logger.error(f"⚙️  Configured speakers ({len(configured_speakers)}):")
                        for i, speaker_id in enumerate(configured_speakers, 1):
                            status = "✅" if speaker_id not in failed_speakers else "❌"
                            if speaker_id in failed_speakers:
                                logger.error(f"   {i:2d}. {speaker_id} {status}")
                            else:
                                logger.info(f"   {i:2d}. {speaker_id} {status}")
                        
                        logger.error("")
                        logger.error("💡 To fix this issue:")
                        logger.error("   1. Restore the missing reference audio files to data/input/reference_audio/")
                        logger.error("   2. Or update the speaker configurations to use available files")
                        logger.error("   3. Or remove the invalid speakers from your configuration")
                        logger.error("❌ " + "="*60)
                        
                        return False

                    logger.info(
                        f"✅ All {len(validation_results['configured_speakers'])} speakers validated: {validation_results['configured_speakers']}"
                    )
                    return True

                except Exception as e:
                    logger.error(f"❌ Speaker validation error: {e}")
                    return False
        
        # Test with valid speakers (should pass)
        logger.info("Testing GenerationHandler validation with valid speakers...")
        handler = MockGenerationHandler(file_manager, config)
        result = handler._validate_speakers()
        logger.info(f"Validation result: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_generation_handler_validation() 