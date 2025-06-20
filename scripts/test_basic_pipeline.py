#!/usr/bin/env python3
"""
Basic pipeline test script
Tests the pipeline with simple configuration
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Now import pipeline modules
from pipeline.batch_executor import BatchExecutor
from utils.config_manager import ConfigManager
from utils.file_manager import AudioCandidate

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("🧪 Starting Basic Pipeline Test")
    
    try:
        # Initialize config manager
        config_manager = ConfigManager(project_root)
        
        # Load default config
        default_config_path = project_root / "config" / "default_config.yaml"
        config = config_manager.load_cascading_config(default_config_path)
        
        print(f"✅ Loaded config from: {default_config_path}")
        
        # Create batch executor
        executor = BatchExecutor()
        
        # Create task config and save it to get proper config_path
        task_config = config_manager.create_task_config(config)
        saved_config_path = config_manager.save_task_config(task_config, config)
        
        # Execute single task
        results = executor.execute_batch([task_config])
        
        # Check results
        if results and results[0].success:
            print("✅ Pipeline test completed successfully!")
            print(f"📁 Final audio: {results[0].final_audio_path}")
        else:
            print("❌ Pipeline test failed!")
            if results:
                print(f"Error: {results[0].error_message}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
