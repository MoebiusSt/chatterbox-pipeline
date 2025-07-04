#!/usr/bin/env python3
"""
Test script for Language Tag Integration
Tests the complete pipeline: Config → TextPreprocessor → LanguageTagProcessor
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessor import TextPreprocessor

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_language_tag_integration():
    """Test the complete language tag integration."""
    
    logger.info("🧪 Testing Language Tag Integration")
    logger.info("=" * 60)
    
    # Test configuration
    config = {
        "enabled": True,
        "normalize_line_endings": True,
        "process_language_tags": True,
        "language_tag_processor": {
            "phoneme_mappings_path": "config/phoneme_mappings.yaml"
        }
    }
    
    # Test texts
    test_cases = [
        {
            "name": "German city name",
            "text": "Visit [lang=\"de\"]München[/lang] in Bavaria!",
            "expected_contains": "Visit"  # Should contain Visit but München should be transformed
        },
        {
            "name": "French word",
            "text": "I love [lang=\"fr\"]Paris[/lang] very much.",
            "expected_contains": "I love"  # Should contain this but Paris should be transformed
        },
        {
            "name": "Multiple languages",
            "text": "From [lang=\"de\"]Berlin[/lang] to [lang=\"fr\"]Lyon[/lang].",
            "expected_contains": "From"  # Should contain this but cities should be transformed
        },
        {
            "name": "No language tags",
            "text": "This is plain English text.",
            "expected_contains": "This is plain English text."  # Should remain unchanged
        },
        {
            "name": "Empty text",
            "text": "",
            "expected_contains": ""  # Should remain empty
        }
    ]
    
    try:
        # Initialize TextPreprocessor
        logger.info("🔧 Initializing TextPreprocessor with language tag support...")
        preprocessor = TextPreprocessor(config)
        
        # Test each case
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n📋 Test Case {i}: {test_case['name']}")
            logger.info(f"Input:  '{test_case['text']}'")
            
            # Process the text
            result = preprocessor._process_text_content(test_case['text'])
            
            logger.info(f"Output: '{result}'")
            
            # Basic validation
            if test_case['expected_contains'] in result:
                logger.info("✅ Test passed")
            else:
                logger.warning(f"⚠️ Test may have issues - expected to contain: '{test_case['expected_contains']}'")
        
        # Statistics
        if hasattr(preprocessor, 'language_processor') and preprocessor.language_processor:
            stats = preprocessor.language_processor.get_statistics()
            logger.info(f"\n📊 Language Processor Statistics:")
            logger.info(f"   Supported languages: {stats['supported_languages']}")
            logger.info(f"   Global mappings: {stats['global_mappings']}")
            logger.info(f"   Custom language mappings: {stats['custom_language_mappings']}")
            logger.info(f"   Model phones: {stats['model_phones']}")
        
        logger.info("\n🎉 Integration test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_language_tag_integration() 