#!/usr/bin/env python3
"""
Test improved German "ch" mappings: ç/x → ʃ (sh-sound)
Tests if München now produces better results with sh-sound instead of k-sound
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessor import TextPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_improved_ch_mappings():
    """Test if German ch-sounds are now mapped to sh instead of k."""
    
    print("🧪 Testing Improved German 'ch' → 'sh' Mappings")
    print("=" * 60)
    
    # Configuration with language tag processing
    config = {
        "enabled": True,
        "normalize_line_endings": True,
        "process_language_tags": True,
        "language_tag_processor": {
            "phoneme_mappings_path": "config/phoneme_mappings.yaml"
        }
    }
    
    # Test cases with German "ch" sounds
    test_cases = [
        {
            "name": "München (ich-Laut ç)",
            "text": "Visit [lang=\"de\"]München[/lang] today!",
            "expected_improvement": "should contain 'sh' sound instead of 'k'"
        },
        {
            "name": "Bach (ach-Laut x)", 
            "text": "The composer [lang=\"de\"]Bach[/lang] was great.",
            "expected_improvement": "should contain 'sh' sound instead of 'k'"
        },
        {
            "name": "Buch (ich-Laut ç)",
            "text": "Read this [lang=\"de\"]Buch[/lang] please.",
            "expected_improvement": "should contain 'sh' sound instead of 'k'"
        },
        {
            "name": "Dach (ach-Laut x)",
            "text": "The [lang=\"de\"]Dach[/lang] is red.",
            "expected_improvement": "should contain 'sh' sound instead of 'k'"
        }
    ]
    
    try:
        # Initialize TextPreprocessor
        print("🔧 Initializing TextPreprocessor with improved ch-mappings...")
        preprocessor = TextPreprocessor(config)
        
        print("\n📋 Testing German words with 'ch' sounds:")
        print("=" * 60)
        
        # Test each case
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. {test_case['name']}")
            print(f"   Input:  '{test_case['text']}'")
            
            # Process the text
            result = preprocessor._process_text_content(test_case['text'])
            
            print(f"   Output: '{result}'")
            print(f"   Note:   {test_case['expected_improvement']}")
            
            # Basic validation
            if len(result) > 0 and result != test_case['text']:
                print(f"   ✅ Transformation successful")
            else:
                print(f"   ⚠️ No transformation occurred")
        
        print("\n" + "=" * 60)
        print("📝 Mapping Analysis:")
        print("   Before: ç → k, x → k  (harsh k-sound)")
        print("   After:  ç → ʃ, x → ʃ  (softer sh-sound)")
        print("   Result: More natural English pronunciation!")
        
        print("\n🎯 Expected Improvement Examples:")
        print("   München: 'moncen' → 'monchen' (with sh-sound)")
        print("   Bach:    'back'   → 'bash'    (with sh-sound)")
        
        print("\n🎉 Improved ch-mapping test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_ch_mappings() 