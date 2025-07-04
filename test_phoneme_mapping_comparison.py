#!/usr/bin/env python3
"""
Language Tag Segmentation Test
=============================

Testet die verbesserte Segmentierung, die Satzzeichen und Leerzeichen erhält.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup minimal logging
logging.basicConfig(level=logging.WARNING)

def test_segmentation_based_processing():
    """Testet die verbesserte segmentierungs-basierte Verarbeitung."""
    
    print("🧩 Language Tag Segmentation Test")
    print("=" * 70)
    
    # Import dependencies
    try:
        from src.preprocessor.language_tag_processor import LanguageTagProcessor
        from src.preprocessor.text_preprocessor import TextPreprocessor
        print("✅ Dependencies loaded")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Initialize processor
    print("🔧 Initializing language tag processor...")
    try:
        config = {
            "process_language_tags": True,
            "language_tag_processor": {
                "phoneme_mappings_path": "config/phoneme_mappings_minimal.yaml"
            }
        }
        
        processor = LanguageTagProcessor(config["language_tag_processor"])
        text_preprocessor = TextPreprocessor(config)
        print("✅ Processors initialized")
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return
    
    # Test cases for segmentation
    test_cases = [
        # Basic punctuation preservation
        ('[lang="de"]Hallo, Welt![/lang]', 'Basic punctuation'),
        
        # Complex punctuation
        ('[lang="de"]"Hallo," sagte er... "na ja."[/lang]', 'Complex punctuation'),
        
        # Compound words with hyphens
        ('[lang="de"]Das ist ein Bind-Wort-Test.[/lang]', 'Compound words'),
        
        # Contractions
        ('[lang="de"]Wie geht\'s? Es geht\'s gut![/lang]', 'Contractions'),
        
        # Multiple sentences
        ('[lang="de"]Erste Satz. Zweiter Satz! Dritter Satz?[/lang]', 'Multiple sentences'),
        
        # Line breaks and paragraphs
        ('[lang="de"]Erste Zeile.\nZweite Zeile.\n\nNeuer Absatz.[/lang]', 'Line breaks'),
        
        # Mixed content with numbers and symbols
        ('[lang="de"]Test @123 & Symbole... #hashtag![/lang]', 'Numbers and symbols'),
        
        # French with accents
        ('[lang="fr"]C\'est magnifique, n\'est-ce pas?[/lang]', 'French with accents'),
        
        # Mixed languages
        ('Hello [lang="de"]und auf Deutsch,[/lang] back to [lang="fr"]français![/lang]', 'Mixed languages'),
        
        # Complex real-world example
        ('[lang="de"]München ist schön.\n\nWie geht\'s? "Gut," sagt er.\nBind-Wörter... ja![/lang]', 'Complex real-world')
    ]
    
    print("\n🔬 Testing segmentation-based processing...")
    print("=" * 70)
    
    for i, (test_text, description) in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {description}")
        print(f"   Input:  '{test_text}'")
        
        try:
            # Process through improved language tag processor
            result = processor.process_text(test_text)
            print(f"   Output: '{result}'")
            
            # Analyze preservation
            analysis = analyze_preservation(test_text, result)
            print(f"   📊 Analysis: {analysis}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test segmentation function directly
    print(f"\n\n🔧 Testing segmentation function directly...")
    print("=" * 70)
    
    test_segmentation_function(processor)
    
    # Full integration test
    print(f"\n\n🔄 Full integration test...")
    print("=" * 70)
    
    complex_text = """
    This is normal text. [lang="de"]Aber hier: "Hallo, wie geht's?" Das ist toll![/lang]
    
    Back to English. [lang="fr"]Et voici: C'est magnifique, n'est-ce pas?[/lang]
    
    Final text with punctuation.
    """
    
    try:
        result = text_preprocessor._process_text_content(complex_text)
        print(f"Complex test input:\n{complex_text}")
        print(f"Complex test output:\n{result}")
        
        print(f"\n💡 Integration analysis:")
        print(f"   - Punctuation outside tags: {'preserved' if '.' in result else 'lost'}")
        print(f"   - Line breaks: {'preserved' if '\\n' in result else 'lost'}")
        print(f"   - Mixed content: {'working' if 'This is normal text' in result else 'broken'}")
        
    except Exception as e:
        print(f"❌ Integration test error: {e}")

def test_segmentation_function(processor):
    """Test the segmentation function directly."""
    
    test_texts = [
        "Hallo, Welt!",
        "Bind-Wort test",
        "Wie geht's?",
        "Text mit\nZeilenumbruch",
        "Text... mit... Ellipsen!",
        '"Anführungszeichen" und (Klammern)',
        "123 Zahlen & Symbole @#$"
    ]
    
    for text in test_texts:
        print(f"\n   Segmenting: '{text}'")
        try:
            segments = processor.segment_text(text)
            print(f"   Segments:")
            for j, segment in enumerate(segments):
                print(f"     {j+1}. {segment['type']:8} → '{segment['text']}'")
        except Exception as e:
            print(f"   ❌ Segmentation error: {e}")

def analyze_preservation(original: str, processed: str) -> str:
    """Analyze what was preserved in processing."""
    
    # Extract content from language tags for comparison
    import re
    tag_pattern = r'\[lang="[^"]+"\](.*?)\[/lang\]'
    
    original_tagged_content = ""
    for match in re.finditer(tag_pattern, original, re.DOTALL):
        original_tagged_content += match.group(1)
    
    if not original_tagged_content:
        return "No language tags found"
    
    # Count various elements
    orig_spaces = original_tagged_content.count(' ')
    proc_spaces = processed.count(' ')
    
    orig_punct = len([c for c in original_tagged_content if not c.isalnum() and not c.isspace()])
    proc_punct = len([c for c in processed if not c.isalnum() and not c.isspace()])
    
    orig_lines = original_tagged_content.count('\n')
    proc_lines = processed.count('\n')
    
    # Generate analysis
    results = []
    if proc_spaces >= orig_spaces * 0.8:  # Allow some tolerance
        results.append("✅ Spaces preserved")
    else:
        results.append(f"⚠️ Spaces: {orig_spaces}→{proc_spaces}")
    
    if proc_punct >= orig_punct * 0.8:  # Allow some tolerance
        results.append("✅ Punctuation preserved")
    else:
        results.append(f"⚠️ Punctuation: {orig_punct}→{proc_punct}")
    
    if proc_lines >= orig_lines:
        results.append("✅ Line breaks preserved")
    else:
        results.append(f"⚠️ Line breaks: {orig_lines}→{proc_lines}")
    
    return " | ".join(results)

if __name__ == "__main__":
    test_segmentation_based_processing() 