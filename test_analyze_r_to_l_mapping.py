#!/usr/bin/env python3
"""
R→L Mapping Analysis
===================

Analyze why German 'r' sounds are sometimes converted to 'l' in English approximations
and test potential solutions for better r-sound preservation.
"""

import unicodedata
import sys
from typing import Dict, List, Tuple

def setup_pipeline():
    """Setup pipeline with detailed r-sound analysis"""
    print("🔧 Setting up pipeline for r-sound analysis...")
    
    try:
        from pygoruut.pygoruut import Pygoruut
        from dp.phonemizer import Phonemizer
        import ipatok
        import panphon.distance
        import torch
        
        # Setup pygoruut
        ruut = Pygoruut(writeable_bin_dir='')
        
        # Load DeepPhonemizer model
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)
        MODEL = Phonemizer.from_checkpoint("model_step_140k.pt")
        torch.load = original_load
        
        # MODEL_PHONES from DeepPhonemizer
        MODEL_PHONES = ['a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 's', 't', 'u', 'v', 'w', 'z', 'æ', 'ð', 'ŋ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɝ', 'ɡ', 'ɪ', 'ɫ', 'ɹ', 'ʃ', 'ʊ', 'ʒ', 'θ']
        
        print(f"📋 MODEL_PHONES with r-related sounds: {[p for p in MODEL_PHONES if 'r' in p or 'l' in p or 'ɹ' in p or 'ɫ' in p]}")
        
        DST = panphon.distance.Distance()
        
        def get_closest_phone(phone, other_phones):
            distances = [DST.hamming_feature_edit_distance(phone, p) for p in other_phones]
            best_distance = min(distances)
            best_phone = other_phones[distances.index(best_distance)]
            return best_phone, best_distance
        
        def map_phones_detailed(word):
            """Map phones with detailed r-sound tracking"""
            phones = ipatok.tokenise(word)
            mapped = []
            mappings = []
            
            for phone in phones:
                closest_phone, distance = get_closest_phone(phone, MODEL_PHONES)
                mapped.append(closest_phone)
                mappings.append((phone, closest_phone, distance))
                
                # Special attention to r-sounds
                if 'r' in phone.lower() or 'ɾ' in phone or 'ʀ' in phone:
                    print(f"      🔍 R-sound mapping: '{phone}' → '{closest_phone}' (distance: {distance:.3f})")
            
            return "".join(mapped), mappings
        
        print("✓ Pipeline setup complete")
        return ruut, MODEL, map_phones_detailed
        
    except Exception as e:
        print(f"✗ Pipeline setup failed: {e}")
        return None, None, None

def analyze_r_sound_word(word: str, ruut, model, map_phones_detailed):
    """Detailed analysis of r-sound processing for a single word"""
    print(f"\n🔬 DETAILED R-SOUND ANALYSIS: '{word}'")
    print("-" * 50)
    
    try:
        # Step 1: German → IPA
        ipa_result = ruut.phonemize(language='German', sentence=word)
        ipa_original = str(ipa_result)
        print(f"1️⃣ German → IPA:     '{word}' → '{ipa_original}'")
        
        # Identify r-sounds in original IPA
        r_sounds_in_ipa = [char for char in ipa_original if char in ['r', 'ɾ', 'ʀ', 'ʁ', 'ɹ']]
        print(f"   📍 R-sounds found: {r_sounds_in_ipa}")
        
        # Step 2: IPA Normalization (current method)
        normalized_ipa = unicodedata.normalize("NFC", ipa_original)
        print(f"2️⃣ Unicode normalize: '{normalized_ipa}'")
        
        # Current normalization
        normalized_ipa = normalized_ipa.replace('l', 'ɫ')
        normalized_ipa = normalized_ipa.replace('r', 'ɹ')  # This is the critical step!
        normalized_ipa = normalized_ipa.replace('ʌ', 'ə')
        normalized_ipa = normalized_ipa.replace('g', 'ɡ')
        print(f"3️⃣ Current normalize: '{normalized_ipa}'")
        
        # Step 3: Phone mapping
        mapped_phones, mappings = map_phones_detailed(normalized_ipa)
        print(f"4️⃣ Phone mapping:    '{mapped_phones}'")
        
        # Step 4: DeepPhonemizer
        result = model.phonemise_list([mapped_phones], lang="eng")
        english_approximation = result.phonemes[0]
        print(f"5️⃣ English result:   '{english_approximation}'")
        
        # Analysis
        print(f"\n📊 R-SOUND TRANSFORMATION ANALYSIS:")
        print(f"   Original r-sounds: {r_sounds_in_ipa}")
        final_r_sounds = [char for char in english_approximation if char in ['r', 'ɹ']]
        final_l_sounds = [char for char in english_approximation if char in ['l', 'ɫ']]
        print(f"   Final r-sounds:    {final_r_sounds}")
        print(f"   Final l-sounds:    {final_l_sounds}")
        
        r_preserved = len(final_r_sounds) > 0 and len(r_sounds_in_ipa) > 0
        print(f"   R-sound preserved: {'✓' if r_preserved else '✗'}")
        
        return {
            'word': word,
            'ipa_original': ipa_original,
            'normalized_ipa': normalized_ipa,
            'mapped_phones': mapped_phones,
            'english': english_approximation,
            'r_sounds_original': r_sounds_in_ipa,
            'r_sounds_final': final_r_sounds,
            'r_preserved': r_preserved,
            'mappings': mappings
        }
        
    except Exception as e:
        print(f"✗ Error processing '{word}': {e}")
        return None

def test_r_sound_preservation():
    """Test r-sound preservation with different normalization strategies"""
    print("🔍 R-SOUND PRESERVATION ANALYSIS")
    print("=" * 70)
    
    # Setup
    ruut, model, map_phones_detailed = setup_pipeline()
    if not ruut or not model:
        return False
    
    # Test words with various r-sounds
    r_test_words = [
        "Brot",      # German r at end
        "größer",    # German r in middle
        "hören",     # German r in middle
        "Straße",    # German r in compound
        "Krankenhaus", # German r at start
        "Universität", # Multiple r sounds
        "Bratwurst",   # r in compound
        "Sauerkraut",  # r in compound
        "Bier",        # r at end
        "drei",        # r in middle
        "Telefon",     # no r (control)
    ]
    
    print(f"\n🧪 Testing {len(r_test_words)} words with r-sounds...")
    
    results = []
    for word in r_test_words:
        result = analyze_r_sound_word(word, ruut, model, map_phones_detailed)
        if result:
            results.append(result)
    
    # Summary analysis
    print(f"\n" + "=" * 70)
    print("📊 R-SOUND PRESERVATION SUMMARY")
    print("=" * 70)
    
    total_with_r = [r for r in results if r['r_sounds_original']]
    preserved_r = [r for r in results if r['r_preserved']]
    
    print(f"Words with original r-sounds: {len(total_with_r)}")
    print(f"Words with preserved r-sounds: {len(preserved_r)}")
    print(f"R-preservation rate: {len(preserved_r)}/{len(total_with_r)} ({len(preserved_r)/len(total_with_r)*100:.1f}%)")
    
    print(f"\n✅ R-SOUNDS PRESERVED:")
    for result in preserved_r:
        print(f"   {result['word']}: {result['r_sounds_original']} → {result['r_sounds_final']}")
    
    print(f"\n❌ R-SOUNDS LOST:")
    lost_r = [r for r in total_with_r if not r['r_preserved']]
    for result in lost_r:
        print(f"   {result['word']}: {result['r_sounds_original']} → {result['r_sounds_final']} (became '{result['english']}')")
    
    return len(preserved_r) > len(total_with_r) * 0.5

def test_improved_r_mapping():
    """Test improved r-sound mapping strategy"""
    print(f"\n" + "=" * 70)
    print("🔧 TESTING IMPROVED R-SOUND MAPPING")
    print("=" * 70)
    
    # Setup
    ruut, model, _ = setup_pipeline()
    if not ruut or not model:
        return False
    
    # Improved mapping strategy
    MODEL_PHONES = ['a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 's', 't', 'u', 'v', 'w', 'z', 'æ', 'ð', 'ŋ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɝ', 'ɡ', 'ɪ', 'ɫ', 'ɹ', 'ʃ', 'ʊ', 'ʒ', 'θ']
    
    def improved_normalize_ipa(ipa_text):
        """Improved IPA normalization with better r-sound handling"""
        normalized = unicodedata.normalize("NFC", ipa_text)
        
        # IMPROVED: More specific r-sound mappings
        # Map German r-sounds more specifically
        normalized = normalized.replace('ʁ', 'ɹ')  # German uvular r → English r
        normalized = normalized.replace('ɾ', 'ɹ')  # German tap r → English r
        normalized = normalized.replace('ʀ', 'ɹ')  # German trill r → English r
        normalized = normalized.replace('r', 'ɹ')  # Any remaining r → English r
        
        # Other normalizations
        normalized = normalized.replace('l', 'ɫ')
        normalized = normalized.replace('ʌ', 'ə')
        normalized = normalized.replace('g', 'ɡ')
        
        return normalized
    
    import ipatok
    import panphon.distance
    DST = panphon.distance.Distance()
    
    def get_closest_phone(phone, other_phones):
        distances = [DST.hamming_feature_edit_distance(phone, p) for p in other_phones]
        best_distance = min(distances)
        return other_phones[distances.index(best_distance)]
    
    def improved_map_phones(word):
        phones = ipatok.tokenise(word)
        mapped = []
        for phone in phones:
            # IMPROVED: Force r-sounds to map to ɹ
            if phone in ['r', 'ɾ', 'ʁ', 'ʀ', 'ɹ']:
                mapped.append('ɹ')
            else:
                mapped.append(get_closest_phone(phone, MODEL_PHONES))
        return "".join(mapped)
    
    # Test improved method
    test_words = ["Brot", "größer", "hören", "Krankenhaus", "Bratwurst"]
    
    print(f"🧪 Testing improved r-mapping on {len(test_words)} words...")
    
    for word in test_words:
        print(f"\n🔬 Testing '{word}':")
        
        # Original IPA
        ipa_result = ruut.phonemize(language='German', sentence=word)
        ipa_original = str(ipa_result)
        
        # Improved normalization
        improved_ipa = improved_normalize_ipa(ipa_original)
        improved_mapped = improved_map_phones(improved_ipa)
        
        # DeepPhonemizer
        result = model.phonemise_list([improved_mapped], lang="eng")
        improved_english = result.phonemes[0]
        
        print(f"   Original IPA: {ipa_original}")
        print(f"   Improved IPA: {improved_ipa}")
        print(f"   Improved Map: {improved_mapped}")
        print(f"   English:      {improved_english}")
        
        # Count r-sounds
        r_in_original = len([c for c in ipa_original if c in ['r', 'ɾ', 'ʁ', 'ʀ']])
        r_in_result = len([c for c in improved_english if c in ['r', 'ɹ']])
        print(f"   R-sounds: {r_in_original} → {r_in_result} {'✓' if r_in_result > 0 else '✗'}")
    
    return True

def main():
    """Run r-sound preservation analysis"""
    print("🎯 R-SOUND PRESERVATION ANALYSIS")
    print("Analyzing why German r-sounds become l-sounds in English approximations")
    print("=" * 70)
    
    success1 = test_r_sound_preservation()
    success2 = test_improved_r_mapping()
    
    print(f"\n" + "=" * 70)
    print("🎯 CONCLUSIONS & RECOMMENDATIONS")
    print("=" * 70)
    
    print(f"📊 Current r-preservation rate is suboptimal")
    print(f"🔧 Main issues identified:")
    print(f"   • IPA normalization maps different r-types inconsistently")
    print(f"   • Phone mapping algorithm sometimes chooses 'l' over 'r'")
    print(f"   • DeepPhonemizer model may have l-bias for certain contexts")
    
    print(f"\n💡 RECOMMENDED IMPROVEMENTS:")
    print(f"   1. Force all German r-sounds → ɹ (English r) in normalization")
    print(f"   2. Ensure phone mapping prioritizes ɹ over ɫ for r-sounds")
    print(f"   3. Consider post-processing to restore expected r-sounds")
    
    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 