#!/usr/bin/env python3
"""
Test script to validate candidate generation logic according to user specifications.

Test cases:
- 1 candidate + conservative enabled = 1 conservative
- 1 candidate + conservative disabled = 1 expressive (exact config)
- 2 candidates + conservative enabled = 1 expressive + 1 conservative
- 2 candidates + conservative disabled = 1 expressive exact + 1 expressive varied
- 5 candidates + conservative enabled = 4 expressive (1 exact + 3 varied) + 1 conservative
- 5 candidates + conservative disabled = 5 expressive (1 exact + 4 varied)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from generation.tts_generator import TTSGenerator
import yaml

def test_candidate_generation():
    """Test candidate generation logic with different configurations."""
    
    # Load base config
    with open('config/default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override for testing (no actual TTS model)
    tts_generator = TTSGenerator(config, device="cpu")
    
    test_text = "This is a test sentence."
    
    print("=== CANDIDATE GENERATION LOGIC TEST ===\n")
    
    # Test Case 1: 1 candidate + conservative enabled
    print("TEST 1: 1 candidate + conservative enabled = 1 conservative")
    config['generation']['conservative_candidate']['enabled'] = True
    test_candidates_logic(tts_generator, test_text, num_candidates=1, conservative_enabled=True)
    
    # Test Case 2: 1 candidate + conservative disabled
    print("\nTEST 2: 1 candidate + conservative disabled = 1 expressive (exact config)")
    test_candidates_logic(tts_generator, test_text, num_candidates=1, conservative_enabled=False)
    
    # Test Case 3: 2 candidates + conservative enabled
    print("\nTEST 3: 2 candidates + conservative enabled = 1 expressive + 1 conservative")
    test_candidates_logic(tts_generator, test_text, num_candidates=2, conservative_enabled=True)
    
    # Test Case 4: 2 candidates + conservative disabled
    print("\nTEST 4: 2 candidates + conservative disabled = 1 expressive exact + 1 expressive varied")
    test_candidates_logic(tts_generator, test_text, num_candidates=2, conservative_enabled=False)
    
    # Test Case 5: 5 candidates + conservative enabled
    print("\nTEST 5: 5 candidates + conservative enabled = 4 expressive (1 exact + 3 varied) + 1 conservative")
    test_candidates_logic(tts_generator, test_text, num_candidates=5, conservative_enabled=True)
    
    # Test Case 6: 5 candidates + conservative disabled
    print("\nTEST 6: 5 candidates + conservative disabled = 5 expressive (1 exact + 4 varied)")
    test_candidates_logic(tts_generator, test_text, num_candidates=5, conservative_enabled=False)

def test_candidates_logic(tts_generator, text, num_candidates, conservative_enabled):
    """Test candidate generation and analyze the parameters."""
    
    # Mock the actual generation to avoid model loading
    original_generate_single = tts_generator.generate_single
    def mock_generate_single(text, exaggeration=0.6, cfg_weight=0.7, temperature=1.0, **kwargs):
        import torch
        # Return mock audio tensor
        return torch.zeros(1000)
    
    tts_generator.generate_single = mock_generate_single
    
    # Configure conservative candidate
    conservative_config = {
        'enabled': conservative_enabled,
        'exaggeration': 0.45,
        'cfg_weight': 0.4,
        'temperature': 0.7
    } if conservative_enabled else {'enabled': False}
    
    try:
        # Generate candidates
        candidates = tts_generator.generate_candidates(
            text=text,
            num_candidates=num_candidates,
            exaggeration=0.40,  # MAX value
            cfg_weight=0.30,    # MIN value
            temperature=0.8,    # MIN value
            conservative_config=conservative_config
        )
        
        print(f"Generated {len(candidates)} candidates:")
        
        # Analyze parameters
        for i, candidate in enumerate(candidates):
            params = candidate.generation_params
            candidate_type = params.get('type', 'UNKNOWN')
            exag = params.get('exaggeration', 0)
            cfg = params.get('cfg_weight', 0)
            temp = params.get('temperature', 0)
            
            print(f"  Candidate {i+1} ({candidate_type}): exag={exag:.2f}, cfg={cfg:.2f}, temp={temp:.2f}")
            
        # Validate expectations
        validate_expectations(candidates, num_candidates, conservative_enabled)
        
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        # Restore original method
        tts_generator.generate_single = original_generate_single

def validate_expectations(candidates, num_candidates, conservative_enabled):
    """Validate that candidates match expected behavior."""
    
    expected_conservative = 1 if conservative_enabled else 0
    expected_expressive = num_candidates - expected_conservative
    
    conservative_candidates = [c for c in candidates if c.generation_params.get('type') == 'CONSERVATIVE']
    expressive_candidates = [c for c in candidates if c.generation_params.get('type') == 'EXPRESSIVE']
    
    print(f"  Expected: {expected_expressive} expressive + {expected_conservative} conservative")
    print(f"  Actual: {len(expressive_candidates)} expressive + {len(conservative_candidates)} conservative")
    
    # Validate conservative candidate logic
    if conservative_enabled:
        if num_candidates == 1:
            # Special case: only conservative
            if len(conservative_candidates) == 1 and len(expressive_candidates) == 0:
                print("  ✅ CORRECT: Single conservative candidate")
            else:
                print("  ❌ ERROR: Expected single conservative candidate")
        else:
            # Conservative should be last candidate
            if len(conservative_candidates) == 1 and conservative_candidates[0].candidate_idx == num_candidates - 1:
                print("  ✅ CORRECT: Conservative candidate is last")
            else:
                print("  ❌ ERROR: Conservative candidate should be last")
    
    # Validate expressive candidate parameters
    if expressive_candidates:
        first_expressive = expressive_candidates[0]
        params = first_expressive.generation_params
        
        # First expressive should have exact config values
        if abs(params.get('exaggeration', 0) - 0.40) < 0.001:
            print("  ✅ CORRECT: First expressive has exact exaggeration config")
        else:
            print(f"  ❌ ERROR: First expressive exaggeration {params.get('exaggeration', 0):.3f} != 0.400")
        
        if abs(params.get('cfg_weight', 0) - 0.30) < 0.001:
            print("  ✅ CORRECT: First expressive has exact cfg_weight config")
        else:
            print(f"  ❌ ERROR: First expressive cfg_weight {params.get('cfg_weight', 0):.3f} != 0.300")
            
        if abs(params.get('temperature', 0) - 0.8) < 0.001:
            print("  ✅ CORRECT: First expressive has exact temperature config")
        else:
            print(f"  ❌ ERROR: First expressive temperature {params.get('temperature', 0):.3f} != 0.800")
    
    # Validate ramping direction for multi-expressive
    if len(expressive_candidates) > 1:
        # Check that exaggeration ramps DOWN
        exag_values = [c.generation_params.get('exaggeration', 0) for c in expressive_candidates]
        if all(exag_values[i] >= exag_values[i+1] for i in range(len(exag_values)-1)):
            print("  ✅ CORRECT: Exaggeration ramps DOWN")
        else:
            print(f"  ❌ ERROR: Exaggeration should ramp DOWN: {exag_values}")
        
        # Check that cfg_weight ramps UP
        cfg_values = [c.generation_params.get('cfg_weight', 0) for c in expressive_candidates]
        if all(cfg_values[i] <= cfg_values[i+1] for i in range(len(cfg_values)-1)):
            print("  ✅ CORRECT: CFG weight ramps UP")
        else:
            print(f"  ❌ ERROR: CFG weight should ramp UP: {cfg_values}")
        
        # Check that temperature ramps UP
        temp_values = [c.generation_params.get('temperature', 0) for c in expressive_candidates]
        if all(temp_values[i] <= temp_values[i+1] for i in range(len(temp_values)-1)):
            print("  ✅ CORRECT: Temperature ramps UP")
        else:
            print(f"  ❌ ERROR: Temperature should ramp UP: {temp_values}")

if __name__ == "__main__":
    test_candidate_generation() 