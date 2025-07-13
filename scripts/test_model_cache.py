#!/usr/bin/env python3
"""
Test script to demonstrate model caching functionality.
Shows cache behavior within a single program run.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from generation.model_cache import ChatterboxModelCache


def test_model_cache():
    """Test model caching functionality within a single session."""
    
    print("=" * 60)
    print("🧪 TESTING MODEL CACHE BEHAVIOR")
    print("=" * 60)
    
    # Show initial cache state
    print("\n1. INITIAL CACHE STATE")
    print("-" * 30)
    cache_info = ChatterboxModelCache.get_cache_info()
    print(f"Cache info: {cache_info}")
    
    # Test 1: First model load (should be cache miss)
    print("\n2. FIRST MODEL LOAD (CACHE MISS)")
    print("-" * 30)
    
    start_time = time.time()
    model = ChatterboxModelCache.get_model("cpu")  # Use CPU for faster testing
    load_time = time.time() - start_time
    
    print(f"Model loaded in {load_time:.2f} seconds")
    print(f"Model type: {type(model)}")
    
    # Show updated cache info
    cache_info = ChatterboxModelCache.get_cache_info()
    print(f"Updated cache info: {cache_info}")
    
    # Test 2: Second model load in same process (should be cache hit)
    print("\n3. SECOND MODEL LOAD IN SAME PROCESS (CACHE HIT)")
    print("-" * 30)
    
    start_time = time.time()
    model2 = ChatterboxModelCache.get_model("cpu")
    load_time = time.time() - start_time
    
    print(f"Model loaded in {load_time:.2f} seconds")
    print(f"Same model object: {model is model2}")
    
    # Test 3: Clear cache and reload
    print("\n4. CLEAR CACHE AND RELOAD")
    print("-" * 30)
    
    ChatterboxModelCache.clear_cache()
    print("Cache cleared")
    
    start_time = time.time()
    model3 = ChatterboxModelCache.get_model("cpu")
    load_time = time.time() - start_time
    
    print(f"Model loaded in {load_time:.2f} seconds")
    print(f"Same model object as first: {model is model3}")
    
    # Show final cache info
    cache_info = ChatterboxModelCache.get_cache_info()
    print(f"Final cache info: {cache_info}")
    
    print("\n" + "=" * 60)
    print("✅ MODEL CACHE TEST COMPLETED")
    print("=" * 60)


def demonstrate_cache_explanation():
    """Demonstrate the cache explanation feature."""
    
    print("\n" + "=" * 60)
    print("🧑‍🏫 CACHE EXPLANATION DEMO")
    print("=" * 60)
    
    # Load a model first to populate cache
    print("Loading model to populate cache...")
    model = ChatterboxModelCache.get_model("cpu")
    
    # Show explanation
    ChatterboxModelCache.explain_cache_behavior()


if __name__ == "__main__":
    try:
        test_model_cache()
        demonstrate_cache_explanation()
        
        print("\n" + "=" * 60)
        print("💡 NEXT STEPS:")
        print("=" * 60)
        print("1. Run this script again to see cache miss behavior")
        print("2. Use 'python src/cbpipe.py --explain-cache' for explanation")
        print("3. Use 'python src/cbpipe.py --mode all' to process multiple tasks in one session")
        print("4. Run 'python scripts/test_model_cache.py' to test cache again")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        
    except Exception as e:
        print(f"\n\nError during test: {e}")
        import traceback
        traceback.print_exc() 