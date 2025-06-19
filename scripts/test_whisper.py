import torch
import whisper


def detect_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


device = detect_device()
print(f"Using device: {device}")
print()

# Test if models load on detected device
models = ["base", "small", "medium"]
for model_name in models:
    try:
        print(f"Testing {model_name} on {device}...")
        model = whisper.load_model(model_name, device=device)
        print(f"✅ {model_name} loaded successfully on {device}")
    except Exception as e:
        print(f"❌ {model_name} failed on {device}: {e}")
    print()
