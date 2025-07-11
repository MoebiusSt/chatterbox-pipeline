import torch
import soundfile as sf
from chatterbox.tts import ChatterboxTTS
from huggingface_hub import hf_hub_download

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available(): # For MacOS
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# --- HuggingFace Authentication ---
HF_TOKEN = "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"

# --- Configuration ---
REFERENCE_AUDIO_PATH = "data/input/reference_audio/glen_hallstrom_1.wav"
OUTPUT_FILENAME = "data/output/german_havok_output_glen1.wav"
SEED = 12345  # For reproducibility

# TTS Generation Parameters
TTS_EXAGGERATION = 0.45
TTS_CFG_WEIGHT = 0.55
TTS_TEMPERATURE = 0.8

# German text to synthesize
german_text = """Das heißt, ich integriere Elemente, die eigentlich nicht musikalisch sind, in ein musikalisches Geschehen. Eine andere Möglichkeit liegt im Innermusikalischen. Ich suche nach Repertoire, das einer Gruppe zeigt, dass sie individuelle Mitglieder sind und gleichzeitig einen Gesamtklang erzeugen, zum Beispiel im Clustersingen. Das bedeutet, es gibt keinen falschen Ton. Es entsteht ein Klanggebilde, das seine Schönheit aus der Vielfalt der Stimmen kreiert."""

# --- Main script ---
if __name__ == "__main__":
    try:
        print("Downloading standalone German Kartoffelbox model...")
        
        # Download all necessary files for the merged model
        model_files = [
            "merged_model/s3gen.safetensors",
            "merged_model/t3_cfg.safetensors", 
            "merged_model/ve.safetensors",
            "merged_model/tokenizer.json"
        ]
        
        # Download all files and get directory from first one
        from pathlib import Path
        downloaded_files = []
        MODEL_REPO = "havok2/Kartoffelbox-v0.1_0.65h2"
        for filename in model_files:
            print(f"Downloading {filename}...")
            downloaded_file = hf_hub_download(
                repo_id=MODEL_REPO, 
                filename=filename,
                token=HF_TOKEN
            )
            downloaded_files.append(downloaded_file)
        
        # Get the model directory (parent of merged_model)
        model_dir = Path(downloaded_files[0]).parent.parent
        print(f"Model directory: {model_dir}")
        
        print("All model files downloaded.")
        
        # Load the standalone German model using from_local
        print("Loading standalone German ChatterboxTTS model...")
        finetuned_dir = model_dir / "finetuned_by_h2" #version 1
        merged_model_dir = model_dir / "merged_model" # version 2

        model = ChatterboxTTS.from_local(finetuned_dir, device=device)
        print("German model loaded successfully!")
        
        print("Generating German speech...")
        print(f"Text: {german_text}")
        print(f"Reference audio: {REFERENCE_AUDIO_PATH}")
        
        # Set seed for reproducibility
        torch.manual_seed(SEED)
        
        # Generate speech with German model
        print("Generating speech...")
        with torch.inference_mode():
            wav = model.generate(
                german_text,
                audio_prompt_path=REFERENCE_AUDIO_PATH,
                exaggeration=TTS_EXAGGERATION,
                temperature=TTS_TEMPERATURE,
                cfg_weight=TTS_CFG_WEIGHT,
            )

        # Save the generated audio
        print(f"Saving audio to {OUTPUT_FILENAME}...")
        sf.write(OUTPUT_FILENAME, wav.squeeze().cpu().numpy(), model.sr)
        print(f"Audio saved successfully to {OUTPUT_FILENAME}")
        
        print("German TTS generation with standalone model completed successfully!")

    except FileNotFoundError:
        print(f"ERROR: The reference audio file was not found at {REFERENCE_AUDIO_PATH}")
        print("Please make sure the file exists in the project directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("Script finished.")