import logging
import sys
from pathlib import Path

import yaml

# Add src to the Python path to allow for absolute imports
# This is a common practice for development scripts
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from chunking.chunk_validator import ChunkValidator
from chunking.spacy_chunker import SpaCyChunker

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """
    Main function to run the chunking process for testing.
    """
    project_root = Path(__file__).resolve().parents[1]
    logging.info(f"Project root detected at: {project_root}")

    # --- 1. Load Configuration ---
    config_path = project_root / "config" / "default_config.yaml"
    logging.info(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    chunking_config = config.get("chunking", {})
    logging.info(f"Chunking config: {chunking_config}")

    # --- 2. Read Input Text ---
    input_text_path = project_root / "data" / "input" / "texts" / "input-document.txt"
    logging.info(f"Loading input text from: {input_text_path}")
    with open(input_text_path, "r", encoding="utf-8") as f:
        text_input = f.read()

    # --- 3. Instantiate and Run SpaCyChunker ---
    logging.info("Initializing SpaCyChunker...")
    chunker = SpaCyChunker(
        model_name=chunking_config.get("spacy_model", "en_core_web_sm"),
        target_limit=chunking_config.get("target_chunk_limit", 500),
        max_limit=chunking_config.get("max_chunk_limit", 600),
        min_length=chunking_config.get("min_chunk_length", 200),
    )

    logging.info("Running chunker...")
    chunks = chunker.chunk_text(text_input)

    if not chunks:
        logging.error("Chunking produced no output. Exiting.")
        return

    # --- 4. Instantiate and Run ChunkValidator ---
    logging.info("Initializing ChunkValidator...")
    validator = ChunkValidator(
        max_limit=chunking_config.get("max_chunk_limit", 600),
        min_length=chunking_config.get("min_chunk_length", 200),
    )

    logging.info("Running validator on generated chunks...")
    is_valid = validator.run_all_validations(chunks)

    logging.info(f"Validation result: {'PASS' if is_valid else 'FAIL'}")

    # --- 5. Print Chunks ---
    print("\n" + "=" * 50)
    print(f"Generated {len(chunks)} Chunks")
    print("=" * 50)
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1}/{len(chunks)} ---")
        print(
            f"Length: {len(chunk.text)}, Para Break: {chunk.has_paragraph_break}, Tokens (est): {chunk.estimated_tokens}"
        )
        print(f"Start/End Pos: {chunk.start_pos}/{chunk.end_pos}")
        print("-" * 20)
        print(chunk.text)
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
