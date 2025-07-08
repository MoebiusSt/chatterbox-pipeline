#!/usr/bin/env python3
"""
ChunkIOHandler for text chunk operations.
Handles saving and loading of text chunks.
"""

import json
import logging
from pathlib import Path
from typing import List

from chunking.base_chunker import TextChunk

logger = logging.getLogger(__name__)


class ChunkIOHandler:
    """Handles text chunk I/O operations."""

    def __init__(self, texts_dir: Path):
        """
        Initialize ChunkIOHandler.

        Args:
            texts_dir: Directory for text chunk files
        """
        self.texts_dir = texts_dir
        self.texts_dir.mkdir(parents=True, exist_ok=True)

    def save_chunks(self, chunks: List[TextChunk]) -> bool:
        """
        Save text chunks to files.

        Args:
            chunks: List of TextChunk objects

        Returns:
            True if successful
        """
        try:
            # Save individual chunk files
            for chunk in chunks:
                chunk_filename = f"chunk_{chunk.idx+1:03d}.txt"
                chunk_path = self.texts_dir / chunk_filename

                with open(chunk_path, "w", encoding="utf-8") as f:
                    f.write(chunk.text)

            # Save chunk metadata
            chunk_metadata = {
                "total_chunks": len(chunks),
                "chunks": [
                    {
                        "idx": chunk.idx,
                        "text_length": len(chunk.text),
                        "is_paragraph_break": chunk.has_paragraph_break,
                        "filename": f"chunk_{chunk.idx+1:03d}.txt",
                        # Speaker-System Metadaten
                        "speaker_id": chunk.speaker_id,
                        "speaker_transition": chunk.speaker_transition,
                        "original_markup": chunk.original_markup,
                    }
                    for chunk in chunks
                ],
            }

            metadata_path = self.texts_dir / "chunks_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(chunk_metadata, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved {len(chunks)} chunks to {self.texts_dir}")
            return True

        except Exception as e:
            logger.error(f"Error saving chunks: {e}")
            return False

    def get_chunks(self) -> List[TextChunk]:
        """
        Load text chunks from files.

        Returns:
            List of TextChunk objects
        """
        chunks = []

        # Load metadata if available
        metadata_path = self.texts_dir / "chunks_metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        # Load chunk files
        chunk_files = sorted(self.texts_dir.glob("chunk_*.txt"))
        for chunk_file in chunk_files:
            # Extract chunk index from filename (convert from 1-based filename to 0-based idx)
            chunk_idx = int(chunk_file.stem.split("_")[1]) - 1

            # Load text content
            with open(chunk_file, "r", encoding="utf-8") as f:
                text = f.read()

            # Get metadata for this chunk (metadata uses 0-based idx)
            chunk_meta = None
            if metadata and "chunks" in metadata:
                chunk_meta = next(
                    (c for c in metadata["chunks"] if c["idx"] == chunk_idx), None
                )

            is_paragraph_break = (
                chunk_meta.get("is_paragraph_break", False) if chunk_meta else False
            )

            # Speaker-System Metadaten
            speaker_id = (
                chunk_meta.get("speaker_id", "default") if chunk_meta else "default"
            )
            speaker_transition = (
                chunk_meta.get("speaker_transition", False) if chunk_meta else False
            )
            original_markup = chunk_meta.get("original_markup") if chunk_meta else None

            chunk = TextChunk(
                idx=chunk_idx,
                text=text,
                start_pos=0,  # Not tracked in saved chunks
                end_pos=len(text),  # Not tracked in saved chunks
                has_paragraph_break=is_paragraph_break,
                estimated_tokens=len(text.split()),
                is_fallback_split=False,
                speaker_id=speaker_id,
                speaker_transition=speaker_transition,
                original_markup=original_markup,
            )
            chunks.append(chunk)

        # Sort by index
        chunks.sort(key=lambda c: c.idx)

        logger.debug(f"Loaded {len(chunks)} chunks from {self.texts_dir}")
        return chunks
