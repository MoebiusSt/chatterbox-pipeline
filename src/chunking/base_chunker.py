from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


# Data model for a text chunk, as defined in the development plan.
@dataclass
class TextChunk:
    """Represents a single chunk of text for processing."""

    text: str
    start_pos: int
    end_pos: int
    has_paragraph_break: bool
    # This is an estimate, not a precise count.
    estimated_tokens: int
    # Flag to indicate this chunk was created by fallback splitting of a very long sentence
    is_fallback_split: bool = False
    # Index for ordering chunks (set by pipeline)
    idx: int = 0

    # Speaker-System Extensions
    speaker_id: str = "default"                # Current speaker
    speaker_transition: bool = False           # Chunk starts with speaker transition
    original_markup: Optional[str] = None      # Original markup for debugging


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""

    @abstractmethod
    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Splits a given text into a list of TextChunk objects.

        Returns:
            A list of TextChunk objects.
        """
        pass
