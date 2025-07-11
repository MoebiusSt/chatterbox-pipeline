import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import spacy
from spacy.tokens import Span

from .base_chunker import BaseChunker, TextChunk

# Use the centralized logging configuration from main.py
logger = logging.getLogger(__name__)


class SpeakerMarkupParser:
    """
    Parser for speaker markup in text.

    Supports markup syntax:
    - <speaker:id> switches to speaker with corresponding ID
    - <speaker:0> or <speaker:reset> returns to default speaker
    """

    SPEAKER_PATTERN = r"<speaker:([^>]+)>"

    def parse_speaker_transitions(self, text: str) -> List[Tuple[int, str]]:
        """
        Parse speaker transitions and return (position, speaker_id) tuples.

        Args:
            text: Text with speaker markup

        Returns:
            List of (position, speaker_id) tuples
        """
        transitions = []
        for match in re.finditer(self.SPEAKER_PATTERN, text):
            position = match.start()
            speaker_id = match.group(1).strip()
            transitions.append((position, speaker_id))
        return transitions

    def remove_markup(self, text: str) -> str:
        """
        Remove speaker markup tags from text.

        Args:
            text: Text with speaker markup

        Returns:
            Text without markup tags
        """
        return re.sub(self.SPEAKER_PATTERN, "", text)

    def validate_speaker_id(
        self, speaker_id: str, available_speakers: List[str], default_speaker_id: str
    ) -> str:
        """
        Validate and normalize speaker ID.

        Args:
            speaker_id: Speaker ID to validate
            available_speakers: List of available speaker IDs
            default_speaker_id: The actual default speaker ID to use

        Returns:
            Validated/normalized speaker ID
        """
        # Normalize special IDs (default speaker aliases)
        if speaker_id in ["0", "default", "reset"]:
            return available_speakers[0] if available_speakers else default_speaker_id

        # Check if speaker is available
        if speaker_id in available_speakers:
            return speaker_id

        logger.warning(
            f"Unknown speaker '{speaker_id}', falling back to default speaker"
        )
        return available_speakers[0] if available_speakers else default_speaker_id


class SpaCyChunker(BaseChunker):
    """
    A text chunker that uses SpaCy for linguistic sentence segmentation.
    Enhanced with speaker-aware chunking support.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        target_limit: int = 500,
        max_limit: int = 600,
        min_length: int = 200,
    ):
        self.target_limit = target_limit
        self.max_limit = max_limit
        self.min_length = min_length
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.info(f"Spacy model '{model_name}' not found. Downloading...")
            try:
                import subprocess
                import sys

                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", model_name]
                )
                self.nlp = spacy.load(model_name)
            except (subprocess.CalledProcessError, ImportError) as e:
                logger.error(f"Failed to download spacy model '{model_name}': {e}")
                raise RuntimeError(
                    f"SpaCy model '{model_name}' not available and download failed"
                )

        # Speaker system components
        self.speaker_parser = SpeakerMarkupParser()
        self.available_speakers: List[str] = []  # Set externally
        self.default_speaker_id: Optional[str] = None  # Must be set externally

        logger.info(
            f"SpaCy Chunker initialized with model '{model_name}' and speaker support."
        )

    def set_available_speakers(self, speakers: List[str]):
        """
        Set available speaker IDs for validation.

        Args:
            speakers: List of available speaker IDs
        """
        self.available_speakers = speakers
        logger.debug(f"Set available speakers: {speakers}")

    def set_default_speaker_id(self, default_speaker_id: str):
        """
        Set the default speaker ID to use when no speaker markup is found.

        Args:
            default_speaker_id: The default speaker ID
        """
        self.default_speaker_id = default_speaker_id
        logger.debug(f"Set default speaker ID: {default_speaker_id}")

    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Chunks the text using SpaCy's sentence segmentation with speaker-aware splitting.
        Speaker changes have highest chunking priority.
        """
        if not text or not text.strip():
            return []

        # Validate that default_speaker_id is set
        if self.default_speaker_id is None:
            raise RuntimeError(
                "default_speaker_id not set. Call set_default_speaker_id() before chunking."
            )

        # 1. Parse speaker transitions
        transitions = self.speaker_parser.parse_speaker_transitions(text)

        if transitions:
            logger.debug(f"Found {len(transitions)} speaker transitions")
            # Use speaker-aware chunking
            return self._chunk_text_with_speakers(text, transitions)
        else:
            # Use traditional chunking without speaker support
            return self._chunk_text_traditional(text)

    def _chunk_text_with_speakers(
        self, text: str, transitions: List[Tuple[int, str]]
    ) -> List[TextChunk]:
        """
        Chunking with speaker support - speaker changes have highest priority.

        Args:
            text: Complete text with markup
            transitions: List of (position, speaker_id) tuples

        Returns:
            List of TextChunk objects with speaker information
        """
        # 1. Create primary splits at speaker changes
        clean_text = self.speaker_parser.remove_markup(text)
        primary_splits = self._create_speaker_splits(text, clean_text, transitions)

        # 2. Apply normal chunking logic to each speaker section
        all_chunks = []
        for speaker_section in primary_splits:
            section_chunks = self._chunk_speaker_section(speaker_section)
            all_chunks.extend(section_chunks)

        # 3. Post-processing and indexing
        return self._finalize_chunks(all_chunks)

    def _create_speaker_splits(
        self, original_text: str, clean_text: str, transitions: List[Tuple[int, str]]
    ) -> List[dict]:
        """
        Create primary division into speaker sections.

        Args:
            original_text: Text with markup (for extracting speaker IDs)
            clean_text: Text without markup (for actual splitting)
            transitions: List of (position, speaker_id) tuples from original text

        Returns:
            List of speaker sections
        """
        sections = []
        # Since we validated in chunk_text that default_speaker_id is not None, we can assert here
        assert self.default_speaker_id is not None
        current_speaker = (
            self.available_speakers[0] if self.available_speakers else self.default_speaker_id
        )

        # Strategy: Parse the original text to create speaker sections,
        # then map each section to the corresponding clean text

        # Create pattern to find speaker tags and split points
        import re

        speaker_pattern = r"<speaker:([^>]+)>"

        # Split original text by speaker tags
        parts = re.split(speaker_pattern, original_text)

        i = 0
        while i < len(parts):
            text_part = parts[i]

            if i == 0:
                # First part (before any speaker tag)
                if text_part.strip():
                    sections.append(
                        {
                            "text": text_part.lstrip(),
                            "speaker_id": current_speaker,
                            "start_pos": 0,
                            "speaker_transition": False,
                            "original_markup": None,
                        }
                    )
            else:
                # We have a speaker ID (from the regex split)
                if i % 2 == 1:
                    # This is a speaker ID
                    new_speaker_id = text_part.strip()
                    validated_speaker = self.speaker_parser.validate_speaker_id(
                        new_speaker_id, self.available_speakers, self.default_speaker_id
                    )
                    current_speaker = validated_speaker
                else:
                    # This is text content after a speaker tag
                    if text_part.strip():
                        sections.append(
                            {
                                "text": text_part.lstrip(),
                                "speaker_id": current_speaker,
                                "start_pos": 0,  # Will be recalculated
                                "speaker_transition": True,
                                "original_markup": new_speaker_id if i > 1 else None,
                            }
                        )
            i += 1

        logger.debug(f"Created {len(sections)} speaker sections")
        return sections

    def _chunk_speaker_section(self, section: dict) -> List[TextChunk]:
        """
        Normal chunking logic for individual speaker section.

        Args:
            section: Speaker section with text, speaker_id, etc.

        Returns:
            List of TextChunk objects
        """
        # Use traditional chunking for the speaker section
        base_chunks = self._chunk_text_traditional(section["text"])

        # Enhance with speaker information
        enhanced_chunks = []
        for i, chunk in enumerate(base_chunks):
            enhanced_chunk = TextChunk(
                text=chunk.text,
                start_pos=chunk.start_pos + section["start_pos"],
                end_pos=chunk.end_pos + section["start_pos"],
                has_paragraph_break=chunk.has_paragraph_break,
                estimated_tokens=chunk.estimated_tokens,
                is_fallback_split=chunk.is_fallback_split,
                idx=chunk.idx,
                speaker_id=section["speaker_id"],
                speaker_transition=(i == 0 and section["speaker_transition"]),
                original_markup=section["original_markup"] if i == 0 else None,
            )
            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks

    def _finalize_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Post-processing and indexing of chunks.

        Args:
            chunks: List of TextChunk objects

        Returns:
            Finalized list of TextChunk objects
        """
        # Set correct indices
        for i, chunk in enumerate(chunks):
            chunk.idx = i

        logger.debug(f"Finalized {len(chunks)} chunks with speaker information")
        return chunks

    def _chunk_text_traditional(self, text: str) -> List[TextChunk]:
        """
        Original chunking logic without speaker support.

        Args:
            text: Text to chunk

        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []

        # Since we validated in chunk_text that default_speaker_id is not None, we can assert here
        assert self.default_speaker_id is not None

        # Text preprocessing (line ending normalization) is now handled by TextPreprocessor
        doc = self.nlp(text)
        sentences = list(doc.sents)

        chunks: List[TextChunk] = []
        current_sent_idx = 0

        while current_sent_idx < len(sentences):
            chunk_sents: List[Span] = []
            current_chunk_len = 0

            # Greedily add sentences to the chunk
            while current_sent_idx < len(sentences):
                sent = sentences[current_sent_idx]
                sent_len = len(sent.text_with_ws)

                # If adding the next sentence exceeds max_limit, and the chunk is not empty, break.
                if (
                    current_chunk_len > 0
                    and current_chunk_len + sent_len > self.max_limit
                ):
                    break

                # NEW: Handle extremely long sentences that exceed max_limit on their own
                if current_chunk_len == 0 and sent_len > self.max_limit:
                    logger.warning(
                        f"Very long sentence ({sent_len} chars) exceeds max_limit ({self.max_limit}). "
                        "Attempting fallback splitting..."
                    )

                    # Try to split at secondary delimiters to avoid breaking Whisper context window
                    split_chunks = self._fallback_split_long_sentence(
                        sent, self.max_limit
                    )

                    if len(split_chunks) == 2:
                        logger.info(
                            "✅ Successfully split long sentence into 2 parts using fallback delimiters"
                        )

                        # Add the first split chunk immediately as a complete chunk
                        first_part = split_chunks[
                            0
                        ].lstrip()  # Only strip leading whitespace
                        chunks.append(
                            TextChunk(
                                text=first_part,
                                start_pos=sent.start_char,  # Approximate
                                end_pos=sent.start_char
                                + len(first_part),  # Approximate
                                has_paragraph_break=self._ends_with_paragraph_break(
                                    first_part
                                ),
                                estimated_tokens=self._estimate_token_length(
                                    first_part
                                ),
                                is_fallback_split=True,
                                speaker_id=self.default_speaker_id,  # Use configured default speaker
                                speaker_transition=False,  # No speaker transition in traditional chunking
                                original_markup=None,  # No markup in traditional chunking
                            )
                        )

                        # Add the second split chunk immediately as a complete chunk too
                        second_part = split_chunks[
                            1
                        ].lstrip()  # Only strip leading whitespace
                        if second_part.strip():  # Check if chunk has content
                            chunks.append(
                                TextChunk(
                                    text=second_part,
                                    start_pos=sent.start_char
                                    + len(first_part),  # Approximate
                                    end_pos=sent.end_char,  # Approximate
                                    has_paragraph_break=self._ends_with_paragraph_break(
                                        second_part
                                    ),
                                    estimated_tokens=self._estimate_token_length(
                                        second_part
                                    ),
                                    is_fallback_split=True,
                                    speaker_id=self.default_speaker_id,  # Use configured default speaker
                                    speaker_transition=False,  # No speaker transition in traditional chunking
                                    original_markup=None,  # No markup in traditional chunking
                                )
                            )

                        # Move to next sentence - this sentence is completely processed
                        current_sent_idx += 1
                        continue
                    else:
                        logger.warning(
                            f"❌ Fallback splitting failed, creating oversized chunk anyway: '{sent.text[:100]}...'"
                        )
                        # Fall through to original behavior
                        chunk_sents.append(sent)
                        current_chunk_len += sent_len
                        current_sent_idx += 1
                        break  # Move to next chunk immediately

                chunk_sents.append(sent)
                current_chunk_len += sent_len
                current_sent_idx += 1

                # If the chunk is now over the target_limit, it's a good place to stop.
                if current_chunk_len >= self.target_limit:
                    break

            if not chunk_sents:
                # This should not be reached if there are sentences, but as a safeguard.
                break

            chunk_text = "".join([s.text_with_ws for s in chunk_sents])

            if chunk_text.strip():  # Check if chunk has content after stripping
                # Use character indices from the original doc for accuracy
                start_char = chunk_sents[0].start_char
                end_char = chunk_sents[-1].end_char

                # The text for the chunk is re-sliced from the original doc
                # We preserve whitespace to keep paragraph breaks for detection
                final_chunk_text = doc.text[start_char:end_char]

                # Only strip leading whitespace, preserve trailing for paragraph break detection
                final_chunk_text = final_chunk_text.lstrip()

                chunks.append(
                    TextChunk(
                        text=final_chunk_text,
                        start_pos=start_char,
                        end_pos=end_char,
                        has_paragraph_break=self._ends_with_paragraph_break(
                            final_chunk_text
                        ),
                        estimated_tokens=self._estimate_token_length(final_chunk_text),
                        is_fallback_split=False,  # Regular chunks are not fallback splits
                        speaker_id=self.default_speaker_id,  # Use configured default speaker
                        speaker_transition=False,  # No speaker transition in traditional chunking
                        original_markup=None,  # No markup in traditional chunking
                    )
                )

        return chunks

    def _estimate_token_length(self, text: str) -> int:
        """
        Estimates the number of tokens in a text string.
        A simple proxy for token count.
        """
        return len(text.split())

    def _ends_with_paragraph_break(self, text: str) -> bool:
        """
        Check if a text chunk ends with a paragraph break.

        This determines whether a longer pause should be inserted AFTER this chunk
        during audio assembly.

        Args:
            text: The text to check

        Returns:
            True if the chunk ends with a paragraph break (indicating a paragraph pause should follow)
        """
        if not text:
            return False

        # Remove trailing whitespace except newlines to check the actual end pattern
        # We want to preserve trailing newlines for paragraph break detection
        text_for_check = text.rstrip(" \t\r")

        # Check if text ends with double newline (paragraph break)
        # This indicates that after this chunk, a paragraph pause should be inserted
        return text_for_check.endswith("\n\n")

    def _find_optimal_split_point(self, sentences: List[Span]) -> int:
        """
        Finds the optimal split point within a list of sentences to form a chunk.
        Args:
            sentences: A list of SpaCy Span objects (sentences).

        Returns:
            The index of the sentence to split after.
        """
        return len(sentences)

    def _fallback_split_long_sentence(
        self, sentence: Span, max_limit: int
    ) -> List[str]:
        """
        Attempts to split a very long sentence ONCE at a good delimiter near the middle
        to avoid breaking Whisper's context window while minimally disrupting text flow.
        """
        text = sentence.text_with_ws.lstrip()  # Only strip leading whitespace
        text_length = len(text)
        ideal_split_point = text_length // 2  # Aim for middle

        # Define secondary delimiters in order of preference
        secondary_delimiters = [";", "—", "–", '"', '"', ":", ","]

        logger.debug(
            f"Attempting to split {text_length} char sentence near position {ideal_split_point}..."
        )

        best_split_pos = None
        best_delimiter = None
        best_distance = float("inf")

        # Find the delimiter closest to the middle that creates valid chunks
        for delimiter in secondary_delimiters:
            if delimiter not in text:
                continue

            # Find all positions of this delimiter
            for i, char in enumerate(text):
                if char != delimiter:
                    continue

                # Split position would be after the delimiter
                split_pos = i + 1

                # Check if this split creates two reasonable chunks
                first_part = text[:split_pos].lstrip()  # Only strip leading whitespace
                second_part = text[split_pos:].lstrip()  # Only strip leading whitespace

                # Both parts must be under max_limit and non-empty
                if (
                    len(first_part) <= max_limit
                    and len(second_part) <= max_limit
                    and len(first_part.strip())
                    > 0  # Check content without affecting whitespace
                    and len(second_part.strip()) > 0
                ):

                    # Calculate distance from ideal split point
                    distance = abs(split_pos - ideal_split_point)

                    # Prefer this split if it's closer to the middle
                    if distance < best_distance:
                        best_distance = distance
                        best_split_pos = split_pos
                        best_delimiter = delimiter

        # If we found a good split point, use it
        if best_split_pos is not None:
            first_part = text[:best_split_pos].lstrip()  # Only strip leading whitespace
            second_part = text[
                best_split_pos:
            ].lstrip()  # Only strip leading whitespace
            logger.info(
                f"✅ Split using '{best_delimiter}' near middle: {len(first_part)} + {len(second_part)} chars"
            )
            return [first_part, second_part]

        # If no good splits found, return original text as single chunk
        logger.warning(
            "❌ No suitable split point found near middle with secondary delimiters"
        )
        return [text]

    def save_chunks_to_disk(
        self, chunks: List[TextChunk], output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Save text chunks to individual text files for analysis and debugging.

        Returns:
            List of saved file paths
        """
        if output_dir is None:
            # Default to project data/output/chunks directory
            project_root = Path(__file__).resolve().parents[2]
            output_path = project_root / "data" / "output" / "chunks"
        else:
            output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        timestamp = datetime.now().strftime("%H%M%S")

        for i, chunk in enumerate(chunks):
            try:
                # Create filename with chunk index and timestamp
                filename = f"chunk_{i+1:03d}_{timestamp}.txt"
                filepath = output_path / filename

                # Create content with metadata
                content = f"=== CHUNK {i+1:03d} ===\n"
                content += f"Length: {len(chunk.text)} characters\n"
                content += f"Tokens: {chunk.estimated_tokens}\n"
                content += f"Start pos: {chunk.start_pos}\n"
                content += f"End pos: {chunk.end_pos}\n"
                content += f"Has paragraph break: {chunk.has_paragraph_break}\n"
                # Speaker-System Metadaten
                content += f"Speaker ID: {chunk.speaker_id}\n"
                content += f"Speaker transition: {chunk.speaker_transition}\n"
                if chunk.original_markup:
                    content += f"Original markup: {chunk.original_markup}\n"
                content += f"{'='*50}\n\n"
                content += chunk.text

                # Save to file
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

                saved_paths.append(str(filepath))
                logger.debug(f"Saved chunk {i+1} to: {filepath}")

            except Exception as e:
                logger.error(f"Failed to save chunk {i+1}: {e}")
                continue

        if saved_paths:
            logger.info(f"Saved {len(saved_paths)} chunks to: {output_path}")

        return saved_paths
