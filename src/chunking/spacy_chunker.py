import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import spacy
from spacy.tokens import Doc, Span

from .base_chunker import BaseChunker, TextChunk

# Use the centralized logging configuration from main.py
logger = logging.getLogger(__name__)


class SpaCyChunker(BaseChunker):
    """
    A text chunker that uses SpaCy for linguistic sentence segmentation.
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
                subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
                self.nlp = spacy.load(model_name)
            except (subprocess.CalledProcessError, ImportError) as e:
                logger.error(f"Failed to download spacy model '{model_name}': {e}")
                raise RuntimeError(f"SpaCy model '{model_name}' not available and download failed")
        logger.info(f"SpaCy Chunker initialized with model '{model_name}'.")

    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Chunks the text using SpaCy's sentence segmentation. It aims to create
        chunks that are close to the target limit without exceeding the max limit,
        while respecting sentence boundaries and paragraph breaks.
        """
        if not text or not text.strip():
            return []

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
                        f"Attempting fallback splitting..."
                    )

                    # Try to split at secondary delimiters to avoid breaking Whisper context window
                    split_chunks = self._fallback_split_long_sentence(
                        sent, self.max_limit
                    )

                    if len(split_chunks) == 2:
                        logger.info(
                            f"✅ Successfully split long sentence into 2 parts using fallback delimiters"
                        )

                        # Add the first split chunk immediately as a complete chunk
                        first_part = split_chunks[0].lstrip()  # Only strip leading whitespace
                        chunks.append(
                            TextChunk(
                                text=first_part,
                                start_pos=sent.start_char,  # Approximate
                                end_pos=sent.start_char
                                + len(first_part),  # Approximate
                                has_paragraph_break=self._ends_with_paragraph_break(first_part),
                                estimated_tokens=self._estimate_token_length(
                                    first_part
                                ),
                                is_fallback_split=True,
                            )
                        )

                        # Add the second split chunk immediately as a complete chunk too
                        second_part = split_chunks[1].lstrip()  # Only strip leading whitespace
                        if second_part.strip():  # Check if chunk has content
                            chunks.append(
                                TextChunk(
                                    text=second_part,
                                    start_pos=sent.start_char
                                    + len(first_part),  # Approximate
                                    end_pos=sent.end_char,  # Approximate
                                    has_paragraph_break=self._ends_with_paragraph_break(second_part),
                                    estimated_tokens=self._estimate_token_length(
                                        second_part
                                    ),
                                    is_fallback_split=True,
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
                        has_paragraph_break=self._ends_with_paragraph_break(final_chunk_text),
                        estimated_tokens=self._estimate_token_length(final_chunk_text),
                        is_fallback_split=False,  # Regular chunks are not fallback splits
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
        text_for_check = text.rstrip(' \t\r')
        
        # Check if text ends with double newline (paragraph break)
        # This indicates that after this chunk, a paragraph pause should be inserted
        return text_for_check.endswith('\n\n')

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
                    and len(first_part.strip()) > 0  # Check content without affecting whitespace
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
            second_part = text[best_split_pos:].lstrip()  # Only strip leading whitespace
            logger.info(
                f"✅ Split using '{best_delimiter}' near middle: {len(first_part)} + {len(second_part)} chars"
            )
            return [first_part, second_part]

        # If no good splits found, return original text as single chunk
        logger.warning(
            f"❌ No suitable split point found near middle with secondary delimiters"
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
