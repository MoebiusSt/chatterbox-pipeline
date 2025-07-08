"""
Progress tracking utility for the TTS pipeline.
Provides ASCII progress bars, timing information, and ETA calculations.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Tracks progress for long-running processes with ASCII progress bar,
    timing information, and ETA calculations.
    """

    def __init__(
        self, total_items: int, description: str = "Processing", bar_width: int = 50
    ):
        """
        Initialize progress tracker.

        Args:
            total_items: Total number of items to process
            description: Description of the process
            bar_width: Width of the ASCII progress bar
        """
        self.total_items = total_items
        self.description = description
        self.bar_width = bar_width
        self.current_item = 0
        self.start_time = time.time()
        self.item_times: List[float] = []

    def update(self, current_item: Optional[int] = None, message: str = "") -> None:
        """
        Update progress tracker.

        Args:
            current_item: Current item number (if None, increments by 1)
            message: Additional message to display
        """
        if current_item is not None:
            self.current_item = current_item
        else:
            self.current_item += 1

        current_time = time.time()
        self.item_times.append(current_time)

        # Calculate progress
        progress = self.current_item / self.total_items
        elapsed_time = current_time - self.start_time

        # Calculate ETA
        if self.current_item > 0:
            avg_time_per_item = elapsed_time / self.current_item
            remaining_items = self.total_items - self.current_item
            eta_seconds = avg_time_per_item * remaining_items
            eta = datetime.now() + timedelta(seconds=eta_seconds)
        else:
            eta = None

        # Create ASCII progress bar
        filled_width = int(self.bar_width * progress)
        bar = "█" * filled_width + "░" * (self.bar_width - filled_width)

        # Format timing information
        elapsed_str = self._format_duration(elapsed_time)
        eta_str = eta.strftime("%H:%M:%S") if eta else "N/A"

        # Log progress information
        progress_msg = (
            f"{self.description} Progress: "
            f"[{bar}] {progress:.1%} ({self.current_item}/{self.total_items}) | "
            f"Elapsed: {elapsed_str}"
        )

        if eta:
            progress_msg += f" | ETA: {eta_str}"

        if message:
            progress_msg += f" | {message}"

        logger.info(progress_msg)

    def finish(self) -> None:
        """Mark the process as finished and log final statistics."""
        end_time = time.time()
        total_time = end_time - self.start_time
        total_time_str = self._format_duration(total_time)

        logger.info(
            f"{self.description} COMPLETED - {self.total_items} items in {total_time_str}"
        )

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {mins}m {secs:.1f}s"


class ValidationProgressTracker(ProgressTracker):
    """
    Specialized progress tracker for validation processes that need to display
    text excerpts during Whisper validation.
    """

    def __init__(
        self, total_items: int, description: str = "Validation", bar_width: int = 50
    ):
        super().__init__(total_items, description, bar_width)
        # Only log start for validation if there are multiple items
        if total_items > 1:
            logger.debug(
                f"Starting {description}: {total_items} candidates to validate"
            )

    def update_with_texts(
        self,
        chunk_idx: int,
        current_text: str = "",
        transcribed_text: str = "",
        success: bool = True,
    ) -> None:
        """Update progress with text comparison information."""
        # Update base progress
        self.update(
            chunk_idx, f"Text: {current_text[:50]}..." if current_text else "No text"
        )

        # Display text comparison with clear separation
        if current_text or transcribed_text:
            logger.debug("=" * 80)
            logger.debug("TEXT COMPARISON")
            logger.debug("=" * 80)

            if current_text:
                logger.debug("ORIGINAL:")
                logger.debug(f"{current_text}")

            if transcribed_text:
                logger.debug("WHISPER RESULT:")
                logger.debug(f"{transcribed_text}")

        if not success:
            logger.info("Validation failed")
