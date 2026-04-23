#!/usr/bin/env python3
"""
ConfigManager for cascading configuration loading.
Supports default-yaml + job-yaml merging and task-yaml creation.
"""

import copy
import logging
from collections import OrderedDict
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import yaml

from utils.language_registry import VIBEVOICE_MODEL_TYPES

logger = logging.getLogger(__name__)


# Custom YAML dumper is no longer needed - using standard YAML dump
# All config files will use consistent formatting without quotes for simple strings


# YAML reserved words that YAML parsers may coerce to non-strings if unquoted.
_YAML_RESERVED_WORDS: set[str] = {
    "no",
    "yes",
    "true",
    "false",
    "null",
    "on",
    "off",
    "y",
    "n",
    "~",
}


@dataclass
class TaskConfig:
    """Configuration for a task."""

    task_name: str
    run_label: str
    timestamp: str
    base_output_dir: Path
    config_path: Path
    job_name: str
    force_final_generation: bool = False  # Force regeneration of final audio
    preloaded_config: Optional[Dict[str, Any]] = None  # Avoid redundant config loading
    rerender_all: bool = (
        False  # Delete all candidates and re-render everything from scratch
    )

    def __post_init__(self) -> None:
        # Ensure Path objects
        self.base_output_dir = Path(self.base_output_dir)
        self.config_path = Path(self.config_path)


class ConfigManager:
    """
    Central configuration manager for cascading config loading.

    Handles:
    - default-yaml (default_config.yaml) as fallback
    - job-yaml (custom configs) for specific jobs
    - task-yaml (completed configs with timestamp in job directories)
    """

    def __init__(self, project_root: Path):
        """
        Initializes the ConfigManager.
        """
        self.project_root = project_root
        self.config_dir = project_root / "config"
        self.output_dir = project_root / "data" / "output"
        self.default_config_path = self.config_dir / "default_config.yaml"

        # Cache for loaded configs
        self._config_cache: Dict[str, Dict[str, Any]] = {}

        # Cache for default config key order (hierarchical)
        self._default_key_order: Optional[Dict[str, Any]] = None

    def _parse_yaml_key_order(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse YAML file to extract hierarchical key order structure.

        Returns:
            Nested dictionary containing the key order for each level
        """
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        key_order: Dict[str, Any] = {}
        current_path: List[str] = []

        for line in lines:
            # Skip comments and empty lines
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Calculate indentation level
            indent_level = (len(line) - len(line.lstrip())) // 2

            # Check if this line contains a key
            if ":" in stripped:
                key = stripped.split(":")[0].strip()

                # Adjust current_path based on indentation
                current_path = current_path[:indent_level]

                # Navigate to current position in key_order structure
                current_dict = key_order
                for path_key in current_path:
                    if path_key not in current_dict:
                        current_dict[path_key] = {}
                    current_dict = current_dict[path_key]

                # Add this key to current level if not exists
                if "_order" not in current_dict:
                    current_dict["_order"] = []
                if key not in current_dict["_order"]:
                    current_dict["_order"].append(key)

                # Prepare for nested keys
                if key not in current_dict:
                    current_dict[key] = {}

                # Update current path for potential nested keys
                current_path.append(key)

        return key_order

    def _get_default_key_order(self) -> Dict[str, Any]:
        """Get the hierarchical key order from default_config.yaml to maintain consistent ordering."""
        if self._default_key_order is None:
            self._default_key_order = self._parse_yaml_key_order(
                self.default_config_path
            )
            logger.debug("Extracted hierarchical default key order")

        return self._default_key_order

    # ------------------------------
    # Identifier Sanitization
    # ------------------------------
    def _coerce_to_string_identifier(self, value: Any) -> str:
        """
        Coerce YAML-parsed value to a safe string for identifier fields.

        - Booleans/None → canonical lower-case strings ("true"/"false"/"null")
        - Other non-strings → str(value)
        - Strings returned as-is
        """
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        return str(value)

    def _sanitize_identifiers_post_load(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize YAML-reserved words that were parsed as non-strings where string IDs are required.

        This only touches string-typed identifier fields and leaves real boolean flags intact.
        Fields adjusted:
          - job.name
          - job.run_label
          - generation.default_speaker
          - generation.speakers[].id
          - generation.speakers[].language
        """
        if not isinstance(config, dict):
            return config

        cfg = copy.deepcopy(config)

        # job identifiers
        job = cfg.get("job")
        if isinstance(job, dict):
            if "name" in job:
                original = job["name"]
                job["name"] = self._coerce_to_string_identifier(job["name"])
                if original != job["name"]:
                    logger.debug(f"Identifier normalization: job.name '{original}' → '{job['name']}'")
            # Backward compatibility: migrate legacy key "run-label" → "run_label"
            if "run-label" in job and "run_label" not in job:
                job["run_label"] = job.pop("run-label")
                logger.debug("Backward compat: migrated job.run-label → job.run_label")
            elif "run-label" in job:
                # Both keys present: run_label wins, discard run-label
                job.pop("run-label")
                logger.debug("Backward compat: discarded legacy job.run-label (run_label already set)")

            if "run_label" in job:
                original = job["run_label"]
                # Gracefully handle None/False/True for run_label → empty string
                val = original
                if val is None:
                    job["run_label"] = ""
                elif isinstance(val, bool):
                    job["run_label"] = ""  # booleans are never valid labels
                else:
                    job["run_label"] = self._coerce_to_string_identifier(val)
                if original != job["run_label"]:
                    logger.debug(f"Identifier normalization: job.run_label '{original}' → '{job['run_label']}'")

        # generation identifiers
        gen = cfg.get("generation")
        if isinstance(gen, dict):
            if "default_speaker" in gen:
                original = gen["default_speaker"]
                gen["default_speaker"] = self._coerce_to_string_identifier(gen["default_speaker"])
                if original != gen["default_speaker"]:
                    logger.debug(
                        f"Identifier normalization: generation.default_speaker '{original}' → '{gen['default_speaker']}'"
                    )

            speakers = gen.get("speakers")
            if isinstance(speakers, list):
                for spk in speakers:
                    if not isinstance(spk, dict):
                        continue
                    if "id" in spk:
                        original = spk["id"]
                        spk["id"] = self._coerce_to_string_identifier(spk["id"])
                        if original != spk["id"]:
                            logger.debug(f"Identifier normalization: speaker.id '{original}' → '{spk['id']}'")
                    if "language" in spk:
                        original_lang = spk["language"]
                        new_lang: str = ""
                        # Graceful normalization for language field
                        if original_lang is None:
                            new_lang = ""  # will fall back later
                        elif isinstance(original_lang, bool):
                            # Unquoted YAML 'no' becomes False → map to 'no'; True is invalid → fallback
                            new_lang = "no" if original_lang is False else ""
                        else:
                            str_lang = str(original_lang).strip()
                            low = str_lang.lower()
                            if low == "no":
                                new_lang = "no"
                            elif self._is_valid_language_code(str_lang):
                                new_lang = low
                            elif low == "false" and str(spk.get("id", "")).lower() in {"norwegian", "no", "nb", "nn"}:
                                # Heuristic: previous runs may have persisted 'false' when user intended 'no'
                                new_lang = "no"
                            else:
                                # Invalid/ambiguous → empty to trigger fallback
                                new_lang = ""
                        spk["language"] = new_lang
                        if original_lang != spk["language"]:
                            logger.debug(
                                f"Identifier normalization: speaker.language '{original_lang}' → '{spk['language']}'"
                            )

        return cfg

    def _sort_dict_hierarchically(
        self, data: Dict[str, Any], key_order_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recursively sort dictionary according to hierarchical key order structure.

        Args:
            data: The data dictionary to sort
            key_order_structure: The hierarchical key order structure from default config

        Returns:
            Sorted OrderedDict
        """
        if not isinstance(data, dict):
            return data

        sorted_dict = OrderedDict()

        # Get the key order for this level
        level_order = key_order_structure.get("_order", [])

        # First, add keys in the specified order
        for key in level_order:
            if key in data:
                value = data[key]
                if isinstance(value, dict) and key in key_order_structure:
                    # Recursively sort nested dictionaries
                    sorted_dict[key] = self._sort_dict_hierarchically(
                        value, key_order_structure[key]
                    )
                else:
                    sorted_dict[key] = value

        # Then add any remaining keys that weren't in the default order (edge case)
        remaining_keys = set(data.keys()) - set(level_order)
        for key in sorted(remaining_keys):
            value = data[key]
            if isinstance(value, dict):
                # Try to find nested structure or use empty structure for unknown keys
                nested_structure = key_order_structure.get(key, {})
                sorted_dict[key] = self._sort_dict_hierarchically(
                    value, nested_structure
                )
            else:
                sorted_dict[key] = value

        return sorted_dict

    def _convert_ordered_dict_to_dict(self, data: Any) -> Any:
        """
        Recursively convert OrderedDict to regular dict for proper YAML serialization.
        """
        if isinstance(data, OrderedDict):
            return {
                key: self._convert_ordered_dict_to_dict(value)
                for key, value in data.items()
            }
        elif isinstance(data, dict):
            return {
                key: self._convert_ordered_dict_to_dict(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._convert_ordered_dict_to_dict(item) for item in data]
        else:
            return data

    def _is_valid_language_code(self, value: Any) -> bool:
        """
        Validate a language code for speakers.

        Rules:
        - Must be a string
        - Accept explicit 'no' (Norwegian) even though it's a YAML reserved token when unquoted
        - Must not be other YAML-reserved words (true, false, null, yes, on, off, ...)
        - Must match a simple language pattern: 2-3 letters optionally followed by '-' and 2-3 letters
          Examples: 'de', 'en', 'fr', 'nl', 'no', 'pt-br', 'zh-cn'
        """
        if not isinstance(value, str):
            return False
        code = value.strip().lower()

        # Special-case: allow Norwegian 'no' explicitly
        if code == "no":
            return True

        # Disallow other YAML reserved literals
        if code in _YAML_RESERVED_WORDS:
            return False

        # Basic ISO 639-1/2 pattern with optional region/script suffix
        if re.fullmatch(r"[a-z]{2,3}(?:-[a-z]{2,3})?", code) is None:
            return False

        return True

    def _sanitize_path_identifier(self, value: str) -> str:
        """
        Sanitize path identifiers by replacing underscores with hyphens.

        Keeps underscores reserved for the filename schema separators:
        - single underscore _ : separates timestamp parts (YYYYMMDD_HHMMSS)
        - double underscore __ : separates run_label from text_base

        Args:
            value: The string to sanitize

        Returns:
            String with underscores replaced by hyphens
        """
        if not isinstance(value, str):
            return value
        return value.replace("_", "-")

    def _apply_path_sanitization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply path sanitization to identifiers used in path generation.

        IMPORTANT: This method ONLY sanitizes path-generating identifiers.
        Input file names (text_file, reference_audio) are NOT modified
        to ensure they can still be found by the file system.

        Sanitizes:
        - job: name (used for directory names)
        - job: run_label (used in filename schema)

        The filename schema: {run_label}__{text_base}_{timestamp}
        (double underscore separates run_label from text_base)

        Args:
            config: Configuration dictionary

        Returns:
            Config with sanitized path identifiers (input files unchanged)
        """
        # Create a deep copy to avoid modifying the original
        sanitized_config = copy.deepcopy(config)

        # Sanitize job-related identifiers
        if "job" in sanitized_config:
            job_section = sanitized_config["job"]

            # Sanitize job name
            if "name" in job_section and isinstance(job_section["name"], str):
                original_name = job_section["name"]
                sanitized_name = self._sanitize_path_identifier(original_name)
                if original_name != sanitized_name:
                    job_section["name"] = sanitized_name
                    logger.debug(
                        f"Sanitized job name: '{original_name}' → '{sanitized_name}'"
                    )

            # Sanitize run_label
            if "run_label" in job_section and isinstance(job_section["run_label"], str):
                original_label = job_section["run_label"]
                sanitized_label = self._sanitize_path_identifier(original_label)
                if original_label != sanitized_label:
                    job_section["run_label"] = sanitized_label
                    logger.debug(
                        f"Sanitized run_label: '{original_label}' → '{sanitized_label}'"
                    )

        # NOTE: Input files (text_file, reference_audio) are NOT sanitized!
        # They must remain unchanged to ensure file system can find them.
        # Only their STEMS are used (sanitized) for path generation in create_task_config().

        return sanitized_config

    def _dump_yaml_with_order(
        self, config_data: Dict[str, Any], file_handle: TextIO
    ) -> None:
        """Dump YAML with hierarchically preserved key order from default_config.yaml."""
        key_order_structure = self._get_default_key_order()

        # Sort the entire config data hierarchically
        sorted_config = self._sort_dict_hierarchically(config_data, key_order_structure)

        # Convert OrderedDict to regular dict for proper YAML serialization
        clean_config = self._convert_ordered_dict_to_dict(sorted_config)

        # Dump as YAML with preserved order
        yaml_str = yaml.dump(
            clean_config, default_flow_style=False, allow_unicode=True, sort_keys=False
        )
        file_handle.write(yaml_str)

    def _resolve_parent_chain(
        self,
        config_data: Dict[str, Any],
        origin_path: Path,
        _visited: Optional[set] = None,
        _depth: int = 0,
    ) -> Dict[str, Any]:
        """
        Recursively resolve the parent chain for a config file.

        Walks the chain of ``parent:`` references until either the root
        ``default_config.yaml`` is reached (no ``parent:`` key) or a file
        explicitly references ``default_config.yaml`` itself.

        Args:
            config_data: Already-loaded YAML dict of the *current* config.
            origin_path: Filesystem path of *current* config (for error messages).
            _visited: Internal set of resolved paths for cycle detection.
            _depth: Internal recursion depth counter.

        Returns:
            Fully-merged base config (parent chain collapsed, ``parent`` key
            stripped) to use as the ``default_config`` argument for the
            caller's ``merge_configs`` call.

        Raises:
            RuntimeError: On circular references or chain depth > 10.
            FileNotFoundError: When a referenced parent file does not exist.
        """
        MAX_DEPTH = 10
        if _depth > MAX_DEPTH:
            raise RuntimeError(
                f"Parent config chain exceeds maximum depth {MAX_DEPTH}. "
                f"Check for unintended deep nesting starting from {origin_path}."
            )

        if _visited is None:
            _visited = set()

        parent_ref = config_data.get("parent")
        if parent_ref is None:
            # No parent declared → fall back to the global default_config.yaml
            return self.load_default_config()

        parent_path = (self.config_dir / str(parent_ref)).resolve()

        if parent_path in _visited:
            raise RuntimeError(
                f"Circular parent reference detected: {parent_path} "
                f"is already in the resolution chain from {origin_path}."
            )
        _visited.add(parent_path)

        # Check whether the parent IS default_config.yaml itself (terminal case)
        if parent_path == self.default_config_path.resolve():
            return self.load_default_config()

        if not parent_path.exists():
            raise FileNotFoundError(
                f"Parent config not found: '{parent_path}' "
                f"(referenced from '{origin_path}')."
            )

        logger.debug(f"Resolving parent chain: {origin_path.name} -> {parent_path.name} (depth={_depth})")

        parent_config = self.load_job_config(parent_path)
        parent_base = self._resolve_parent_chain(
            parent_config, parent_path, _visited, _depth + 1
        )
        merged = self.merge_configs(parent_config, parent_base)
        merged.pop("parent", None)
        return merged

    def load_default_config(self) -> Dict[str, Any]:
        """Load the default pipeline configuration."""
        if "default" not in self._config_cache:
            logger.debug(f"Loading default config: {self.default_config_path}")
            with open(self.default_config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
                self._config_cache["default"] = self._sanitize_identifiers_post_load(raw)
        return copy.deepcopy(self._config_cache["default"])

    def load_job_config(self, config_path: Path) -> Dict[str, Any]:
        """Load a job configuration file."""
        config_key = str(config_path)
        if config_key not in self._config_cache:
            logger.debug(f"Loading job config: {config_path}")
            with open(config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
                self._config_cache[config_key] = self._sanitize_identifiers_post_load(raw)
        return copy.deepcopy(self._config_cache[config_key])
        
    def load_cascading_config(
        self, config_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Load configuration with N-level cascading logic via ``parent:`` references.

        Resolution order (from lowest to highest priority):
            default_config.yaml
                ↑ (optional intermediate parent chain via ``parent:`` key)
            job_config.yaml  (or any parent in the chain)
                ↑
            task_config.yaml  (runtime snapshot – no parent resolution)

        Each level only overrides values that are explicitly defined.
        Missing values automatically fall through to the next higher level.

        The optional top-level ``parent:`` key in any YAML file designates a
        parent config path relative to ``config/``.  The chain is resolved
        recursively until a file without a ``parent:`` is reached (which then
        falls back to ``default_config.yaml``).  Circular references and chains
        longer than 10 hops are detected and raise RuntimeError / FileNotFoundError.

        Args:
            config_path: Path to job-config or task-config file.  ``None``
                returns the bare default config.

        Returns:
            Merged configuration dictionary with sanitized job identifiers and
            the ``parent`` key stripped from the result.
        """
        if config_path is None:
            return self._apply_path_sanitization(self.load_default_config())

        if self.is_task_config(config_path):
            # Task-configs are self-contained runtime snapshots.
            # They are NOT subject to parent-chain resolution to avoid
            # alphabetical-sorting ambiguity and to keep them reproducible.
            task_config_data = self.load_job_config(config_path)
            config = self.load_default_config()
            config = self.merge_configs(task_config_data, config)
            logger.debug(
                f"Merged task config: {config_path} (no parent chain resolution)"
            )
        else:
            # Job-configs: resolve the full parent chain first, then merge the
            # job config on top.
            job_config = self.load_job_config(config_path)
            base = self._resolve_parent_chain(job_config, config_path)
            config = self.merge_configs(job_config, base)
            logger.debug(f"Merged job config with parent chain: {config_path}")

        # Strip the parent metadata key – it must not appear in runtime config
        config.pop("parent", None)

        # Apply path sanitization to final merged config
        # Sanitizes job: name and job: run_label for use in directory/filename schema:
        # {run_label}__{text_base}_{timestamp}
        config = self._apply_path_sanitization(config)

        return config

    def merge_configs(
        self, job_config: Dict[str, Any], default_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge job-yaml with default-yaml using cascading logic.

        Returns:
            Merged configuration
        """
        merged = copy.deepcopy(default_config)

        def merge_recursive(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            """Recursively merge source into target."""
            for key, value in source.items():
                if (
                    key in target
                    and isinstance(target[key], dict)
                    and isinstance(value, dict)
                ):
                    merge_recursive(target[key], value)
                else:
                    target[key] = value

        # Special handling for speakers array - merge by ID instead of replacing entire array
        if "generation" in job_config and "speakers" in job_config["generation"]:
            merged = self._merge_speakers_config(merged, job_config)
            # Remove speakers from job_config so it doesn't get processed again
            job_config_copy = copy.deepcopy(job_config)
            job_config_copy["generation"].pop("speakers", None)
            # Keep replace_speakers on job_config_copy so it is merged into `merged`
            # and persisted in task YAML (task reload uses merge_configs with default).
            merge_recursive(merged, job_config_copy)
        else:
            merge_recursive(merged, job_config)

        # Ensure default_speaker is valid after merging
        self._validate_and_fix_default_speaker(merged)

        return merged

    def _validate_and_fix_default_speaker(self, config: Dict[str, Any]) -> None:
        """
        Validate and fix default_speaker after merging configurations using cascading fallback logic:
        
        1. default_speaker from merged config (task/job YAML) 
        2. default_speaker from default_config.yaml
        3. First speaker from merged speakers list

        Args:
            config: Merged configuration to validate and fix
        """
        generation_config = config.get("generation", {})
        speakers = generation_config.get("speakers", [])
        current_default_speaker = generation_config.get("default_speaker")

        if not speakers:
            return

        speaker_ids = [speaker.get("id", "") for speaker in speakers]

        # Priority 1: Current default_speaker from merged config (task/job YAML)
        if current_default_speaker and current_default_speaker in speaker_ids:
            logger.debug(f"Validated task/job default speaker: '{current_default_speaker}'")
            return

        # Check for alias speakers that may represent the default speaker
        alias_speakers = [
            s for s in speakers if s.get("id") in ["default", "0", "reset"]
        ]
        
        # If we have alias speakers, the current default_speaker is still valid (represented by alias)
        if current_default_speaker and alias_speakers:
            logger.debug(f"Default speaker '{current_default_speaker}' represented by alias speakers")
            return

        # Priority 2: default_speaker from default_config.yaml
        try:
            default_config = self.load_default_config()
            original_default_speaker = default_config.get("generation", {}).get("default_speaker")
            
            if original_default_speaker and original_default_speaker in speaker_ids:
                generation_config["default_speaker"] = original_default_speaker
                logger.warning(
                    f"Task/job default_speaker '{current_default_speaker}' invalid, "
                    f"using default_config.yaml default_speaker: '{original_default_speaker}'"
                )
                return
                
        except Exception as e:
            logger.debug(f"Could not load default_config.yaml for validation fallback: {e}")

        # Priority 3: First speaker from merged speakers list
        if speakers:
            fallback_id = speakers[0].get("id", "default")
            generation_config["default_speaker"] = fallback_id
            
            if current_default_speaker:
                logger.warning(
                    f"Both task/job default_speaker '{current_default_speaker}' and "
                    f"default_config.yaml default_speaker are invalid, using first speaker: '{fallback_id}'"
                )
            else:
                logger.debug(f"No default_speaker specified, using first speaker: '{fallback_id}'")

    def _merge_speakers_config(
        self, base_config: Dict[str, Any], job_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge speakers configuration with intelligent ID-based merging.

        Rules:
        1. Speakers are merged by ID, not position
        2. Named speakers are merged by exact ID match
        3. New speakers are appended
        4. Missing tts_params are inherited from base config
        5. default_speaker aliases ("default", "0", "reset") reference the configured default_speaker
        6. If generation.replace_speakers is true, unprocessed base speakers are not appended

        Args:
            base_config: Base configuration (usually from default_config.yaml)
            job_config: Job-specific configuration

        Returns:
            Merged configuration with properly merged speakers
        """
        merged = copy.deepcopy(base_config)

        base_speakers = merged.get("generation", {}).get("speakers", [])
        job_speakers = job_config.get("generation", {}).get("speakers", [])
        replace_speakers = bool(
            job_config.get("generation", {}).get("replace_speakers")
        )

        if not job_speakers:
            return merged

        # Build lookup maps
        base_speakers_by_id = {speaker.get("id"): speaker for speaker in base_speakers}

        # Get default speaker info from base config
        base_default_speaker_id = self.get_default_speaker_id(merged)

        merged_speakers: List[Dict[str, Any]] = []
        processed_ids = set()

        # Process each job speaker
        for job_speaker in job_speakers:
            job_speaker_id = job_speaker.get("id")

            # 1. Handle default speaker aliases - merge with default speaker but keep alias ID
            if job_speaker_id in ["default", "0", "reset"]:
                # Use the configured default speaker from base config for merging
                base_speaker = base_speakers_by_id.get(base_default_speaker_id)
                # NOTE: Do NOT mark default speaker as processed - it should remain available!

                # Create merged speaker with alias ID, but inherit from default speaker
                if base_speaker:
                    merged_speaker = self._merge_single_speaker(
                        base_speaker, job_speaker
                    )
                    # Keep the alias ID instead of the original default speaker ID
                    merged_speaker["id"] = job_speaker_id
                else:
                    merged_speaker = copy.deepcopy(job_speaker)

                merged_speakers.append(merged_speaker)
                continue

            # 2. Apply complete cascading inheritance for all other speakers
            else:
                # Mark ID as processed if it exists in base config
                if job_speaker_id in base_speakers_by_id:
                    processed_ids.add(job_speaker_id)

                # Apply cascading inheritance to fill missing parameters
                merged_speaker = self._apply_cascading_inheritance(
                    job_speaker, job_config, base_config
                )

            merged_speakers.append(merged_speaker)

        # Add remaining base speakers that weren't overridden (unless job opts out)
        if not replace_speakers:
            for base_speaker in base_speakers:
                base_speaker_id = base_speaker.get("id")
                if base_speaker_id not in processed_ids:
                    merged_speakers.append(copy.deepcopy(base_speaker))

        # Update merged config
        merged["generation"]["speakers"] = merged_speakers

        logger.debug(f"Merged speakers: {[s.get('id') for s in merged_speakers]}")
        return merged

    def _merge_single_speaker(
        self, base_speaker: Dict[str, Any], job_speaker: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge a single speaker configuration with proper inheritance.

        Only overrides fields that are explicitly defined in job_speaker.
        Missing fields are inherited from base_speaker.

        Args:
            base_speaker: Base speaker configuration
            job_speaker: Job-specific speaker configuration

        Returns:
            Merged speaker configuration
        """
        merged = copy.deepcopy(base_speaker)

        # Only merge fields that are explicitly defined in job_speaker
        for key, value in job_speaker.items():
            if key in ["tts_params", "conservative_candidate"]:
                # Deep merge for nested objects
                if (
                    key in merged
                    and isinstance(merged[key], dict)
                    and isinstance(value, dict)
                ):
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = copy.deepcopy(value)
            else:
                # Direct override for simple fields (id, reference_audio)
                # Only override if the field is explicitly defined in job_speaker
                merged[key] = value

        return merged

    def _apply_cascading_inheritance(
        self, 
        job_speaker: Dict[str, Any], 
        job_config: Dict[str, Any], 
        base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply complete cascading inheritance for missing speaker parameters.

        Fallback hierarchy for missing parameters:
        1. Local default_speaker (from job config)
        2. Same speaker ID in default_config.yaml
        3. Default_speaker from default_config.yaml

        Args:
            job_speaker: Job speaker configuration (potentially incomplete)
            job_config: Complete job configuration  
            base_config: Complete default configuration

        Returns:
            Complete speaker configuration with all missing parameters filled
        """
        speaker_id = job_speaker.get("id", "unknown")
        merged = copy.deepcopy(job_speaker)

        # Get fallback sources in priority order
        fallback_sources = []
        
        # 1. Local default_speaker (from job config)
        job_default_speaker_id = job_config.get("generation", {}).get("default_speaker")
        if job_default_speaker_id and job_default_speaker_id != speaker_id:
            job_speakers = job_config.get("generation", {}).get("speakers", [])
            for speaker in job_speakers:
                if speaker.get("id") == job_default_speaker_id:
                    fallback_sources.append(("job default_speaker", speaker))
                    break

        # 2. Same speaker ID in default_config.yaml
        base_speakers = base_config.get("generation", {}).get("speakers", [])
        for speaker in base_speakers:
            if speaker.get("id") == speaker_id:
                fallback_sources.append(("base same ID", speaker))
                break

        # 3. Default_speaker from default_config.yaml
        base_default_speaker_id = base_config.get("generation", {}).get("default_speaker")
        if base_default_speaker_id:
            for speaker in base_speakers:
                if speaker.get("id") == base_default_speaker_id:
                    fallback_sources.append(("base default_speaker", speaker))
                    break

        # Apply cascading inheritance for missing parameters
        self._inherit_missing_parameters(merged, fallback_sources, speaker_id)

        return merged

    def _inherit_missing_parameters(
        self, 
        target_speaker: Dict[str, Any], 
        fallback_sources: List[tuple], 
        speaker_id: str
    ) -> None:
        """
        Inherit missing parameters from fallback sources in priority order.

        Args:
            target_speaker: Speaker to fill missing parameters for (modified in-place)
            fallback_sources: List of (source_name, speaker_config) tuples in priority order
            speaker_id: ID of target speaker for logging
        """
        # Define all possible speaker parameters
        all_params: Dict[str, Any] = {
            "reference_audio": None,
            "language": None,
            "seed": None,
            "tts_params": {},
            "conservative_candidate": {}
        }

        # Check and inherit top-level parameters
        for param_name in ["reference_audio", "language", "seed"]:
            if not target_speaker.get(param_name):
                for source_name, source_speaker in fallback_sources:
                    source_value = source_speaker.get(param_name)
                    if source_value:
                        target_speaker[param_name] = source_value
                        logger.debug(f"Speaker '{speaker_id}' inherited {param_name}='{source_value}' from {source_name}")
                        break

        # Check and inherit nested parameters  
        for nested_param in ["tts_params", "conservative_candidate"]:
            if nested_param not in target_speaker:
                target_speaker[nested_param] = {}
            
            target_nested = target_speaker[nested_param]
            
            # Define expected nested parameters
            if nested_param == "tts_params":
                # Union of all keys present in any fallback (model-agnostic; Qwen3/VibeVoice
                # keys are not hardcoded). Static allowlists went stale for top_k/subtalker_*.
                expected_keys_set: set[str] = set()
                for _, source_speaker in fallback_sources:
                    source_nested = source_speaker.get(nested_param, {})
                    if isinstance(source_nested, dict):
                        expected_keys_set.update(source_nested.keys())
                expected_keys = sorted(expected_keys_set)
            elif nested_param == "conservative_candidate":
                expected_keys = [
                    "enabled", "exaggeration", "cfg_weight", 
                    "temperature", "min_p", "top_p"
                ]
            else:
                expected_keys = []

            # Inherit missing nested parameters
            for key in expected_keys:
                if key not in target_nested or target_nested[key] is None:
                    for source_name, source_speaker in fallback_sources:
                        source_nested = source_speaker.get(nested_param, {})
                        if key in source_nested and source_nested[key] is not None:
                            target_nested[key] = source_nested[key]
                            logger.debug(f"Speaker '{speaker_id}' inherited {nested_param}.{key}={source_nested[key]} from {source_name}")
                            break

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration completeness.

        Returns:
            True if configuration is valid
        """
        required_sections = [
            "job",
            "input",
            "chunking",
            "generation",
            "validation",
            "audio",
        ]
        required_job_fields = ["name"]
        required_input_fields = ["text_file"]  # reference_audio now handled by speakers

        # Check main sections
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required config section: {section}")
                return False

        # Check job fields
        for field in required_job_fields:
            if field not in config["job"]:
                logger.error(f"Missing required job field: {field}")
                return False

        # Check input fields
        for field in required_input_fields:
            if field not in config["input"]:
                logger.error(f"Missing required input field: {field}")
                return False

        # Validate model_type against known values (soft check: warn only)
        known_model_types = {"standard", "multilingual", "turbo", "qwen3"} | set(VIBEVOICE_MODEL_TYPES)
        model_type = config.get("generation", {}).get("model_type", "standard")
        if model_type not in known_model_types:
            logger.warning(
                f"Unknown model_type '{model_type}' in generation config. "
                f"Supported values: {sorted(known_model_types)}"
            )

        # Validate speaker configuration
        if not self.validate_speakers_config(config):
            logger.error("Speaker configuration validation failed")
            return False

        return True

    def validate_speakers_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate speakers[] array configuration and default_speaker key.

        Returns:
            True if speakers configuration is valid
        """
        generation_config = config.get("generation", {})
        speakers = generation_config.get("speakers", [])
        default_speaker = generation_config.get("default_speaker", "")

        if not speakers:
            logger.error("No speakers defined in generation.speakers")
            return False

        # Validate default_speaker key
        if not default_speaker:
            logger.error("Missing required 'default_speaker' key in generation config")
            return False

        # Check if default_speaker ID exists in speakers list
        speaker_ids = [speaker.get("id", "") for speaker in speakers]
        if default_speaker not in speaker_ids:
            logger.error(
                f"default_speaker '{default_speaker}' not found in speakers list: {speaker_ids}"
            )
            return False

        logger.debug(f"Default speaker: '{default_speaker}'")

        # Validate unique IDs
        if len(speaker_ids) != len(set(speaker_ids)):
            logger.error("Duplicate speaker IDs found")
            return False

        # Validate each speaker has required fields
        for speaker in speakers:
            speaker_id = speaker.get("id", "")

            # Check required fields
            if not speaker_id:
                logger.error("Speaker configuration missing 'id' field")
                return False

            if not speaker.get("reference_audio"):
                logger.error(f"Speaker '{speaker_id}' missing reference_audio")
                return False

            if not speaker.get("tts_params"):
                logger.error(f"Speaker '{speaker_id}' missing tts_params")
                return False

            # Validate tts_params structure.
            # For Chatterbox models (standard/multilingual/turbo), the core Chatterbox params
            # are required.  For qwen3 these params are not applicable (gracefully skipped at
            # runtime), so we only check that tts_params itself is present and non-empty.
            tts_params = speaker.get("tts_params", {})
            model_type_for_validation = generation_config.get("model_type", "standard")
            if model_type_for_validation == "qwen3":
                if not tts_params:
                    logger.warning(
                        f"Speaker '{speaker_id}' has empty tts_params. "
                        "For qwen3, at least temperature is recommended."
                    )
            elif model_type_for_validation in VIBEVOICE_MODEL_TYPES:
                if not tts_params:
                    logger.warning(
                        f"Speaker '{speaker_id}' has empty tts_params. "
                        "For VibeVoice variants, cfg_scale, temperature, diffusion_steps, etc. are recommended."
                    )
            else:
                required_tts_params = ["exaggeration", "cfg_weight", "temperature", "min_p", "top_p"]
                for param in required_tts_params:
                    if param not in tts_params:
                        logger.error(f"Speaker '{speaker_id}' missing tts_params.{param}")
                        return False
            
            # Validate language: Graceful behavior; do not abort on invalid/missing values
            if speaker_id == default_speaker:
                lang_value = speaker.get("language")
                if not lang_value or not self._is_valid_language_code(lang_value):
                    # Degrade to info and allow fallbacks downstream (default_language → 'en')
                    logger.info(
                        "Default speaker '%s' has missing/invalid language '%s' – falling back at runtime (e.g. generation.default_language or 'en')",
                        speaker_id,
                        lang_value,
                    )
                else:
                    logger.debug(
                        f"Default speaker '{speaker_id}' language validated: {str(lang_value).strip().lower()}"
                    )

            # Validate optional language field for non-default speakers
            if "language" in speaker and speaker_id != default_speaker:
                lang_value = speaker.get("language")
                if lang_value:
                    if not self._is_valid_language_code(lang_value):
                        # Do not fail – let downstream fallbacks handle it
                        logger.info(
                            "Speaker '%s' has invalid language '%s' – falling back at runtime (e.g. default_language or 'en')",
                            speaker_id,
                            lang_value,
                        )
                    else:
                        logger.debug(f"Speaker '{speaker_id}' configured for language: {lang_value}")

        logger.debug(f"✅ Validated {len(speakers)} speakers: {speaker_ids}")
        return True
        
    def get_default_speaker_id(self, config: Dict[str, Any]) -> str:
        """
        Get the ID of the default speaker using cascading fallback logic:
        
        1. default_speaker from task/job config (if valid)
        2. default_speaker from default_config.yaml (if valid) 
        3. First speaker from merged speakers list

        Args:
            config: Merged configuration dictionary

        Returns:
            Default speaker ID
        """
        generation_config = config.get("generation", {})
        speakers = generation_config.get("speakers", [])
        speaker_ids = [speaker.get("id", "") for speaker in speakers]
        
        if not speakers:
            raise RuntimeError("No speakers configured")

        # Priority 1: default_speaker from merged config (task/job YAML)
        current_default_speaker = generation_config.get("default_speaker")
        if current_default_speaker and current_default_speaker in speaker_ids:
            logger.debug(f"Using task/job default speaker: '{current_default_speaker}'")
            return current_default_speaker

        # Priority 2: default_speaker from default_config.yaml
        try:
            default_config = self.load_default_config()
            original_default_speaker = default_config.get("generation", {}).get("default_speaker")
            
            if original_default_speaker and original_default_speaker in speaker_ids:
                logger.warning(
                    f"Task/job default_speaker '{current_default_speaker}' invalid, "
                    f"falling back to default_config.yaml default_speaker: '{original_default_speaker}'"
                )
                return original_default_speaker
                
        except Exception as e:
            logger.debug(f"Could not load default_config.yaml for fallback: {e}")

        # Priority 3: First speaker from merged speakers list
        fallback_id = speakers[0].get("id", "default")
        
        if current_default_speaker:
            logger.warning(
                f"Both task/job default_speaker '{current_default_speaker}' and "
                f"default_config.yaml default_speaker are invalid, "
                f"falling back to first speaker: '{fallback_id}'"
            )
        else:
            logger.warning(
                f"No default_speaker configured, using first speaker: '{fallback_id}'"
            )
            
        return fallback_id

    def _parse_task_filename(self, filename: str) -> tuple:
        """
        Parse a task filename into (run_label, text_base, timestamp).

        Supports two schemas:
        - New (double-underscore separator):  {run_label}__{text_base}_{YYYYMMDD}_{HHMMSS}
        - Legacy (single-underscore, compat): {text_base}_{YYYYMMDD}_{HHMMSS}
          or heuristic {run_label}_{text_base}_{YYYYMMDD}_{HHMMSS}

        Returns:
            Tuple of (run_label, text_base, timestamp)
        """
        fallback_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if "__" in filename:
            # New schema: {run_label}__{text_base}_{YYYYMMDD}_{HHMMSS}
            label_part, rest = filename.split("__", 1)
            parts = rest.split("_")
            if len(parts) >= 2:
                time_part = parts[-1]
                date_part = parts[-2]
                timestamp = f"{date_part}_{time_part}"
                text_base = "_".join(parts[:-2]) if len(parts) > 2 else parts[0]
            else:
                timestamp = fallback_timestamp
                text_base = rest
            return label_part, text_base, timestamp

        # Legacy schema: no __ separator
        parts = filename.split("_")
        if len(parts) >= 3:
            time_part = parts[-1]
            date_part = parts[-2]
            timestamp = f"{date_part}_{time_part}"
            if len(parts) >= 4:
                # Heuristic: first part is run_label
                run_label = parts[0]
                text_base = "_".join(parts[1:-2])
            else:
                run_label = ""
                text_base = parts[0]
        elif len(parts) >= 2:
            time_part = parts[-1]
            date_part = parts[-2]
            timestamp = f"{date_part}_{time_part}"
            run_label = ""
            text_base = parts[0]
        else:
            timestamp = fallback_timestamp
            run_label = ""
            text_base = filename

        return run_label, text_base, timestamp

    def create_task_config(
        self, config: Dict[str, Any], timestamp: Optional[str] = None
    ) -> TaskConfig:
        """
        Create a TaskConfig object from merged configuration.

        Sanitizes ONLY the path components used for directory/file generation,
        while preserving original input file names for system lookups.

        Returns:
            TaskConfig object
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        job_name = config["job"][
            "name"
        ]  # Already sanitized in _apply_path_sanitization()
        run_label = config["job"].get("run_label", "")  # Already sanitized
        text_file = config["input"]["text_file"]  # Original filename preserved

        # Extract and sanitize text_base for path generation ONLY
        original_text_base = Path(text_file).stem
        sanitized_text_base = self._sanitize_path_identifier(original_text_base)

        # Log sanitization if needed for debugging
        if original_text_base != sanitized_text_base:
            logger.debug(
                f"Sanitized text_base for path generation: '{original_text_base}' → '{sanitized_text_base}'"
            )

        # Create task directory name using sanitized components.
        # Double underscore __ separates run_label from text_base for unambiguous parsing.
        if run_label:
            task_dir_name = f"{run_label}__{sanitized_text_base}_{timestamp}"
            task_name = f"{run_label}__{sanitized_text_base}_{timestamp}"
        else:
            task_dir_name = f"{sanitized_text_base}_{timestamp}"
            task_name = f"{sanitized_text_base}_{timestamp}"

        # Create task directory path
        job_dir = self.output_dir / job_name
        task_directory = job_dir / task_dir_name

        return TaskConfig(
            task_name=task_name,
            run_label=run_label,
            timestamp=timestamp,
            base_output_dir=task_directory,
            config_path=Path(),  # Will be set when saving
            job_name=job_name,
        )

    def save_task_config(
        self, task_config: TaskConfig, config_data: Dict[str, Any]
    ) -> Path:
        """
        Save task configuration as task-yaml file with preserved key order.

        Uses sanitized filename components while preserving original input data.

        Returns:
            Path to saved task-yaml file
        """
        # Create job directory
        job_dir = self.output_dir / task_config.job_name
        job_dir.mkdir(parents=True, exist_ok=True)

        # Create task-yaml filename using sanitized components
        # Extract and sanitize text_base for filename generation
        text_file = config_data["input"][
            "text_file"
        ]  # Original filename preserved in config
        original_text_base = Path(text_file).stem
        sanitized_text_base = self._sanitize_path_identifier(original_text_base)

        if task_config.run_label:
            config_filename = f"{task_config.run_label}__{sanitized_text_base}_{task_config.timestamp}_config.yaml"
        else:
            config_filename = (
                f"{sanitized_text_base}_{task_config.timestamp}_config.yaml"
            )

        config_path = job_dir / config_filename

        # Save configuration with preserved key order
        # The config_data preserves original input filenames for system lookups
        with open(config_path, "w", encoding="utf-8") as f:
            self._dump_yaml_with_order(config_data, f)

        # Update task_config with saved path
        task_config.config_path = config_path

        logger.debug(f"Saved task config: {config_path}")
        return config_path

    def is_task_config(self, config_path: Path) -> bool:
        """
        Check if a config file is a task-yaml (located in job directory).

        Returns:
            True if it's a task-yaml file
        """
        try:
            # Check if path is within output directory structure
            relative_path = config_path.relative_to(self.output_dir)
            path_parts = relative_path.parts

            # Task-yaml should be: output/{job_name}/{task_config}.yaml
            if len(path_parts) == 2 and path_parts[1].endswith("_config.yaml"):
                return True
        except ValueError:
            # Path is not relative to output directory
            pass

        return False

    def load_task_config(self, config_path: Path) -> TaskConfig:
        """
        Load a saved task-yaml configuration file.

        Returns:
            TaskConfig object
        """
        # Load config data once and embed it immediately to avoid redundant loading
        config_data = self.load_cascading_config(config_path)

        # Extract info from file path
        filename = config_path.stem  # Remove .yaml extension
        if filename.endswith("_config"):
            filename = filename[:-7]  # Remove _config suffix

        run_label, text_base, timestamp = self._parse_task_filename(filename)

        # Determine task directory
        job_name = config_data["job"]["name"]
        task_directory = config_path.parent / filename

        task_name = f"{run_label}__{text_base}_{timestamp}" if run_label else f"{text_base}_{timestamp}"
        task_config = TaskConfig(
            task_name=task_name,
            run_label=run_label,
            timestamp=timestamp,
            base_output_dir=task_directory,
            config_path=config_path,
            job_name=job_name,
            preloaded_config=config_data,  # Embed config immediately to avoid redundant loading
        )

        return task_config

    def load_task_config_shallow(self, config_path: Path) -> TaskConfig:
        """Load a saved task-yaml with minimal parsing (no cascading merge).

        This avoids heavy operations (speaker inheritance, validation) during menu listing.
        The full cascading merge will occur later when executing the task.
        """
        # Directly parse the YAML without merging with defaults
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                raw = yaml.safe_load(f) or {}
                config_data = self._sanitize_identifiers_post_load(raw)
            except Exception:
                config_data = {}

        # Extract filename-based metadata (same as in load_task_config)
        filename = config_path.stem
        if filename.endswith("_config"):
            filename = filename[:-7]

        run_label, text_base, timestamp = self._parse_task_filename(filename)

        # Determine job name directly from YAML (fallback to parent dir name)
        job_name = (
            (config_data.get("job", {}) or {}).get("name")
            if isinstance(config_data, dict)
            else None
        ) or config_path.parent.name

        task_directory = config_path.parent / filename

        task_name = f"{run_label}__{text_base}_{timestamp}" if run_label else f"{text_base}_{timestamp}"
        task_config = TaskConfig(
            task_name=task_name,
            run_label=run_label,
            timestamp=timestamp,
            base_output_dir=task_directory,
            config_path=config_path,
            job_name=job_name,
            preloaded_config=None,  # Force full merge later during execution
        )

        return task_config

    def find_configs_by_job_name(self, job_name: str) -> List[Path]:
        """
        Find all configuration files (job-yamls) related to a specific job name or pattern.

        Supports glob patterns like:
        - "testjob*" - matches all jobs starting with "testjob"
        - "test*job" - matches jobs starting with "test" and ending with "job"
        - "testjob?" - matches "testjob" + single character
        - "testjob[12]" - matches "testjob1" or "testjob2"

        Returns:
            List of paths to job configuration files.
        """
        import fnmatch

        configs = []
        matched_job_names = set()  # Track matched job names to avoid duplicates

        # Search in config directory for job-yaml files
        for config_file in self.config_dir.glob("*.yaml"):
            if config_file.name == "default_config.yaml":
                continue  # Skip default config

            try:
                config_data = self.load_job_config(config_file)
                config_job_name = config_data.get("job", {}).get("name")
                if config_job_name and fnmatch.fnmatch(config_job_name, job_name):
                    configs.append(config_file)
                    matched_job_names.add(config_job_name)
            except Exception as e:
                logger.warning(f"Error reading config {config_file}: {e}")

        # Search in output directory for task-yaml files
        # Use glob patterns for directory names if job_name contains wildcards
        if any(char in job_name for char in ["*", "?", "["]):
            # Pattern matching for directory names
            try:
                for job_dir in self.output_dir.iterdir():
                    if job_dir.is_dir() and fnmatch.fnmatch(job_dir.name, job_name):
                        matched_job_names.add(job_dir.name)
                        for config_file in job_dir.glob("*_config.yaml"):
                            configs.append(config_file)
            except Exception as e:
                logger.warning(f"Error searching output directory: {e}")
        else:
            # Exact match for backward compatibility
            job_dir = self.output_dir / job_name
            if job_dir.exists():
                matched_job_names.add(job_name)
                for config_file in job_dir.glob("*_config.yaml"):
                    configs.append(config_file)

        # Log matching results
        if matched_job_names:
            logger.info(
                f"Pattern '{job_name}' matched jobs: {', '.join(sorted(matched_job_names))}"
            )
        else:
            logger.info(f"No jobs found matching pattern '{job_name}'")

        return configs

    def find_existing_tasks(
        self, job_name: str, run_label: Optional[str] = None, *, shallow: bool = False
    ) -> List[TaskConfig]:
        """
        Find existing completed task configurations within the output directory for a given job.

        Args:
            job_name: Name of the job to search for
            run_label: Optional run_label to filter tasks by. If provided, only tasks with matching run_label are returned.

        Returns:
            List of TaskConfig objects, sorted by timestamp (newest first)
        """
        tasks: List[TaskConfig] = []
        job_dir = self.output_dir / job_name

        if not job_dir.exists():
            if run_label:
                logger.debug(
                    f"Found 0 tasks for job '{job_name}' with run_label '{run_label}' (job directory not found)"
                )
            else:
                logger.debug(
                    f"Found 0 tasks for job '{job_name}' (job directory not found)"
                )
            return tasks

        # Pre-filter files based on filename pattern if run_label is specified
        if run_label:
            # Sanitize run_label for filename matching (same logic as in create_task_config).
            # Use single-underscore glob to match both new (__) and legacy (_) schemas,
            # since {run_label}__* is also matched by {run_label}_*.
            sanitized_run_label = self._sanitize_path_identifier(run_label)
            pattern = f"{sanitized_run_label}_*_config.yaml"
            config_files = list(job_dir.glob(pattern))
            logger.debug(
                f"Pre-filtering by filename pattern '{pattern}': found {len(config_files)} matching files"
            )
        else:
            config_files = list(job_dir.glob("*_config.yaml"))
            logger.debug(f"Scanning all config files: found {len(config_files)} files")

        for config_file in config_files:
            try:
                task_config = (
                    self.load_task_config_shallow(config_file)
                    if shallow
                    else self.load_task_config(config_file)
                )

                # Double-check run_label match (filename might not be perfect due to sanitization edge cases)
                if run_label and task_config.run_label != run_label:
                    logger.debug(
                        f"Skipping task {task_config.task_name} - run_label mismatch after loading: '{task_config.run_label}' != '{run_label}'"
                    )
                    continue

                tasks.append(task_config)
            except Exception as e:
                logger.warning(f"Error loading task config {config_file}: {e}")

        # Sort by timestamp (newest first) - convert to datetime for proper sorting
        def parse_timestamp(timestamp_str: str) -> datetime:
            try:
                return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except ValueError:
                # Fallback for malformed timestamps - put them at the end
                return datetime.min

        tasks.sort(key=lambda t: parse_timestamp(t.timestamp), reverse=True)

        if run_label:
            logger.debug(
                f"Found {len(tasks)} tasks for job '{job_name}' with run_label '{run_label}'"
            )
        else:
            logger.debug(
                f"Found {len(tasks)} tasks for job '{job_name}' (no run_label filter)"
            )

        return tasks
