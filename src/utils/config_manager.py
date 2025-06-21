#!/usr/bin/env python3
"""
ConfigManager for cascading configuration loading.
Supports default-yaml + job-yaml merging and task-yaml creation.
"""

import copy
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """Configuration for a task."""

    task_name: str
    run_label: str
    timestamp: str
    base_output_dir: Path
    config_path: Path
    job_name: str
    add_final: bool = False  # Force regeneration of final audio
    preloaded_config: Optional[Dict[str, Any]] = None  # Avoid redundant config loading

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

    def load_default_config(self) -> Dict[str, Any]:
        """Load the default pipeline configuration."""
        if "default" not in self._config_cache:
            logger.debug(f"Loading default config: {self.default_config_path}")
            with open(self.default_config_path, "r", encoding="utf-8") as f:
                self._config_cache["default"] = yaml.safe_load(f)
        return copy.deepcopy(self._config_cache["default"])

    def load_job_config(self, config_path: Path) -> Dict[str, Any]:
        """Load a job configuration file."""
        config_key = str(config_path)
        if config_key not in self._config_cache:
            logger.debug(f"Loading job config: {config_path}")
            with open(config_path, "r", encoding="utf-8") as f:
                self._config_cache[config_key] = yaml.safe_load(f)
        return copy.deepcopy(self._config_cache[config_key])

    def find_parent_job_config(self, task_config_path: Path) -> Optional[Path]:
        """
        Find the parent job-config file for a given task-config.
        
        Args:
            task_config_path: Path to the task-config file
            
        Returns:
            Path to the parent job-config file, or None if not found
        """
        if not self.is_task_config(task_config_path):
            return None
        
        # Extract job_name from task-config data
        task_data = self.load_job_config(task_config_path)
        job_name = task_data.get("job", {}).get("name")
        
        if not job_name:
            logger.warning(f"No job name found in task config: {task_config_path}")
            return None
        
        # Search for job-config files in config directory
        for config_file in self.config_dir.glob("*.yaml"):
            if config_file.name == "default_config.yaml":
                continue  # Skip default config
                
            try:
                config_data = self.load_job_config(config_file)
                if config_data.get("job", {}).get("name") == job_name:
                    logger.debug(f"Found parent job config for {task_config_path}: {config_file}")
                    return config_file
            except Exception as e:
                logger.warning(f"Error reading config {config_file}: {e}")
        
        logger.debug(f"No parent job config found for task: {task_config_path}")
        return None

    def load_cascading_config(
        self, config_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Load configuration with true 3-level cascading logic:
        default_config.yaml → job_config.yaml → task_config.yaml
        
        Each level only overrides values that are explicitly defined.
        Missing values automatically fall through to the next higher level.
        
        Args:
            config_path: Path to job-config or task-config file
            
        Returns:
            Merged configuration dictionary
        """
        # Start with default config (complete base configuration)
        config = self.load_default_config()
        
        if config_path is None:
            return config
        
        if self.is_task_config(config_path):
            # 3-level cascade: default → job → task
            
            # Load task-config once and reuse it
            task_config_data = self.load_job_config(config_path)
            
            # First, try to find and merge parent job-config using job_name from already loaded task-config
            job_name = task_config_data.get("job", {}).get("name")
            if job_name:
                # Search for job-config files in config directory
                parent_job_config_path = None
                for config_file in self.config_dir.glob("*.yaml"):
                    if config_file.name == "default_config.yaml":
                        continue  # Skip default config
                        
                    try:
                        config_data = self.load_job_config(config_file)
                        if config_data.get("job", {}).get("name") == job_name:
                            parent_job_config_path = config_file
                            break
                    except Exception as e:
                        logger.warning(f"Error reading config {config_file}: {e}")
                
                if parent_job_config_path:
                    job_config = self.load_job_config(parent_job_config_path)
                    config = self.merge_configs(job_config, config)
                    logger.debug(f"Merged parent job config: {parent_job_config_path}")
                else:
                    logger.debug(f"No parent job config found for task: {config_path}")
                    logger.debug(f"No parent job config found, using default config as base")
            else:
                logger.warning(f"No job name found in task config: {config_path}")
                logger.debug(f"No parent job config found, using default config as base")
            
            # Then merge task-config on top (reuse already loaded task_config_data)
            config = self.merge_configs(task_config_data, config)
            logger.debug(f"Merged task config: {config_path}")
            
        else:
            # 2-level cascade: default → job
            job_config = self.load_job_config(config_path)
            config = self.merge_configs(job_config, config)
            logger.debug(f"Merged job config: {config_path}")
        
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

        merge_recursive(merged, job_config)
        return merged

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
        required_input_fields = ["reference_audio", "text_file"]

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

        return True

    def create_task_config(
        self, config: Dict[str, Any], timestamp: Optional[str] = None
    ) -> TaskConfig:
        """
        Create a TaskConfig object from merged configuration.

        Returns:
            TaskConfig object
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        job_name = config["job"]["name"]
        run_label = config["job"].get("run-label", "")
        text_file = config["input"]["text_file"]

        # Create task directory name
        text_base = Path(text_file).stem
        if run_label:
            task_dir_name = f"{text_base}_{run_label}_{timestamp}"
        else:
            task_dir_name = f"{text_base}_{timestamp}"

        # Create task directory path
        job_dir = self.output_dir / job_name
        task_directory = job_dir / task_dir_name

        return TaskConfig(
            task_name=f"{text_base}_{run_label}_{timestamp}",
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
        Save task configuration as task-yaml file.

        Returns:
            Path to saved task-yaml file
        """
        # Create job directory
        job_dir = self.output_dir / task_config.job_name
        job_dir.mkdir(parents=True, exist_ok=True)

        # Create task-yaml filename
        # Extract original text_base from config_data instead of using task_name (which already has timestamp)
        text_file = config_data["input"]["text_file"]
        text_base = Path(text_file).stem
        if task_config.run_label:
            config_filename = f"{text_base}_{task_config.run_label}_{task_config.timestamp}_config.yaml"
        else:
            config_filename = f"{text_base}_{task_config.timestamp}_config.yaml"

        config_path = job_dir / config_filename

        # Save configuration
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

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

        # Parse filename components
        parts = filename.split("_")
        if len(parts) >= 3:
            # Format: text_base_run_label_YYYYMMDD_HHMMSS
            # Last two parts are date and time
            time_part = parts[-1]  # HHMMSS
            date_part = parts[-2]  # YYYYMMDD
            timestamp = f"{date_part}_{time_part}"  # YYYYMMDD_HHMMSS

            if len(parts) >= 4:
                run_label = parts[-3]
                text_base = "_".join(parts[:-3])
            else:
                run_label = ""
                text_base = parts[0]
        elif len(parts) >= 2:
            # Format: text_base_YYYYMMDD_HHMMSS (no run_label)
            time_part = parts[-1]  # HHMMSS
            date_part = parts[-2]  # YYYYMMDD
            timestamp = f"{date_part}_{time_part}"  # YYYYMMDD_HHMMSS
            run_label = ""
            text_base = parts[0]
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_label = ""
            text_base = filename

        # Determine task directory
        job_name = config_data["job"]["name"]
        task_directory = config_path.parent / filename

        task_config = TaskConfig(
            task_name=f"{text_base}_{run_label}_{timestamp}",
            run_label=run_label,
            timestamp=timestamp,
            base_output_dir=task_directory,
            config_path=config_path,
            job_name=job_name,
            preloaded_config=config_data,  # Embed config immediately to avoid redundant loading
        )
        
        return task_config

    def find_configs_by_job_name(self, job_name: str) -> List[Path]:
        """
        Find all configuration files (job-yamls) related to a specific job name.

        Returns:
            List of paths to job configuration files.
        """
        configs = []

        # Search in config directory for job-yaml files
        for config_file in self.config_dir.glob("*_config.yaml"):
            try:
                config_data = self.load_job_config(config_file)
                if config_data.get("job", {}).get("name") == job_name:
                    configs.append(config_file)
            except Exception as e:
                logger.warning(f"Error reading config {config_file}: {e}")

        # Search in output directory for task-yaml files
        job_dir = self.output_dir / job_name
        if job_dir.exists():
            for config_file in job_dir.glob("*_config.yaml"):
                configs.append(config_file)

        return configs

    def find_existing_tasks(self, job_name: str) -> List[TaskConfig]:
        """
        Find existing completed task configurations within the output directory for a given job.

        Returns:
            List of TaskConfig objects, sorted by timestamp (newest first)
        """
        tasks = []
        job_dir = self.output_dir / job_name

        if job_dir.exists():
            for config_file in job_dir.glob("*_config.yaml"):
                try:
                    task_config = self.load_task_config(config_file)
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
        return tasks
