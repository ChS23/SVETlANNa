"""Logger configuration utilities for SvetlannaTrainer.

Provides easy configuration of various Lightning loggers for optical neural network training.
Supports TensorBoard, Weights & Biases, MLflow, Neptune, Comet, and CSV logging.
"""

import warnings
from pathlib import Path
from typing import Any

try:
    from lightning.pytorch.loggers import (
        CometLogger,
        CSVLogger,
        Logger,
        MLFlowLogger,
        NeptuneLogger,
        TensorBoardLogger,
        WandbLogger,
    )
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False


class LoggerConfig:
    """Utility class for creating and configuring Lightning loggers.
    
    Supports all major ML experiment tracking platforms with sensible defaults
    for optical neural network experiments.
    """

    @staticmethod
    def create_logger(
        logger_type: str,
        save_dir: str = "./logs",
        experiment_name: str = "svetlanna_training",
        **kwargs
    ) -> Logger | None:
        """Create a Lightning logger with specified configuration.
        
        Args:
            logger_type: Type of logger ('tensorboard', 'wandb', 'mlflow', 'neptune', 'comet', 'csv')
            save_dir: Directory to save logs
            experiment_name: Name of the experiment
            **kwargs: Additional logger-specific parameters
            
        Returns:
            Configured Lightning logger or None if logger cannot be created
            
        Examples:
            >>> # TensorBoard logger
            >>> logger = LoggerConfig.create_logger('tensorboard', save_dir='./tb_logs')
            
            >>> # Weights & Biases logger
            >>> logger = LoggerConfig.create_logger('wandb', project='optical-nn', entity='myteam')
            
            >>> # MLflow logger
            >>> logger = LoggerConfig.create_logger('mlflow', experiment_name='d2nn_optimization')
        """
        if not LIGHTNING_AVAILABLE:
            warnings.warn("Lightning not available, cannot create logger")
            return None

        logger_type = logger_type.lower()

        try:
            if logger_type == "tensorboard":
                return LoggerConfig._create_tensorboard_logger(save_dir, experiment_name, **kwargs)
            if logger_type == "wandb":
                return LoggerConfig._create_wandb_logger(save_dir, experiment_name, **kwargs)
            if logger_type == "mlflow":
                return LoggerConfig._create_mlflow_logger(save_dir, experiment_name, **kwargs)
            if logger_type == "neptune":
                return LoggerConfig._create_neptune_logger(save_dir, experiment_name, **kwargs)
            if logger_type == "comet":
                return LoggerConfig._create_comet_logger(save_dir, experiment_name, **kwargs)
            if logger_type == "csv":
                return LoggerConfig._create_csv_logger(save_dir, experiment_name, **kwargs)
            raise ValueError(
                f"Unknown logger type: {logger_type}. "
                f"Supported: 'tensorboard', 'wandb', 'mlflow', 'neptune', 'comet', 'csv'"
            )
        except Exception as e:
            warnings.warn(f"Failed to create {logger_type} logger: {e}")
            return None

    @staticmethod
    def _create_tensorboard_logger(save_dir: str, name: str, **kwargs) -> TensorBoardLogger:
        """Create TensorBoard logger with optical experiment defaults."""
        defaults = {
            "save_dir": save_dir,
            "name": name,
            "version": None,  # Auto-increment
            "log_graph": True,
            "default_hp_metric": True,
            "prefix": ""
        }
        defaults.update(kwargs)

        # Check if tensorboard is available
        try:
            import tensorboard
        except ImportError:
            try:
                import tensorboardX
            except ImportError:
                raise ImportError(
                    "TensorBoard not available. Install with: pip install tensorboard"
                )

        return TensorBoardLogger(**defaults)

    @staticmethod
    def _create_wandb_logger(save_dir: str, name: str, **kwargs) -> WandbLogger:
        """Create Weights & Biases logger with optical experiment defaults."""
        # Check if wandb is available
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "Weights & Biases not available. Install with: pip install wandb"
            )

        defaults = {
            "save_dir": save_dir,
            "name": name,
            "project": kwargs.pop("project", "optical-neural-networks"),
            "entity": kwargs.pop("entity", None),
            "offline": kwargs.pop("offline", False),
            "id": kwargs.pop("id", None),
            "anonymous": kwargs.pop("anonymous", None),
            "version": kwargs.pop("version", None),
            "tags": kwargs.pop("tags", ["optical-nn", "svetlanna"]),
            "log_model": kwargs.pop("log_model", False),
            "prefix": kwargs.pop("prefix", ""),
            "job_type": kwargs.pop("job_type", "train")
        }
        defaults.update(kwargs)

        return WandbLogger(**defaults)

    @staticmethod
    def _create_mlflow_logger(save_dir: str, name: str, **kwargs) -> MLFlowLogger:
        """Create MLflow logger with optical experiment defaults."""
        # Check if mlflow is available
        try:
            import mlflow
        except ImportError:
            raise ImportError(
                "MLflow not available. Install with: pip install mlflow"
            )

        defaults = {
            "experiment_name": name,
            "tracking_uri": kwargs.pop("tracking_uri", f"file://{Path(save_dir).absolute()}"),
            "tags": kwargs.pop("tags", {"framework": "svetlanna", "type": "optical-nn"}),
            "save_dir": save_dir,
            "prefix": kwargs.pop("prefix", ""),
            "artifact_location": kwargs.pop("artifact_location", None),
            "run_id": kwargs.pop("run_id", None)
        }
        defaults.update(kwargs)

        return MLFlowLogger(**defaults)

    @staticmethod
    def _create_neptune_logger(save_dir: str, name: str, **kwargs) -> NeptuneLogger:
        """Create Neptune logger with optical experiment defaults."""
        # Check if neptune is available
        try:
            import neptune
        except ImportError:
            raise ImportError(
                "Neptune not available. Install with: pip install neptune"
            )

        defaults = {
            "project": kwargs.pop("project", "optical-nn/svetlanna"),
            "name": name,
            "save_dir": save_dir,
            "prefix": kwargs.pop("prefix", ""),
            "tags": kwargs.pop("tags", ["optical-nn", "svetlanna"]),
            "api_token": kwargs.pop("api_token", None),  # Should be set via NEPTUNE_API_TOKEN env var
            "mode": kwargs.pop("mode", "async")
        }
        defaults.update(kwargs)

        return NeptuneLogger(**defaults)

    @staticmethod
    def _create_comet_logger(save_dir: str, name: str, **kwargs) -> CometLogger:
        """Create Comet logger with optical experiment defaults."""
        # Check if comet_ml is available
        try:
            import comet_ml
        except ImportError:
            raise ImportError(
                "Comet ML not available. Install with: pip install comet_ml"
            )

        defaults = {
            "save_dir": save_dir,
            "experiment_name": name,
            "project_name": kwargs.pop("project_name", "optical-neural-networks"),
            "workspace": kwargs.pop("workspace", None),
            "api_key": kwargs.pop("api_key", None),  # Should be set via COMET_API_KEY env var
            "experiment_key": kwargs.pop("experiment_key", None),
            "offline": kwargs.pop("offline", False),
            "prefix": kwargs.pop("prefix", ""),
            "tags": kwargs.pop("tags", ["optical-nn", "svetlanna"])
        }
        defaults.update(kwargs)

        return CometLogger(**defaults)

    @staticmethod
    def _create_csv_logger(save_dir: str, name: str, **kwargs) -> CSVLogger:
        """Create CSV logger (always available)."""
        defaults = {
            "save_dir": save_dir,
            "name": name,
            "version": kwargs.pop("version", None),
            "prefix": kwargs.pop("prefix", ""),
            "flush_logs_every_n_steps": kwargs.pop("flush_logs_every_n_steps", 100)
        }
        defaults.update(kwargs)

        return CSVLogger(**defaults)


def create_multiple_loggers(
    logger_configs: list[dict[str, Any]],
    save_dir: str = "./logs",
    experiment_name: str = "svetlanna_training"
) -> list[Logger]:
    """Create multiple loggers for simultaneous logging to different platforms.
    
    Args:
        logger_configs: List of logger configurations
        save_dir: Base directory for logs
        experiment_name: Base experiment name
        
    Returns:
        List of configured loggers
        
    Example:
        >>> loggers = create_multiple_loggers([
        ...     {'type': 'tensorboard'},
        ...     {'type': 'csv'},
        ...     {'type': 'wandb', 'project': 'my-optical-project'}
        ... ])
    """
    loggers = []

    for config in logger_configs:
        logger_type = config.pop("type")
        logger = LoggerConfig.create_logger(
            logger_type=logger_type,
            save_dir=save_dir,
            experiment_name=experiment_name,
            **config
        )
        if logger is not None:
            loggers.append(logger)

    return loggers


# Predefined logger configurations for common use cases
COMMON_LOGGER_CONFIGS = {
    "development": [
        {"type": "csv"},
        {"type": "tensorboard"}
    ],

    "research": [
        {"type": "tensorboard"},
        {"type": "wandb", "project": "optical-research"}
    ],

    "production": [
        {"type": "mlflow"},
        {"type": "csv"}
    ],

    "comprehensive": [
        {"type": "tensorboard"},
        {"type": "wandb", "project": "optical-neural-networks"},
        {"type": "csv"}
    ]
}


def get_logger_for_setup(
    setup_type: str = "development",
    save_dir: str = "./logs",
    experiment_name: str = "svetlanna_training",
    **override_configs
) -> Logger | list[Logger]:
    """Get pre-configured logger(s) for common setups.
    
    Args:
        setup_type: Type of setup ('development', 'research', 'production', 'comprehensive')
        save_dir: Directory to save logs
        experiment_name: Name of the experiment
        **override_configs: Override configurations for specific loggers
        
    Returns:
        Single logger or list of loggers
        
    Examples:
        >>> # Development setup (CSV + TensorBoard)
        >>> loggers = get_logger_for_setup('development')
        
        >>> # Research setup with custom Wandb project
        >>> loggers = get_logger_for_setup('research', wandb_project='my-research')
    """
    if setup_type not in COMMON_LOGGER_CONFIGS:
        raise ValueError(
            f"Unknown setup type: {setup_type}. "
            f"Available: {list(COMMON_LOGGER_CONFIGS.keys())}"
        )

    configs = COMMON_LOGGER_CONFIGS[setup_type].copy()

    # Apply overrides
    for config in configs:
        logger_type = config["type"]
        override_key = f"{logger_type}_"

        for key, value in override_configs.items():
            if key.startswith(override_key):
                config_key = key[len(override_key):]
                config[config_key] = value

    loggers = create_multiple_loggers(configs, save_dir, experiment_name)

    # Return single logger if only one, otherwise return list
    if len(loggers) == 1:
        return loggers[0]
    return loggers
