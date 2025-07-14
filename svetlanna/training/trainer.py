import inspect
import warnings
from collections.abc import Callable
from typing import Any

import lightning as L
import torch
from lightning import Trainer as LightningTrainer
from torch import nn

from svetlanna.setup import LinearOpticalSetup

from .lightning_adapter import LinearOpticalSetupLightning
from .loggers import create_multiple_loggers, get_logger_for_setup, LoggerConfig
from .multi_objective import MultiObjectiveOptimizer, ObjectiveFunction

# Type definitions for enhanced search space
SearchSpaceValue = tuple[
    tuple[int, float] | tuple[int, float, dict[str, Any]], # (min, max) for int/float
    tuple[int, float, dict[str, Any]], # (min, max, kwargs)
    list[str | int | float],                             # choices for categorical
    dict[str, Any]                                            # full specification
]


class SvetlannaTrainer:
    """High-level trainer for optical neural networks with automatic hyperparameter optimization.
    
    Provides a two-level API:
    - Simple interface for physicists: just pass generator and datamodule
    - Advanced interface for ML experts: full control over Lightning configuration
    
    Features:
    - Automatic mode detection based on generator signature
    - Simple training without optimization
    - Hyperparameter optimization with Optuna (optional)
    - Full PyTorch Lightning compatibility
    - Easy access to internal components
    
    Philosophy: "Simple by default, powerful when needed"
    
    Example:
        >>> # Simple training
        >>> trainer = SvetlannaTrainer.from_generator(
        ...     generator=d2nn_generator,
        ...     datamodule=my_datamodule
        ... )
        >>> results = trainer.fit()
        
        >>> # With hyperparameter optimization
        >>> trainer = SvetlannaTrainer.from_generator(
        ...     generator=d2nn_generator,
        ...     datamodule=my_datamodule,
        ...     search_space={
        ...         'n_layers': (3, 10),
        ...         'layer_distance': (0.05, 0.2)
        ...     }
        ... )
        >>> study = trainer.optimize()
    """

    def __init__(
        self,
        lightning_module: LinearOpticalSetupLightning,
        datamodule: L.LightningDataModule,
        trainer_config: dict[str, Any] | None = None
    ):
        """Initialize SvetlannaTrainer with pre-configured components.
        
        This is the low-level constructor. For high-level usage, use from_generator().
        
        Args:
            lightning_module: Pre-configured LinearOpticalSetupLightning module
            datamodule: Lightning DataModule with training data
            trainer_config: Configuration for Lightning Trainer
        """
        self._lightning_module = lightning_module
        self._datamodule = datamodule
        self._trainer_config = trainer_config or {}

        # Lightning trainer is created lazily
        self._lightning_trainer = None

        # Optuna components (set when using optimization)
        self._optuna_optimizer = None
        self._search_space = None

    @classmethod
    def from_generator(
        cls,
        generator: Callable[..., LinearOpticalSetup],
        datamodule: L.LightningDataModule,
        search_space: dict[str, SearchSpaceValue] | None = None,
        n_trials: int = 100,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        optimizer_config: dict[str, Any] | None = None,
        trainer_config: dict[str, Any] | None = None,
        metrics: list[Any] | None = None,
        objectives: list[ObjectiveFunction] | None = None,
        multi_objective_approach: str = "pareto",
        sampler: str | Any | None = None,
        sampler_config: dict[str, Any] | None = None,
        pruner: str | Any | None = None,
        pruner_config: dict[str, Any] | None = None,
        storage: str | Any | None = None,
        study_name: str | None = None,
        callbacks: list[Callable] | None = None,
        logger: str | dict[str, Any] | list[dict[str, Any]] | Any | None = None,
        log_dir: str = "./logs"
    ) -> "SvetlannaTrainer":
        # TODO: mb move part parameters to train start method
        """Create SvetlannaTrainer from optical system generator.
        
        This is the main factory method that automatically determines the training mode
        based on whether search_space is provided.
        
        Args:
            generator: Function that generates LinearOpticalSetup instances
            datamodule: Lightning DataModule with training data
            search_space: Enhanced search space supporting all Optuna suggest methods:
                - (min, max): suggest_int/suggest_float based on type
                - (min, max, kwargs): with additional Optuna parameters
                - [choice1, choice2, ...]: suggest_categorical
                - {'type': 'int', 'low': 1, 'high': 10, 'step': 2}: full specification
                - {'type': 'float', 'low': 0.1, 'high': 1.0, 'log': True}: log scale
                - {'type': 'categorical', 'choices': ['a', 'b', 'c']}: categorical
            n_trials: Number of Optuna trials (only used if search_space provided)
            loss_fn: Loss function (defaults to MSELoss)
            optimizer_config: Optimizer configuration
            trainer_config: Lightning Trainer configuration
            metrics: List of metrics to track
            objectives: List of ObjectiveFunction for multi-objective optimization
            multi_objective_approach: 'pareto' for Pareto optimization, 'weighted' for scalarization
            sampler: Optuna sampler ('tpe', 'random', 'grid', 'cmaes', 'nsgaii') or sampler instance
            sampler_config: Configuration parameters for the chosen sampler
            pruner: Optuna pruner ('hyperband', 'median', 'successive_halving', 'percentile') or pruner instance
            pruner_config: Configuration parameters for the chosen pruner
            storage: Storage backend (string URL or Storage instance) for study persistence
            study_name: Name for the study (used with storage backends)
            callbacks: List of callback functions called during optimization
            logger: Logger configuration:
                - str: logger type ('tensorboard', 'wandb', 'csv', 'development', 'research', 'production')  
                - dict: single logger config {'type': 'wandb', 'project': 'my-project'}
                - list: multiple loggers [{'type': 'tensorboard'}, {'type': 'csv'}]
                - Lightning logger instance
                - None: disable logging
            log_dir: Directory to save logs (default: "./logs")
            
        Returns:
            SvetlannaTrainer: Configured trainer instance
            
        Raises:
            ValueError: If search_space doesn't match generator signature
            TypeError: If generator is not callable
            
        Examples:
            Enhanced search space formats:
            >>> search_space = {
            ...     # Simple range - auto-detects int/float
            ...     'n_layers': (3, 10),
            ...     'layer_distance': (0.05, 0.2),
            ...     
            ...     # With step parameter
            ...     'batch_size': (16, 128, {'step': 16}),
            ...     
            ...     # Categorical choices
            ...     'method': ['fresnel', 'AS', 'FFT'],
            ...     
            ...     # Log scale for learning rate
            ...     'learning_rate': {
            ...         'type': 'float',
            ...         'low': 1e-5,
            ...         'high': 1e-1,
            ...         'log': True
            ...     },
            ...     
            ...     # Integer with step
            ...     'hidden_dim': {
            ...         'type': 'int',
            ...         'low': 64,
            ...         'high': 512,
            ...         'step': 64
            ...     }
            ... }
        """
        if not callable(generator):
            raise TypeError(f"generator must be callable, got {type(generator)}")

        # Validate search_space against generator signature
        cls._validate_search_space(generator, search_space)

        # Set default loss function
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        # Determine training mode
        if search_space is None:
            # Simple training mode: use default parameters
            optical_setup = cls._create_optical_setup_with_defaults(generator)

            # Create Lightning module
            lightning_module = LinearOpticalSetupLightning(
                optical_setup=optical_setup,
                loss_fn=loss_fn,
                optimizer_config=optimizer_config,
                metrics=metrics
            )

            # Create trainer
            trainer = cls(
                lightning_module=lightning_module,
                datamodule=datamodule,
                trainer_config=trainer_config
            )

        else:
            # Optimization mode: defer optical setup creation to Optuna
            # For now, create a dummy optical setup to satisfy Lightning module requirements
            # The real optimization will happen in optimize() method
            dummy_optical_setup = cls._create_optical_setup_with_defaults(generator)

            lightning_module = LinearOpticalSetupLightning(
                optical_setup=dummy_optical_setup,
                loss_fn=loss_fn,
                optimizer_config=optimizer_config,
                metrics=metrics
            )

            trainer = cls(
                lightning_module=lightning_module,
                datamodule=datamodule,
                trainer_config=trainer_config
            )

            # Set up Optuna components
            trainer._search_space = search_space
            trainer._generator = generator
            trainer._loss_fn = loss_fn
            trainer._optimizer_config = optimizer_config
            trainer._metrics = metrics
            trainer._n_trials = n_trials
            trainer._objectives = objectives
            trainer._multi_objective_approach = multi_objective_approach
            trainer._sampler = sampler
            trainer._sampler_config = sampler_config or {}
            trainer._pruner = pruner
            trainer._pruner_config = pruner_config or {}
            trainer._storage = storage
            trainer._study_name = study_name
            trainer._callbacks = callbacks or []

        # Configure logger
        trainer._configure_logger(logger, log_dir, trainer_config)

        return trainer

    def _configure_logger(
        self,
        logger_config: str | dict[str, Any] | list[dict[str, Any]] | Any | None,
        log_dir: str,
        trainer_config: dict[str, Any]
    ) -> None:
        """Configure Lightning logger based on provided configuration."""
        if logger_config is None:
            # Explicitly disable logging
            trainer_config["logger"] = False
            return

        if isinstance(logger_config, str):
            # String configuration - either logger type or preset
            if logger_config in ["development", "research", "production", "comprehensive"]:
                # Preset configuration
                try:
                    lightning_logger = get_logger_for_setup(
                        setup_type=logger_config,
                        save_dir=log_dir,
                        experiment_name=f"svetlanna_{logger_config}"
                    )
                    trainer_config["logger"] = lightning_logger
                except Exception as e:
                    warnings.warn(f"Failed to create preset logger '{logger_config}': {e}")
                    trainer_config["logger"] = False
            else:
                # Single logger type
                try:
                    lightning_logger = LoggerConfig.create_logger(
                        logger_type=logger_config,
                        save_dir=log_dir,
                        experiment_name="svetlanna_training"
                    )
                    trainer_config["logger"] = lightning_logger
                except Exception as e:
                    warnings.warn(f"Failed to create {logger_config} logger: {e}")
                    trainer_config["logger"] = False

        elif isinstance(logger_config, dict):
            # Single logger with custom config
            logger_type = logger_config.get("type", "csv")
            try:
                lightning_logger = LoggerConfig.create_logger(
                    logger_type=logger_type,
                    save_dir=log_dir,
                    experiment_name="svetlanna_training",
                    **{k: v for k, v in logger_config.items() if k != "type"}
                )
                trainer_config["logger"] = lightning_logger
            except Exception as e:
                warnings.warn(f"Failed to create {logger_type} logger: {e}")
                trainer_config["logger"] = False

        elif isinstance(logger_config, list):
            # Multiple loggers
            try:
                lightning_loggers = create_multiple_loggers(
                    logger_configs=logger_config,
                    save_dir=log_dir,
                    experiment_name="svetlanna_training"
                )
                if lightning_loggers:
                    trainer_config["logger"] = lightning_loggers
                else:
                    trainer_config["logger"] = False
            except Exception as e:
                warnings.warn(f"Failed to create multiple loggers: {e}")
                trainer_config["logger"] = False

        else:
            # Assume it's already a Lightning logger instance
            trainer_config["logger"] = logger_config

    @staticmethod
    def _validate_search_space(
        generator: Callable,
        search_space: dict[str, SearchSpaceValue] | None
    ) -> None:
        """Validate that search_space parameters match generator signature and formats are correct.
        
        Args:
            generator: Generator function to validate against
            search_space: Enhanced search space to validate
            
        Raises:
            ValueError: If search_space contains invalid parameters or formats
        """
        if search_space is None:
            return

        # Get generator signature
        sig = inspect.signature(generator)
        generator_params = set(sig.parameters.keys())
        search_params = set(search_space.keys())

        # Check for parameters in search_space that don't exist in generator
        missing_params = search_params - generator_params
        if missing_params:
            raise ValueError(
                f"Parameters {missing_params} from search_space not found in generator signature. "
                f"Available parameters: {generator_params}"
            )

        # Validate each search space entry
        for param_name, param_spec in search_space.items():
            try:
                SvetlannaTrainer._validate_single_param_spec(param_name, param_spec)
            except ValueError as e:
                raise ValueError(f"Invalid search_space['{param_name}']: {e}")

    @staticmethod
    def _validate_single_param_spec(param_name: str, param_spec: SearchSpaceValue) -> None:
        """Validate a single parameter specification."""

        if isinstance(param_spec, dict):
            # Full specification format: {'type': '...', 'low': ..., 'high': ..., ...}
            SvetlannaTrainer._validate_dict_spec(param_name, param_spec)

        elif isinstance(param_spec, list):
            # Categorical format: ['choice1', 'choice2', ...]
            if len(param_spec) == 0:
                raise ValueError("categorical choices list cannot be empty")

        elif isinstance(param_spec, tuple):
            # Range format: (min, max) or (min, max, kwargs)
            if len(param_spec) == 2:
                min_val, max_val = param_spec
                if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                    raise ValueError(
                        f"tuple format values must be numeric, got ({type(min_val)}, {type(max_val)})"
                    )
                if min_val >= max_val:
                    raise ValueError(f"min value ({min_val}) must be < max value ({max_val})")

            elif len(param_spec) == 3:
                min_val, max_val, kwargs = param_spec
                if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                    raise ValueError(
                        f"tuple format values must be numeric, got ({type(min_val)}, {type(max_val)})"
                    )
                if min_val >= max_val:
                    raise ValueError(f"min value ({min_val}) must be < max value ({max_val})")
                if not isinstance(kwargs, dict):
                    raise ValueError(f"third tuple element must be dict, got {type(kwargs)}")
            else:
                raise ValueError(f"tuple format must have 2 or 3 elements, got {len(param_spec)}")
        else:
            raise ValueError(
                f"unsupported format {type(param_spec)}. "
                f"Use tuple (min, max), list [choices], or dict specification"
            )

    @staticmethod
    def _validate_dict_spec(param_name: str, spec: dict[str, Any]) -> None:
        """Validate dictionary specification format."""
        if "type" not in spec:
            raise ValueError("dict format must include 'type' field")

        param_type = spec["type"]

        if param_type == "int" or param_type == "float":
            required = ["low", "high"]
            optional = ["step", "log"]
        elif param_type == "categorical":
            required = ["choices"]
            optional = []
        else:
            raise ValueError(f"unsupported type '{param_type}'. Use 'int', 'float', or 'categorical'")

        # Check required fields
        missing = set(required) - set(spec.keys())
        if missing:
            raise ValueError(f"missing required fields for type '{param_type}': {missing}")

        # Validate ranges for numeric types
        if param_type in ["int", "float"]:
            low, high = spec["low"], spec["high"]
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                raise ValueError("'low' and 'high' must be numeric")
            if low >= high:
                raise ValueError(f"'low' ({low}) must be < 'high' ({high})")

        # Validate categorical choices
        elif param_type == "categorical":
            choices = spec["choices"]
            if not isinstance(choices, list) or len(choices) == 0:
                raise ValueError("'choices' must be non-empty list")

    @staticmethod
    def _create_optical_setup_with_defaults(generator: Callable) -> LinearOpticalSetup:
        """Create optical setup using generator with default parameter values.
        
        Args:
            generator: Generator function
            
        Returns:
            LinearOpticalSetup: Generated optical system
            
        Raises:
            ValueError: If generator has required parameters without defaults
        """
        sig = inspect.signature(generator)

        # Check if generator has parameters
        if not sig.parameters:
            # Generator has no parameters
            return generator()

        # Build kwargs using default values
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param.default == inspect.Parameter.empty:
                raise ValueError(
                    f"Generator parameter '{param_name}' has no default value. "
                    f"Either provide search_space for optimization or add default values to generator."
                )
            kwargs[param_name] = param.default

        return generator(**kwargs)

    def _setup_trainer(self) -> None:
        """Lazily create Lightning Trainer if not already created."""
        if self._lightning_trainer is None:
            self._lightning_trainer = LightningTrainer(**self._trainer_config)

    def fit(self) -> Any:
        """Train the model using simple training (no hyperparameter optimization).
        
        Returns:
            Training results from Lightning Trainer
            
        Raises:
            RuntimeError: If trainer was configured for optimization mode
        """
        if self._search_space is not None:
            raise RuntimeError(
                "This trainer was configured for hyperparameter optimization. "
                "Use optimize() method instead of fit()."
            )

        self._setup_trainer()
        return self._lightning_trainer.fit(self._lightning_module, self._datamodule)

    def optimize(self):
        """Run hyperparameter optimization using Optuna.
        
        Supports both single-objective and multi-objective optimization.
        
        Returns:
            optuna.Study: Completed optimization study
            
        Raises:
            RuntimeError: If trainer was not configured for optimization
            ImportError: If Optuna is not installed
        """
        if self._search_space is None:
            raise RuntimeError(
                "This trainer was not configured for hyperparameter optimization. "
                "Provide search_space parameter when creating trainer."
            )

        try:
            import optuna
        except ImportError:
            raise ImportError(
                "Optuna is required for hyperparameter optimization. "
                "Install it with: pip install optuna"
            )

        # Check if multi-objective optimization is requested
        if self._objectives is not None and len(self._objectives) >= 2:
            return self._run_multi_objective_optimization()
        return self._run_single_objective_optimization()

    def _run_single_objective_optimization(self):
        """Run traditional single-objective optimization."""
        import optuna

        # Create Optuna optimizer if not already created
        if self._optuna_optimizer is None:
            self._optuna_optimizer = OptunaOptimizer(
                generator=self._generator,
                datamodule=self._datamodule,
                search_space=self._search_space,
                loss_fn=self._loss_fn,
                optimizer_config=self._optimizer_config,
                trainer_config=self._trainer_config,
                metrics=self._metrics
            )

        # Create sampler, pruner, and storage
        sampler = self._create_sampler()
        pruner = self._create_pruner()
        storage = self._create_storage()

        # Run optimization
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            study_name=self._study_name,
            load_if_exists=True if storage is not None else False
        )
        study.optimize(
            self._optuna_optimizer.objective,
            n_trials=self._n_trials,
            callbacks=self._callbacks
        )

        # Update internal state with best configuration
        self._update_with_best_params(study.best_params)

        return study

    def _run_multi_objective_optimization(self):
        """Run multi-objective optimization with Pareto front."""
        # Create multi-objective optimizer
        mo_optimizer = MultiObjectiveOptimizer(
            objectives=self._objectives,
            approach=self._multi_objective_approach,
            n_trials=self._n_trials
        )

        # Prepare trainer components
        trainer_components = {
            "generator": self._generator,
            "datamodule": self._datamodule,
            "search_space": self._search_space,
            "loss_fn": self._loss_fn,
            "optimizer_config": self._optimizer_config,
            "trainer_config": self._trainer_config,
            "metrics": self._metrics,
            "sampler": self._sampler,
            "sampler_config": self._sampler_config
        }

        # Run multi-objective optimization
        study = mo_optimizer.optimize(trainer_components)

        # For multi-objective, update with compromise solution or user's choice
        if hasattr(study, "best_trials") and len(study.best_trials) > 0:
            # Use first Pareto optimal solution as default
            # In practice, user should choose based on their preferences
            best_trial = study.best_trials[0]
            self._update_with_best_params(best_trial.params)
        elif hasattr(study, "best_params"):
            # Weighted scalarization case
            self._update_with_best_params(study.best_params)

        return study

    def _create_sampler(self):
        """Create Optuna sampler based on configuration."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for sampler creation")

        if self._sampler is None:
            return None  # Use Optuna default (TPE)

        if isinstance(self._sampler, str):
            # Create sampler from string name
            return self._create_sampler_from_name(self._sampler, self._sampler_config)
        # Assume it's already a sampler instance
        return self._sampler

    def _create_sampler_from_name(self, sampler_name: str, config: dict[str, Any]):
        """Create sampler from string name and configuration."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for sampler creation")

        sampler_name = sampler_name.lower()

        if sampler_name == "tpe":
            return optuna.samplers.TPESampler(**config)
        if sampler_name == "random":
            return optuna.samplers.RandomSampler(**config)
        if sampler_name == "grid":
            return optuna.samplers.GridSampler(**config)
        if sampler_name == "cmaes":
            return optuna.samplers.CmaEsSampler(**config)
        if sampler_name == "nsgaii":
            return optuna.samplers.NSGAIISampler(**config)
        if sampler_name == "nsgaiii":
            return optuna.samplers.NSGAIIISampler(**config)
        if sampler_name == "partialfixed":
            return optuna.samplers.PartialFixedSampler(**config)
        raise ValueError(
            f"Unknown sampler: {sampler_name}. "
            f"Supported: 'tpe', 'random', 'grid', 'cmaes', 'nsgaii', 'nsgaiii', 'partialfixed'"
        )

    def _create_pruner(self):
        """Create Optuna pruner based on configuration."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for pruner creation")

        if self._pruner is None:
            return None  # No pruning

        if isinstance(self._pruner, str):
            # Create pruner from string name
            return self._create_pruner_from_name(self._pruner, self._pruner_config)
        # Assume it's already a pruner instance
        return self._pruner

    def _create_pruner_from_name(self, pruner_name: str, config: dict[str, Any]):
        """Create pruner from string name and configuration."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for pruner creation")

        pruner_name = pruner_name.lower()

        if pruner_name == "hyperband":
            return optuna.pruners.HyperbandPruner(**config)
        if pruner_name == "median":
            return optuna.pruners.MedianPruner(**config)
        if pruner_name == "successive_halving":
            return optuna.pruners.SuccessiveHalvingPruner(**config)
        if pruner_name == "percentile":
            # PercentilePruner requires percentile parameter
            if "percentile" not in config:
                config["percentile"] = 25.0  # Default to 25th percentile
            return optuna.pruners.PercentilePruner(**config)
        if pruner_name == "threshold":
            return optuna.pruners.ThresholdPruner(**config)
        if pruner_name == "nop":
            return optuna.pruners.NopPruner()
        raise ValueError(
            f"Unknown pruner: {pruner_name}. "
            f"Supported: 'hyperband', 'median', 'successive_halving', 'percentile', 'threshold', 'nop'"
        )

    def _create_storage(self):
        """Create Optuna storage based on configuration."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for storage creation")

        if self._storage is None:
            return None  # Use in-memory storage

        if isinstance(self._storage, str):
            # Create storage from URL string
            if self._storage.startswith("sqlite://") or self._storage.startswith("mysql://") or self._storage.startswith("postgresql://"):
                return optuna.storages.RDBStorage(url=self._storage)
            if self._storage.startswith("redis://"):
                return optuna.storages.JournalRedisStorage(url=self._storage)
            # Assume it's a file path for SQLite
            if not self._storage.endswith(".db"):
                self._storage += ".db"
            return optuna.storages.RDBStorage(url=f"sqlite:///{self._storage}")
        # Assume it's already a storage instance
        return self._storage

    def _update_with_best_params(self, best_params: dict[str, Any]) -> None:
        """Update trainer with best parameters found by optimization."""
        # Create optical setup with best parameters
        optical_setup = self._generator(**best_params)

        # Create new Lightning module with best setup
        self._lightning_module = LinearOpticalSetupLightning(
            optical_setup=optical_setup,
            loss_fn=self._loss_fn,
            optimizer_config=self._optimizer_config,
            metrics=self._metrics
        )

        # Reset Lightning trainer to use new module
        self._lightning_trainer = None

    def predict(self, datamodule: L.LightningDataModule | None = None) -> list[torch.Tensor]:
        """Generate predictions using the trained model.
        
        Args:
            datamodule: DataModule for prediction. If None, uses training datamodule
            
        Returns:
            List of prediction tensors
        """
        self._setup_trainer()
        predict_datamodule = datamodule or self._datamodule
        return self._lightning_trainer.predict(self._lightning_module, predict_datamodule)

    def test(self, datamodule: L.LightningDataModule | None = None) -> list[dict[str, Any]]:
        """Test the trained model.
        
        Args:
            datamodule: DataModule for testing. If None, uses training datamodule
            
        Returns:
            Test results
        """
        self._setup_trainer()
        test_datamodule = datamodule or self._datamodule
        return self._lightning_trainer.test(self._lightning_module, test_datamodule)

    # Component access methods
    def get_lightning_trainer(self) -> LightningTrainer:
        """Get the underlying Lightning Trainer."""
        self._setup_trainer()
        return self._lightning_trainer

    def get_lightning_module(self) -> LinearOpticalSetupLightning:
        """Get the Lightning module."""
        return self._lightning_module

    def get_optical_setup(self) -> LinearOpticalSetup:
        """Get the underlying optical setup."""
        return self._lightning_module.get_optical_setup()


class OptunaOptimizer:
    """Optuna optimizer for hyperparameter search.
    
    This class handles the objective function for Optuna optimization,
    creating and training models with different hyperparameter combinations.
    """

    def __init__(
        self,
        generator: Callable[..., LinearOpticalSetup],
        datamodule: L.LightningDataModule,
        search_space: dict[str, SearchSpaceValue],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer_config: dict[str, Any] | None = None,
        trainer_config: dict[str, Any] | None = None,
        metrics: list[Any] | None = None
    ):
        """Initialize Optuna optimizer with enhanced search space support.
        
        Args:
            generator: Function to generate optical setups
            datamodule: Lightning DataModule
            search_space: Enhanced parameter specifications for optimization
            loss_fn: Loss function
            optimizer_config: Optimizer configuration
            trainer_config: Lightning Trainer configuration
            metrics: List of metrics
        """
        self.generator = generator
        self.datamodule = datamodule
        self.search_space = search_space
        self.loss_fn = loss_fn
        self.optimizer_config = optimizer_config
        self.metrics = metrics

        # Configure trainer for optimization (short training for trials)
        self.trainer_config = trainer_config or {}
        if "max_epochs" not in self.trainer_config:
            self.trainer_config["max_epochs"] = 10  # Short training for trials
        if "enable_checkpointing" not in self.trainer_config:
            self.trainer_config["enable_checkpointing"] = False  # Disable for speed
        if "logger" not in self.trainer_config:
            self.trainer_config["logger"] = False  # Disable logging for speed

    def objective(self, trial) -> float:
        """Optuna objective function.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Metric value to minimize (validation loss)
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for hyperparameter optimization")

        # Sample parameters from enhanced search space
        params = {}
        for param_name, param_spec in self.search_space.items():
            params[param_name] = self._suggest_parameter(trial, param_name, param_spec)

        # Create optical setup with sampled parameters
        optical_setup = self.generator(**params)

        # Create Lightning module
        lightning_module = LinearOpticalSetupLightning(
            optical_setup=optical_setup,
            loss_fn=self.loss_fn,
            optimizer_config=self.optimizer_config,
            metrics=self.metrics
        )

        # Create trainer with pruning callback if needed
        trainer_config = self.trainer_config.copy()

        # Add pruning callback if trial supports it
        callbacks = trainer_config.get("callbacks", [])
        if hasattr(trial, "report") and hasattr(trial, "should_prune"):
            from lightning.pytorch.callbacks import Callback

            class OptunaPruningCallback(Callback):
                """Callback for Optuna pruning integration."""

                def __init__(self, trial, monitor="val_loss"):
                    self.trial = trial
                    self.monitor = monitor

                def on_validation_epoch_end(self, trainer, pl_module):
                    """Report intermediate value and check for pruning."""
                    epoch = trainer.current_epoch
                    current_score = trainer.callback_metrics.get(self.monitor)

                    if current_score is not None:
                        # Report intermediate value to Optuna
                        self.trial.report(current_score.item(), epoch)

                        # Check if trial should be pruned
                        if self.trial.should_prune():
                            message = f"Trial was pruned at epoch {epoch}."
                            raise optuna.exceptions.TrialPruned(message)

            callbacks.append(OptunaPruningCallback(trial))
            trainer_config["callbacks"] = callbacks

        trainer = LightningTrainer(**trainer_config)

        # Train model (may be pruned early)
        try:
            trainer.fit(lightning_module, self.datamodule)
        except optuna.exceptions.TrialPruned:
            # Re-raise pruning exception to let Optuna handle it
            raise

        # Return validation loss for minimization
        return trainer.callback_metrics.get("val_loss", float("inf")).item()

    def _suggest_parameter(self, trial, param_name: str, param_spec: SearchSpaceValue) -> Any:
        """Suggest parameter value using appropriate Optuna suggest method.
        
        Args:
            trial: Optuna trial object
            param_name: Parameter name
            param_spec: Parameter specification
            
        Returns:
            Suggested parameter value
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for hyperparameter optimization")

        if isinstance(param_spec, dict):
            # Full specification format
            return self._suggest_from_dict_spec(trial, param_name, param_spec)

        if isinstance(param_spec, list):
            # Categorical format: suggest_categorical
            return trial.suggest_categorical(param_name, param_spec)

        if isinstance(param_spec, tuple):
            if len(param_spec) == 2:
                # Simple range format: (min, max)
                min_val, max_val = param_spec
                if isinstance(min_val, int) and isinstance(max_val, int):
                    return trial.suggest_int(param_name, min_val, max_val)
                return trial.suggest_float(param_name, min_val, max_val)

            if len(param_spec) == 3:
                # Range with kwargs: (min, max, kwargs)
                min_val, max_val, kwargs = param_spec
                if isinstance(min_val, int) and isinstance(max_val, int):
                    return trial.suggest_int(param_name, min_val, max_val, **kwargs)
                return trial.suggest_float(param_name, min_val, max_val, **kwargs)

        raise ValueError(f"Invalid parameter specification for {param_name}: {param_spec}")

    def _suggest_from_dict_spec(self, trial, param_name: str, spec: dict[str, Any]) -> Any:
        """Suggest parameter from dictionary specification."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for hyperparameter optimization")

        param_type = spec["type"]

        if param_type == "int":
            # Extract parameters
            low = spec["low"]
            high = spec["high"]
            step = spec.get("step", 1)
            log = spec.get("log", False)

            return trial.suggest_int(param_name, low, high, step=step, log=log)

        if param_type == "float":
            # Extract parameters
            low = spec["low"]
            high = spec["high"]
            step = spec.get("step")
            log = spec.get("log", False)

            if step is not None:
                return trial.suggest_float(param_name, low, high, step=step, log=log)
            return trial.suggest_float(param_name, low, high, log=log)

        if param_type == "categorical":
            choices = spec["choices"]
            return trial.suggest_categorical(param_name, choices)

        raise ValueError(f"Unsupported parameter type: {param_type}")
