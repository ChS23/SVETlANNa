"""SVETlANNa Training Module

High-level training interfaces for optical neural networks with automatic
hyperparameter optimization support.

Main Components:
- SvetlannaTrainer: High-level trainer with two-level API
- LinearOpticalSetupLightning: Lightning adapter for optical systems
- Generators: Factory functions for common optical architectures

Examples:
    Simple training:
        >>> from svetlanna.training import SvetlannaTrainer
        >>> from svetlanna.training.generators.d2nn import create_d2nn_generator
        >>> 
        >>> trainer = SvetlannaTrainer.from_generator(
        ...     generator=create_d2nn_generator(sim_params),
        ...     datamodule=my_datamodule
        ... )
        >>> results = trainer.fit()
    
    Hyperparameter optimization:
        >>> trainer = SvetlannaTrainer.from_generator(
        ...     generator=create_d2nn_generator(sim_params),
        ...     datamodule=my_datamodule,
        ...     search_space={
        ...         'n_layers': (3, 10),
        ...         'layer_distance': (0.05, 0.2)
        ...     }
        ... )
        >>> study = trainer.optimize()
"""

from .lightning_adapter import LinearOpticalSetupLightning
from .loggers import COMMON_LOGGER_CONFIGS, create_multiple_loggers, get_logger_for_setup, LoggerConfig
from .multi_objective import (
    AccuracyObjective,
    analyze_pareto_front,
    COMMON_OBJECTIVE_SETS,
    FabricationComplexityObjective,
    MultiObjectiveOptimizer,
    ObjectiveFunction,
    OpticalEfficiencyObjective,
    PhysicalSizeObjective,
    plot_pareto_front,
)
from .trainer import SvetlannaTrainer

__all__ = [
    "COMMON_LOGGER_CONFIGS",
    "COMMON_OBJECTIVE_SETS",
    "AccuracyObjective",
    "FabricationComplexityObjective",
    "LinearOpticalSetupLightning",
    "LoggerConfig",
    "MultiObjectiveOptimizer",
    "ObjectiveFunction",
    "OpticalEfficiencyObjective",
    "PhysicalSizeObjective",
    "SvetlannaTrainer",
    "analyze_pareto_front",
    "create_multiple_loggers",
    "get_logger_for_setup",
    "plot_pareto_front"
]
