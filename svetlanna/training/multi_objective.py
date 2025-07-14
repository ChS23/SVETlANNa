"""Multi-objective optimization support for SvetlannaTrainer.

Implements Pareto optimization for optical neural networks where multiple
conflicting objectives need to be optimized simultaneously.

Common optical objectives:
- Accuracy (classification/regression performance)
- Optical efficiency (energy transmission)
- Fabrication complexity (number of elements, precision requirements)
- Physical size (footprint, thickness)
- Wavelength sensitivity (broadband vs narrowband)
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import lightning as L
import torch

from svetlanna.setup import LinearOpticalSetup


class ObjectiveFunction(ABC):
    """Abstract base class for objective functions in multi-objective optimization.
    
    Each objective function computes a single scalar metric that should be
    minimized (lower is better). If you have a metric where higher is better,
    return its negative value.
    """

    def __init__(self, name: str, weight: float = 1.0):
        """Initialize objective function.
        
        Args:
            name: Human-readable name for this objective
            weight: Relative importance weight (used for scalarization)
        """
        self.name = name
        self.weight = weight

    @abstractmethod
    def compute(
        self,
        optical_setup: LinearOpticalSetup,
        lightning_module: Any,
        trainer_results: dict[str, Any],
        trial_params: dict[str, Any]
    ) -> float:
        """Compute objective value for a trained model.
        
        Args:
            optical_setup: The optical system
            lightning_module: Trained Lightning module
            trainer_results: Results from Lightning trainer
            trial_params: Hyperparameters used in this trial
            
        Returns:
            Objective value (lower is better)
        """

    def __call__(self, *args, **kwargs) -> float:
        """Allow calling instance as function."""
        return self.compute(*args, **kwargs)


class AccuracyObjective(ObjectiveFunction):
    """Minimize validation loss (maximize accuracy)."""

    def __init__(self, metric_name: str = "val_loss"):
        super().__init__(name="Accuracy", weight=1.0)
        self.metric_name = metric_name

    def compute(
        self,
        optical_setup: LinearOpticalSetup,
        lightning_module: Any,
        trainer_results: dict[str, Any],
        trial_params: dict[str, Any]
    ) -> float:
        """Return validation loss (lower is better)."""
        if hasattr(trainer_results, "callback_metrics"):
            return trainer_results.callback_metrics.get(self.metric_name, float("inf"))
        return float("inf")


class OpticalEfficiencyObjective(ObjectiveFunction):
    """Minimize optical loss (maximize transmission efficiency)."""

    def __init__(self):
        super().__init__(name="Optical Efficiency", weight=1.0)

    def compute(
        self,
        optical_setup: LinearOpticalSetup,
        lightning_module: Any,
        trainer_results: dict[str, Any],
        trial_params: dict[str, Any]
    ) -> float:
        """Compute optical inefficiency (1 - efficiency)."""
        # Simple approximation: more layers = more loss
        n_layers = trial_params.get("n_layers", 5)
        layer_distance = trial_params.get("layer_distance", 0.1)

        # Model optical losses
        fresnel_losses = n_layers * 0.04  # ~4% per interface
        propagation_losses = layer_distance * n_layers * 0.1  # Distance-dependent

        total_loss = fresnel_losses + propagation_losses
        return min(total_loss, 1.0)  # Cap at 100% loss


class FabricationComplexityObjective(ObjectiveFunction):
    """Minimize fabrication complexity."""

    def __init__(self):
        super().__init__(name="Fabrication Complexity", weight=1.0)

    def compute(
        self,
        optical_setup: LinearOpticalSetup,
        lightning_module: Any,
        trainer_results: dict[str, Any],
        trial_params: dict[str, Any]
    ) -> float:
        """Compute normalized fabrication complexity."""
        n_layers = trial_params.get("n_layers", 5)

        # Complexity factors
        layer_complexity = n_layers / 10.0  # Normalize by max expected layers

        # Phase mask complexity (variance in phase patterns)
        phase_complexity = 0.0
        for name, param in optical_setup.named_parameters():
            if "mask" in name and param.requires_grad:
                # High variance = more complex to fabricate
                phase_var = torch.var(param).item()
                phase_complexity += phase_var

        phase_complexity = min(phase_complexity / (n_layers * 10.0), 1.0)  # Normalize

        return layer_complexity + phase_complexity


class PhysicalSizeObjective(ObjectiveFunction):
    """Minimize physical system size."""

    def __init__(self):
        super().__init__(name="Physical Size", weight=1.0)

    def compute(
        self,
        optical_setup: LinearOpticalSetup,
        lightning_module: Any,
        trainer_results: dict[str, Any],
        trial_params: dict[str, Any]
    ) -> float:
        """Compute normalized physical size."""
        n_layers = trial_params.get("n_layers", 5)
        layer_distance = trial_params.get("layer_distance", 0.1)
        detector_distance = trial_params.get("detector_distance", 0.05)

        # Total system length
        total_length = (n_layers - 1) * layer_distance + detector_distance

        # Normalize by reasonable maximum (1 meter)
        return min(total_length / 1.0, 1.0)


class MultiObjectiveOptimizer:
    """Multi-objective optimizer using NSGA-II-like approach with Optuna.
    
    Supports both Pareto optimization and weighted scalarization approaches.
    """

    def __init__(
        self,
        objectives: list[ObjectiveFunction],
        approach: str = "pareto",
        n_trials: int = 100
    ):
        """Initialize multi-objective optimizer.
        
        Args:
            objectives: List of objective functions to optimize
            approach: 'pareto' for Pareto optimization, 'weighted' for scalarization
            n_trials: Number of optimization trials
        """
        self.objectives = objectives
        self.approach = approach
        self.n_trials = n_trials

        if len(objectives) < 2:
            raise ValueError("Multi-objective optimization requires at least 2 objectives")

    def create_objective_function(self, trainer_components: dict[str, Any]) -> Callable:
        """Create objective function for Optuna study.
        
        Args:
            trainer_components: dict with generator, datamodule, configs, etc.
            
        Returns:
            Objective function for Optuna
        """
        def objective(trial):
            try:
                return self._evaluate_trial(trial, trainer_components)
            except Exception as e:
                warnings.warn(f"Trial failed: {e}")
                # Return worst possible values for all objectives
                if self.approach == "pareto":
                    return [float("inf")] * len(self.objectives)
                return float("inf")

        return objective

    def _evaluate_trial(self, trial, trainer_components: dict[str, Any]):
        """Evaluate a single trial for all objectives."""

        # Sample parameters from search space
        params = {}
        for param_name, param_spec in trainer_components["search_space"].items():
            params[param_name] = self._suggest_parameter(trial, param_name, param_spec)

        # Create optical setup with sampled parameters
        optical_setup = trainer_components["generator"](**params)

        # Create and train Lightning module
        from .lightning_adapter import LinearOpticalSetupLightning
        lightning_module = LinearOpticalSetupLightning(
            optical_setup=optical_setup,
            loss_fn=trainer_components["loss_fn"],
            optimizer_config=trainer_components["optimizer_config"],
            metrics=trainer_components["metrics"]
        )

        # Short training for evaluation
        trainer = L.Trainer(**trainer_components["trainer_config"])
        trainer.fit(lightning_module, trainer_components["datamodule"])

        # Compute all objectives
        objective_values = []
        for obj_func in self.objectives:
            value = obj_func.compute(
                optical_setup=optical_setup,
                lightning_module=lightning_module,
                trainer_results=trainer,
                trial_params=params
            )
            objective_values.append(value)

        # Return based on approach
        if self.approach == "pareto":
            return objective_values  # Multi-objective
        # Weighted scalarization
        weighted_sum = sum(
            obj.weight * val
            for obj, val in zip(self.objectives, objective_values, strict=False)
        )
        return weighted_sum

    def _suggest_parameter(self, trial, param_name: str, param_spec) -> Any:
        """Suggest parameter using Optuna (simplified version)."""
        if isinstance(param_spec, tuple) and len(param_spec) == 2:
            min_val, max_val = param_spec
            if isinstance(min_val, int) and isinstance(max_val, int):
                return trial.suggest_int(param_name, min_val, max_val)
            return trial.suggest_float(param_name, min_val, max_val)
        if isinstance(param_spec, list):
            return trial.suggest_categorical(param_name, param_spec)
        raise ValueError(f"Unsupported param_spec format: {param_spec}")

    def optimize(self, trainer_components: dict[str, Any]):
        """Run multi-objective optimization.
        
        Args:
            trainer_components: Components needed for training
            
        Returns:
            Optuna study with results
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for multi-objective optimization")

        # Create sampler for multi-objective optimization
        sampler = self._create_multi_objective_sampler(trainer_components)

        # Create pruner and storage from trainer components
        pruner = self._create_multi_objective_pruner(trainer_components)
        storage = self._create_multi_objective_storage(trainer_components)
        study_name = trainer_components.get("study_name") or (
            "multi_objective_optical_optimization" if self.approach == "pareto"
            else "weighted_optical_optimization"
        )

        # Create appropriate study
        if self.approach == "pareto":
            # Multi-objective study
            study = optuna.create_study(
                directions=["minimize"] * len(self.objectives),
                study_name=study_name,
                sampler=sampler,
                pruner=pruner,
                storage=storage,
                load_if_exists=True if storage is not None else False
            )
        else:
            # Single-objective study (weighted)
            study = optuna.create_study(
                direction="minimize",
                study_name=study_name,
                sampler=sampler,
                pruner=pruner,
                storage=storage,
                load_if_exists=True if storage is not None else False
            )

        # Run optimization
        objective_func = self.create_objective_function(trainer_components)
        callbacks = trainer_components.get("callbacks", [])
        study.optimize(objective_func, n_trials=self.n_trials, callbacks=callbacks)

        return study

    def _create_multi_objective_sampler(self, trainer_components: dict[str, Any]):
        """Create appropriate sampler for multi-objective optimization."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for sampler creation")

        # Get sampler configuration from trainer components
        sampler_name = trainer_components.get("sampler")
        sampler_config = trainer_components.get("sampler_config", {})

        if sampler_name is None:
            # Use appropriate default based on approach
            if self.approach == "pareto":
                return optuna.samplers.NSGAIISampler(**sampler_config)
            return optuna.samplers.TPESampler(**sampler_config)

        if isinstance(sampler_name, str):
            sampler_name = sampler_name.lower()

            # For multi-objective, some samplers are more appropriate
            if self.approach == "pareto":
                if sampler_name in ["nsgaii", "nsgaiii"]:
                    if sampler_name == "nsgaii":
                        return optuna.samplers.NSGAIISampler(**sampler_config)
                    if sampler_name == "nsgaiii":
                        return optuna.samplers.NSGAIIISampler(**sampler_config)
                else:
                    # For Pareto optimization, use NSGA-II as fallback
                    import warnings
                    warnings.warn(
                        f"Sampler '{sampler_name}' may not be optimal for multi-objective optimization. "
                        f"Consider using 'nsgaii' or 'nsgaiii' for Pareto optimization."
                    )
                    return self._create_single_objective_sampler(sampler_name, sampler_config)
            else:
                # Weighted scalarization can use any single-objective sampler
                return self._create_single_objective_sampler(sampler_name, sampler_config)
        else:
            # Assume it's already a sampler instance
            return sampler_name

    def _create_single_objective_sampler(self, sampler_name: str, config: dict[str, Any]):
        """Create single-objective sampler."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for sampler creation")

        if sampler_name == "tpe":
            return optuna.samplers.TPESampler(**config)
        if sampler_name == "random":
            return optuna.samplers.RandomSampler(**config)
        if sampler_name == "grid":
            return optuna.samplers.GridSampler(**config)
        if sampler_name == "cmaes":
            return optuna.samplers.CmaEsSampler(**config)
        raise ValueError(f"Unknown single-objective sampler: {sampler_name}")

    def _create_multi_objective_pruner(self, trainer_components: dict[str, Any]):
        """Create pruner for multi-objective optimization."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for pruner creation")

        pruner_name = trainer_components.get("pruner")
        pruner_config = trainer_components.get("pruner_config", {})

        if pruner_name is None:
            return None  # No pruning

        if isinstance(pruner_name, str):
            pruner_name = pruner_name.lower()

            if pruner_name == "hyperband":
                return optuna.pruners.HyperbandPruner(**pruner_config)
            if pruner_name == "median":
                return optuna.pruners.MedianPruner(**pruner_config)
            if pruner_name == "successive_halving":
                return optuna.pruners.SuccessiveHalvingPruner(**pruner_config)
            if pruner_name == "percentile":
                # PercentilePruner requires percentile parameter
                if "percentile" not in pruner_config:
                    pruner_config["percentile"] = 25.0  # Default to 25th percentile
                return optuna.pruners.PercentilePruner(**pruner_config)
            if pruner_name == "threshold":
                return optuna.pruners.ThresholdPruner(**pruner_config)
            if pruner_name == "nop":
                return optuna.pruners.NopPruner()
            raise ValueError(f"Unknown pruner: {pruner_name}")
        # Assume it's already a pruner instance
        return pruner_name

    def _create_multi_objective_storage(self, trainer_components: dict[str, Any]):
        """Create storage for multi-objective optimization."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for storage creation")

        storage = trainer_components.get("storage")

        if storage is None:
            return None  # Use in-memory storage

        if isinstance(storage, str):
            # Create storage from URL string
            if storage.startswith("sqlite://") or storage.startswith("mysql://") or storage.startswith("postgresql://"):
                return optuna.storages.RDBStorage(url=storage)
            if storage.startswith("redis://"):
                return optuna.storages.JournalRedisStorage(url=storage)
            # Assume it's a file path for SQLite
            if not storage.endswith(".db"):
                storage += ".db"
            return optuna.storages.RDBStorage(url=f"sqlite:///{storage}")
        # Assume it's already a storage instance
        return storage


def analyze_pareto_front(study, objective_names: list[str]) -> dict[str, Any]:
    """Analyze Pareto front from multi-objective study.
    
    Args:
        study: Completed Optuna multi-objective study
        objective_names: Names of objectives for reporting
        
    Returns:
        Analysis results including Pareto front points
    """
    if not hasattr(study, "best_trials"):
        raise ValueError("Study must be multi-objective to analyze Pareto front")

    pareto_trials = study.best_trials

    analysis = {
        "n_pareto_points": len(pareto_trials),
        "pareto_trials": pareto_trials,
        "objective_names": objective_names,
        "pareto_front": [],
        "hypervolume": None  # Could implement hypervolume calculation
    }

    # Extract Pareto front points
    for trial in pareto_trials:
        point = {
            "trial_number": trial.number,
            "params": trial.params,
            "objectives": {
                name: value
                for name, value in zip(objective_names, trial.values, strict=False)
            }
        }
        analysis["pareto_front"].append(point)

    return analysis


def plot_pareto_front(analysis: dict[str, Any], save_path: str | None = None):
    """Plot Pareto front for 2D or 3D objectives.
    
    Args:
        analysis: Results from analyze_pareto_front()
        save_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError("Matplotlib is required for plotting Pareto front")

    n_objectives = len(analysis["objective_names"])

    if n_objectives == 2:
        # 2D Pareto front
        fig, ax = plt.subplots(figsize=(10, 8))

        x_values = [point["objectives"][analysis["objective_names"][0]]
                   for point in analysis["pareto_front"]]
        y_values = [point["objectives"][analysis["objective_names"][1]]
                   for point in analysis["pareto_front"]]

        ax.scatter(x_values, y_values, c="red", s=50, alpha=0.7, label="Pareto Front")
        ax.set_xlabel(analysis["objective_names"][0])
        ax.set_ylabel(analysis["objective_names"][1])
        ax.set_title("Pareto Front - Multi-Objective Optimization")
        ax.grid(True, alpha=0.3)
        ax.legend()

    elif n_objectives == 3:
        # 3D Pareto front
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        x_values = [point["objectives"][analysis["objective_names"][0]]
                   for point in analysis["pareto_front"]]
        y_values = [point["objectives"][analysis["objective_names"][1]]
                   for point in analysis["pareto_front"]]
        z_values = [point["objectives"][analysis["objective_names"][2]]
                   for point in analysis["pareto_front"]]

        ax.scatter(x_values, y_values, z_values, c="red", s=50, alpha=0.7)
        ax.set_xlabel(analysis["objective_names"][0])
        ax.set_ylabel(analysis["objective_names"][1])
        ax.set_zlabel(analysis["objective_names"][2])
        ax.set_title("3D Pareto Front - Multi-Objective Optimization")

    else:
        # Parallel coordinates plot for >3 objectives
        fig, ax = plt.subplots(figsize=(12, 8))

        # Normalize objectives to [0, 1] for better visualization
        all_values = np.array([[point["objectives"][name] for name in analysis["objective_names"]]
                              for point in analysis["pareto_front"]])

        normalized_values = (all_values - all_values.min(axis=0)) / (all_values.max(axis=0) - all_values.min(axis=0))

        for i, values in enumerate(normalized_values):
            ax.plot(range(n_objectives), values, "o-", alpha=0.7, label=f"Solution {i+1}")

        ax.set_xticks(range(n_objectives))
        ax.set_xticklabels(analysis["objective_names"], rotation=45)
        ax.set_ylabel("Normalized Objective Value")
        ax.set_title("Pareto Front - Parallel Coordinates")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# Predefined objective combinations for common optical scenarios
COMMON_OBJECTIVE_SETS = {
    "accuracy_efficiency": [
        AccuracyObjective(),
        OpticalEfficiencyObjective()
    ],

    "accuracy_size": [
        AccuracyObjective(),
        PhysicalSizeObjective()
    ],

    "accuracy_fabrication": [
        AccuracyObjective(),
        FabricationComplexityObjective()
    ],

    "full_optimization": [
        AccuracyObjective(),
        OpticalEfficiencyObjective(),
        FabricationComplexityObjective(),
        PhysicalSizeObjective()
    ]
}
