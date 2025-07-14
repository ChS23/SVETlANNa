from typing import Literal

import svetlanna.elements as el
import torch
from svetlanna.detector import Detector
from svetlanna.parameters import ConstrainedParameter
from svetlanna.setup import LinearOpticalSetup
from svetlanna.simulation_parameters import SimulationParameters


def d2nn(
    simulation_parameters: SimulationParameters,
    n_layers: int,
    layer_distance: float,
    detector_distance: float,
    phase_masks: list[torch.Tensor] | None = None,
    default_mask: torch.Tensor | None = None,
    mask_norm: float = 2 * torch.pi,
    freespace_method: Literal["fresnel", "AS"] = "AS",
    detector_func: str = "intensity"
) -> LinearOpticalSetup:
    """Generate a Diffractive Deep Neural Network (D2NN) architecture.
    
    Creates a sequential stack of diffractive layers separated by free space propagation:
    FreeSpace → [DiffractiveLayer → FreeSpace] × (n_layers-1) → DiffractiveLayer → FreeSpace → Detector
    
    Based on: Lin et al., "All-optical machine learning using diffractive deep neural networks"
    Science 361, 1004-1008 (2018)
    
    Args:
        simulation_parameters: Simulation grid and physical parameters
        n_layers: Number of diffractive layers (must be >= 1)
        layer_distance: Distance between consecutive diffractive layers
        detector_distance: Distance from last diffractive layer to detector
        phase_masks: List of phase masks for each layer. If None, uses default_mask for all layers
        default_mask: Default phase mask to use when phase_masks is None. If None, creates random masks
        mask_norm: Normalization factor for phase masks (typically 2π)
        freespace_method: Propagation method ('fresnel' or 'AS')
        detector_func: Detector function ('intensity', 'amplitude', etc.)
        
    Returns:
        LinearOpticalSetup: Complete D2NN optical system
        
    Raises:
        ValueError: If n_layers < 1, distances <= 0, or mask dimensions mismatch
        TypeError: If phase_masks elements are not tensors
        
    Example:
        >>> sim_params = SimulationParameters.from_ranges(
        ...     W=(-1e-3, 1e-3, 128), H=(-1e-3, 1e-3, 128), wavelength=532e-9
        ... )
        >>> system = d2nn(
        ...     simulation_parameters=sim_params,
        ...     n_layers=5,
        ...     layer_distance=0.1,
        ...     detector_distance=0.05
        ... )
    """
    # Input validation
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")

    if layer_distance <= 0:
        raise ValueError(f"layer_distance must be > 0, got {layer_distance}")

    if detector_distance <= 0:
        raise ValueError(f"detector_distance must be > 0, got {detector_distance}")

    if mask_norm <= 0:
        raise ValueError(f"mask_norm must be > 0, got {mask_norm}")

    # Get spatial dimensions from simulation parameters
    try:
        spatial_shape = simulation_parameters.axes_size(axs=("H", "W"))
    except KeyError as e:
        raise ValueError(f"SimulationParameters must contain 'H' and 'W' axes: {e}")

    # Handle phase masks
    if phase_masks is None:
        if default_mask is None:
            # Create random trainable phase masks using ConstrainedParameter
            phase_masks = [
                ConstrainedParameter(
                    data=torch.rand(spatial_shape, dtype=torch.float32) * mask_norm,
                    min_value=0.0,
                    max_value=mask_norm
                )
                for _ in range(n_layers)
            ]
        else:
            # Validate default mask shape
            if default_mask.shape != spatial_shape:
                raise ValueError(
                    f"default_mask shape {default_mask.shape} doesn't match "
                    f"simulation grid shape {spatial_shape}"
                )
            # Make default masks trainable using ConstrainedParameter
            phase_masks = [
                ConstrainedParameter(
                    data=default_mask.clone(),
                    min_value=0.0,
                    max_value=mask_norm
                )
                for _ in range(n_layers)
            ]
    else:
        # Validate provided phase masks
        if len(phase_masks) != n_layers:
            raise ValueError(
                f"Length of phase_masks ({len(phase_masks)}) must equal n_layers ({n_layers})"
            )

        for i, mask in enumerate(phase_masks):
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"phase_masks[{i}] must be torch.Tensor, got {type(mask)}")

            if mask.shape != spatial_shape:
                raise ValueError(
                    f"phase_masks[{i}] shape {mask.shape} doesn't match "
                    f"simulation grid shape {spatial_shape}"
                )

    # Build element list
    elements = []

    # Initial free space propagation
    elements.append(el.FreeSpace(
        simulation_parameters=simulation_parameters,
        distance=layer_distance,
        method=freespace_method
    ))

    # Diffractive layers with intermediate free space
    for i in range(n_layers):
        # Add diffractive layer
        elements.append(el.DiffractiveLayer(
            simulation_parameters=simulation_parameters,
            mask=phase_masks[i],
            mask_norm=mask_norm
        ))

        # Add free space after each layer except the last
        if i < n_layers - 1:
            elements.append(el.FreeSpace(
                simulation_parameters=simulation_parameters,
                distance=layer_distance,
                method=freespace_method
            ))

    # Final propagation to detector
    elements.append(el.FreeSpace(
        simulation_parameters=simulation_parameters,
        distance=detector_distance,
        method=freespace_method
    ))

    # Detector
    elements.append(Detector(
        simulation_parameters=simulation_parameters,
        func=detector_func
    ))

    return LinearOpticalSetup(
        simulation_parameters=simulation_parameters,
        elements=elements
    )


def create_d2nn_generator(simulation_parameters: SimulationParameters):
    """Create a D2NN generator function with fixed simulation parameters.
    
    This factory function creates a generator that can be used with hyperparameter
    optimization tools like Optuna, where only the architectural parameters
    (n_layers, distances, etc.) are varied while simulation_parameters remain fixed.
    
    Args:
        simulation_parameters: Fixed simulation grid and physical parameters
        
    Returns:
        Callable: Generator function that accepts only hyperparameters
        
    Example:
        >>> sim_params = SimulationParameters.from_ranges(
        ...     W=(-1e-3, 1e-3, 128), H=(-1e-3, 1e-3, 128), wavelength=532e-9
        ... )
        >>> generator = create_d2nn_generator(sim_params)
        >>> 
        >>> # Now generator only needs hyperparameters
        >>> system = generator(n_layers=5, layer_distance=0.1, detector_distance=0.05)
        >>> 
        >>> # Perfect for Optuna optimization
        >>> trainer = SvetlannaTrainer.from_optical_generator(generator)
    """
    def d2nn_generator(
        n_layers: int = 5,
        layer_distance: float = 0.1,
        detector_distance: float = 0.05,
        mask_norm: float = 2 * torch.pi,
        freespace_method: Literal["fresnel", "AS"] = "AS",
        detector_func: str = "intensity"
    ) -> LinearOpticalSetup:
        """Generate D2NN with hyperparameters only.
        
        Args:
            n_layers: Number of diffractive layers
            layer_distance: Distance between consecutive layers
            detector_distance: Distance from last layer to detector
            mask_norm: Phase mask normalization factor
            freespace_method: Propagation method
            detector_func: Detector function type
            
        Returns:
            LinearOpticalSetup: Complete D2NN system
        """
        return d2nn(
            simulation_parameters=simulation_parameters,
            n_layers=n_layers,
            layer_distance=layer_distance,
            detector_distance=detector_distance,
            mask_norm=mask_norm,
            freespace_method=freespace_method,
            detector_func=detector_func
        )

    return d2nn_generator
