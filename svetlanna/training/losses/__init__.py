"""Optical Loss Functions

Custom loss functions for optical neural networks and physics-informed training.
Incorporates optical physics constraints and specialized optimization objectives.

Available loss functions:
- OpticalMSELoss: MSE with optical physics constraints
- FocusingLoss: Beam focusing optimization
- EfficiencyLoss: Energy efficiency maximization
- PhaseRetrievalLoss: Phase recovery problems
- ZoneAmplificationLoss: Detector zone optimization

Example:
    >>> from svetlanna.training.losses import OpticalMSELoss
    >>> loss_fn = OpticalMSELoss(normalize_detector=True)
    >>> trainer = SvetlannaTrainer.from_generator(
    ...     generator=generator,
    ...     datamodule=datamodule,
    ...     loss_fn=loss_fn
    ... )
"""

# Placeholder for future implementations
# from .optical_mse import OpticalMSELoss
# from .focusing_loss import FocusingLoss
# from .efficiency_loss import EfficiencyLoss
# from .phase_retrieval_loss import PhaseRetrievalLoss
from .zone_amplification import ZoneAmplificationLoss

__all__ = [
    # "OpticalMSELoss",
    # "FocusingLoss", 
    # "EfficiencyLoss",
    # "PhaseRetrievalLoss",
    "ZoneAmplificationLoss"
]