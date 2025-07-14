"""Optical Metrics

Custom metrics for optical neural networks and svetlanna components.
Handles device transitions, DetectorProcessor integration, and physics-aware measurements.

Available metrics:
- OpticalAccuracy: Classification accuracy with DetectorProcessor
- OpticalEfficiency: Energy efficiency measurements  
- FocusingMetric: Beam focusing quality evaluation

Example:
    >>> from svetlanna.training.metrics import OpticalAccuracy
    >>> metric = OpticalAccuracy(detector_processor, num_classes=10)
    >>> trainer = SvetlannaTrainer.from_generator(
    ...     generator=generator,
    ...     datamodule=datamodule, 
    ...     metrics=[metric]
    ... )
"""

# Import optical metrics
from .optical_accuracy import OpticalAccuracy, create_optical_accuracy

# Placeholder for future implementations
# from .optical_efficiency import OpticalEfficiency  
# from .focusing_metric import FocusingMetric

__all__ = [
    "OpticalAccuracy",
    "create_optical_accuracy",
    # "OpticalEfficiency", 
    # "FocusingMetric"
]