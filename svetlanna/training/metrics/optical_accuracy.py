"""OpticalAccuracy metric for svetlanna training module.

Provides physics-aware accuracy computation using DetectorProcessor for classification
tasks with optical neural networks. Handles automatic device transitions and integrates
seamlessly with PyTorch Lightning training loops.
"""

import torch
import torch.nn as nn
import warnings

try:
    import torchmetrics
    from torchmetrics import Metric
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    # Fallback base class if torchmetrics not available
    class Metric(nn.Module):
        def __init__(self):
            super().__init__()
            self.add_state = self._add_state
            
        def _add_state(self, name: str, default: torch.Tensor, dist_reduce_fx: str = "sum"):
            self.register_buffer(name, default)
            
        def reset(self):
            for name, buffer in self.named_buffers():
                buffer.zero_()
                
        def compute(self):
            raise NotImplementedError
            
        def update(self, *args, **kwargs):
            raise NotImplementedError
    TORCHMETRICS_AVAILABLE = False

from svetlanna.detector import DetectorProcessorClf


class OpticalAccuracy(Metric):
    """Optical accuracy metric using DetectorProcessor for classification.
    
    This metric computes classification accuracy using the physics-aware 
    DetectorProcessor, which segments the detector plane into class regions
    and computes probabilities based on optical intensity integrals.
    
    Key features:
    - Automatic device handling (GPU ↔ CPU transitions)
    - Integration with Lightning training loops
    - Support for both detector images and class labels as targets
    - Batch processing for efficiency
    - State accumulation across batches
    
    Args:
        detector_processor: DetectorProcessorClf instance for classification
        task: Either "binary", "multiclass", or "multilabel" (default: "multiclass")
        num_classes: Number of classes (inferred from detector_processor if not provided)
        threshold: Threshold for binary/multilabel classification (default: 0.5)
        top_k: Compute top-k accuracy (default: 1)
        average: Averaging method - "micro", "macro", "weighted", or None (default: "micro")
        
    Example:
        >>> from svetlanna.detector import DetectorProcessorClf
        >>> from svetlanna.training.metrics import OpticalAccuracy
        >>> 
        >>> # Create detector processor
        >>> detector_processor = DetectorProcessorClf(
        ...     num_classes=10,
        ...     simulation_parameters=sim_params,
        ...     segmented_detector=detector_mask
        ... )
        >>> 
        >>> # Create metric
        >>> optical_acc = OpticalAccuracy(detector_processor)
        >>> 
        >>> # Use in training
        >>> preds = optical_system(inputs)  # (batch, 1, H, W) detector images
        >>> targets = torch.randint(0, 10, (batch,))  # Class labels
        >>> optical_acc.update(preds, targets)
        >>> accuracy = optical_acc.compute()
    """
    
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    
    def __init__(
        self,
        detector_processor: DetectorProcessorClf,
        task: str = "multiclass", 
        num_classes: int | None = None,
        threshold: float = 0.5,
        top_k: int = 1,
        average: str | None = "micro",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.detector_processor = detector_processor
        self.task = task.lower()
        self.threshold = threshold
        self.top_k = top_k
        self.average = average
        
        # Infer num_classes from detector_processor if not provided
        self.num_classes = num_classes or detector_processor.num_classes
        
        # Validate task and parameters
        if self.task not in ["binary", "multiclass", "multilabel"]:
            raise ValueError(f"Task '{task}' not supported. Use 'binary', 'multiclass', or 'multilabel'.")
            
        if self.task == "binary" and self.num_classes != 2:
            raise ValueError(f"Binary task requires num_classes=2, got {self.num_classes}")
            
        if self.top_k > self.num_classes:
            raise ValueError(f"top_k ({self.top_k}) cannot be larger than num_classes ({self.num_classes})")
        
        # State variables for accumulation
        self.add_state("correct", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        
        # For per-class metrics (when average is not "micro")
        if self.average in ["macro", "weighted", None]:
            self.add_state("correct_per_class", torch.zeros(self.num_classes, dtype=torch.long), dist_reduce_fx="sum")
            self.add_state("total_per_class", torch.zeros(self.num_classes, dtype=torch.long), dist_reduce_fx="sum")
    
    def _ensure_detector_processor_device(self, tensor_device: torch.device) -> DetectorProcessorClf:
        """Ensure detector processor is on the correct device for computation.
        
        DetectorProcessor typically operates on CPU, so we may need to move data.
        """
        # DetectorProcessor usually works on CPU
        if hasattr(self.detector_processor, '_DetectorProcessorClf__device'):
            dp_device = self.detector_processor._DetectorProcessorClf__device
        else:
            dp_device = torch.device('cpu')  # Default assumption
            
        # If detector processor is on different device, move it
        if dp_device != tensor_device and tensor_device.type == 'cuda':
            # Keep DetectorProcessor on CPU for compatibility, we'll move data instead
            return self.detector_processor
        elif dp_device != tensor_device:
            # Move detector processor to match tensor device
            return self.detector_processor.to(tensor_device)
        else:
            return self.detector_processor
    
    def _process_detector_images(self, detector_images: torch.Tensor) -> torch.Tensor:
        """Convert detector images to class probabilities using DetectorProcessor.
        
        Args:
            detector_images: Tensor of shape (batch_size, ..., H, W) with detector intensities
            
        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        original_device = detector_images.device
        
        # Move to CPU for DetectorProcessor computation (typical requirement)
        detector_images_cpu = detector_images.cpu()
        
        # Get detector processor on correct device
        dp = self._ensure_detector_processor_device(detector_images_cpu.device)
        
        # Compute class probabilities
        with torch.no_grad():
            class_probas = dp.batch_forward(detector_images_cpu)
        
        # Move back to original device if needed
        if original_device != detector_images_cpu.device:
            class_probas = class_probas.to(original_device)
            
        return class_probas
    
    def _format_targets(self, targets: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Format targets to consistent shape and type.
        
        Args:
            targets: Either class labels (batch_size,) or detector images (batch_size, ..., H, W)
            batch_size: Expected batch size
            
        Returns:
            Class labels of shape (batch_size,)
        """
        if targets.dim() == 1 and targets.shape[0] == batch_size:
            # Already class labels
            return targets.long()
        elif targets.dim() >= 3:
            # Detector images - convert to class labels using DetectorProcessor
            class_probas = self._process_detector_images(targets)
            return class_probas.argmax(dim=1)
        else:
            raise ValueError(f"Unsupported target format: shape {targets.shape}")
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric state with batch predictions and targets.
        
        Args:
            preds: Predicted detector images of shape (batch_size, ..., H, W)
            targets: Either class labels (batch_size,) or detector images (batch_size, ..., H, W)
        """
        # Validate inputs
        if preds.dim() < 3:
            raise ValueError(f"Predictions must be detector images (batch, ..., H, W), got shape {preds.shape}")
            
        batch_size = preds.shape[0]
        
        if targets.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch: preds {batch_size}, targets {targets.shape[0]}")
        
        # Convert predictions to class probabilities
        pred_probas = self._process_detector_images(preds)  # (batch_size, num_classes)
        
        # Format targets to class labels
        target_labels = self._format_targets(targets, batch_size)  # (batch_size,)
        
        # Compute predictions based on task
        if self.task == "binary":
            # Binary classification with threshold
            pred_labels = (pred_probas[:, 1] > self.threshold).long()
        elif self.task == "multiclass":
            if self.top_k == 1:
                pred_labels = pred_probas.argmax(dim=1)
            else:
                # Top-k accuracy
                _, top_k_preds = pred_probas.topk(self.top_k, dim=1)
                pred_labels = top_k_preds
        elif self.task == "multilabel":
            # Multilabel with threshold
            pred_labels = (pred_probas > self.threshold).long()
        
        # Compute correct predictions
        if self.task == "multiclass" and self.top_k > 1:
            # Top-k: check if true label is in top-k predictions
            correct_mask = (top_k_preds == target_labels.unsqueeze(1)).any(dim=1)
            correct = correct_mask.sum()
        elif self.task == "multilabel":
            # Multilabel: element-wise comparison
            correct = (pred_labels == target_labels).sum()
            batch_size = batch_size * self.num_classes  # Account for multiple labels
        else:
            # Binary/multiclass: direct comparison
            correct = (pred_labels == target_labels).sum()
        
        # Update global counters (ensure same device)
        self.correct += correct.to(self.correct.device)
        self.total += torch.tensor(batch_size, device=self.total.device)
        
        # Update per-class counters if needed
        if hasattr(self, 'correct_per_class'):
            for class_idx in range(self.num_classes):
                if self.task == "multilabel":
                    class_correct = (pred_labels[:, class_idx] == target_labels[:, class_idx]).sum()
                    class_total = batch_size // self.num_classes
                else:
                    class_mask = (target_labels == class_idx)
                    class_correct = (pred_labels[class_mask] == class_idx).sum()
                    class_total = class_mask.sum()
                
                self.correct_per_class[class_idx] += class_correct.to(self.correct_per_class.device)
                self.total_per_class[class_idx] += torch.tensor(class_total, device=self.total_per_class.device)
    
    def compute(self) -> torch.Tensor | dict:
        """Compute final accuracy metric.
        
        Returns:
            Scalar accuracy tensor or dict of per-class accuracies
        """
        if self.total == 0:
            warnings.warn("No samples processed. Returning 0.0 accuracy.")
            return torch.tensor(0.0, device=self.correct.device)
        
        if self.average == "micro" or not hasattr(self, 'correct_per_class'):
            # Micro-average (global accuracy)
            return self.correct.float() / self.total.float()
        
        elif self.average == "macro":
            # Macro-average (mean of per-class accuracies)
            per_class_acc = self.correct_per_class.float() / torch.clamp(self.total_per_class.float(), min=1)
            valid_classes = self.total_per_class > 0
            if valid_classes.sum() == 0:
                return torch.tensor(0.0, device=self.correct.device)
            return per_class_acc[valid_classes].mean()
        
        elif self.average == "weighted":
            # Weighted average by class frequency
            per_class_acc = self.correct_per_class.float() / torch.clamp(self.total_per_class.float(), min=1)
            weights = self.total_per_class.float() / self.total_per_class.sum()
            return (per_class_acc * weights).sum()
        
        elif self.average is None:
            # Return per-class accuracies
            per_class_acc = self.correct_per_class.float() / torch.clamp(self.total_per_class.float(), min=1)
            return {f"accuracy_class_{i}": acc for i, acc in enumerate(per_class_acc)}
        
        else:
            raise ValueError(f"Unsupported average type: {self.average}")
    
    def reset(self) -> None:
        """Reset metric state for new epoch."""
        self.correct.zero_()
        self.total.zero_()
        if hasattr(self, 'correct_per_class'):
            self.correct_per_class.zero_()
            self.total_per_class.zero_()
    
    def clone(self) -> 'OpticalAccuracy':
        """Create a clone of this metric for validation.
        
        Note: Since DetectorProcessor contains SimulationParameters which may not be
        deepcopy-safe, we create a new instance that shares the detector_processor.
        """
        return OpticalAccuracy(
            detector_processor=self.detector_processor,  # Share reference 
            task=self.task,
            num_classes=self.num_classes,
            threshold=self.threshold,
            top_k=self.top_k,
            average=self.average
        )


# Convenience function for easy usage
def create_optical_accuracy(
    detector_processor: DetectorProcessorClf,
    task: str = "multiclass",
    **kwargs
) -> OpticalAccuracy:
    """Convenience function to create OpticalAccuracy metric.
    
    Args:
        detector_processor: DetectorProcessorClf instance
        task: Classification task type
        **kwargs: Additional arguments for OpticalAccuracy
        
    Returns:
        OpticalAccuracy metric instance
    """
    return OpticalAccuracy(detector_processor=detector_processor, task=task, **kwargs)
