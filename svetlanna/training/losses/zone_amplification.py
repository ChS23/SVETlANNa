import torch
import torch.nn as nn


class ZoneAmplificationLoss(nn.Module):
    """Zone-based amplification loss for optical neural networks.
    
    This loss function encourages amplification in target zones (correct class regions)
    while suppressing amplification in non-target zones. It's particularly useful
    for segmented detector configurations where different spatial regions
    correspond to different output classes.
    
    The loss works by:
    1. Identifying positive zones (target class regions) using detector mask
    2. Identifying negative zones (non-target class regions)  
    3. Maximizing power in positive zones while minimizing power in negative zones
    
    Args:
        detector_mask: Tensor of shape [H, W] containing zone labels (0-indexed)
        margin: Margin for the hinge-like loss (default: 0.0)
        neg_weight: Weight for negative zone suppression (default: 0.15)
        
    Example:
        >>> detector_mask = torch.randint(0, 10, (64, 64))  # 10 classes
        >>> loss_fn = ZoneAmplificationLoss(detector_mask, margin=0.1, neg_weight=0.2)
        >>> detector_output = torch.rand(32, 64, 64)  # batch_size=32
        >>> target_labels = torch.randint(0, 10, (32,))
        >>> loss = loss_fn(detector_output, target_labels)
    """
    
    def __init__(self, detector_mask: torch.Tensor, margin: float = 0.0, neg_weight: float = 0.15, 
                 pos_threshold: float = 0.90, adaptive_weight: bool = True):
        super().__init__()
        self.register_buffer('detector_mask', detector_mask.float())
        self.margin = margin
        self.neg_weight = neg_weight
        self.pos_threshold = pos_threshold
        self.adaptive_weight = adaptive_weight
        
        # Calculate class balance for adaptive weighting
        if adaptive_weight:
            unique_labels = torch.unique(detector_mask)
            n_classes = len(unique_labels)
            pixels_per_class = [(detector_mask == label).sum().item() for label in unique_labels]
            total_pixels = detector_mask.numel()
            
            # Store class weights for balanced learning
            self.register_buffer('class_weights', torch.tensor([
                total_pixels / (n_classes * count) if count > 0 else 1.0 
                for count in pixels_per_class
            ], dtype=torch.float32))
        
    def forward(self, detector_output: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        """Compute zone amplification loss.
        
        Args:
            detector_output: Detector output tensor of shape [B, H, W]
            target_labels: Target class labels of shape [B]
            
        Returns:
            Scalar loss tensor
        """
        if detector_output.dim() != 3:
            raise ValueError(f"Expected detector_output to have 3 dimensions [B, H, W], got {detector_output.dim()}")
        
        if target_labels.dim() != 1:
            raise ValueError(f"Expected target_labels to have 1 dimension [B], got {target_labels.dim()}")
            
        if detector_output.shape[0] != target_labels.shape[0]:
            raise ValueError(f"Batch size mismatch: detector_output has {detector_output.shape[0]}, target_labels has {target_labels.shape[0]}")
        
        batch_size = detector_output.shape[0]
        device = detector_output.device
        
        losses = []
        
        for i in range(batch_size):
            out = detector_output[i]  # [H, W]
            label = target_labels[i].item()  # scalar
            
            # Create masks for positive and negative zones using original logic
            # Positive mask: regions where detector_mask * label >= threshold
            pos_mask = (self.detector_mask * label >= self.pos_threshold).float().to(device)
            
            # Negative mask: regions where detector_mask * (1-label) < threshold  
            neg_mask = (self.detector_mask * (1 - label) < self.pos_threshold).float().to(device)
            
            # Calculate power in positive and negative zones
            pos_power = (out * pos_mask).sum()
            neg_power = (out * neg_mask).sum()
            
            # Apply adaptive weighting if enabled
            if self.adaptive_weight and hasattr(self, 'class_weights'):
                # Weight the loss based on class frequency
                class_weight = self.class_weights[min(label, len(self.class_weights) - 1)]
                effective_neg_weight = self.neg_weight * class_weight
            else:
                effective_neg_weight = self.neg_weight
            
            # Normalize by zone areas to prevent bias towards larger zones
            pos_area = pos_mask.sum() + 1e-8  # avoid division by zero
            neg_area = neg_mask.sum() + 1e-8
            
            normalized_pos_power = pos_power / pos_area
            normalized_neg_power = neg_power / neg_area
            
            # Zone amplification loss: maximize positive power, minimize negative power
            # Using hinge-like loss with margin and normalization
            loss = torch.clamp(-normalized_pos_power + effective_neg_weight * normalized_neg_power + self.margin, min=0.0)
            
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
    def extra_repr(self) -> str:
        return (f'margin={self.margin}, neg_weight={self.neg_weight}, '
                f'pos_threshold={self.pos_threshold}, adaptive_weight={self.adaptive_weight}, '
                f'detector_shape={tuple(self.detector_mask.shape)}')
