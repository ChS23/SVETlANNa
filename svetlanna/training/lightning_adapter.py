from collections.abc import Callable
from typing import Any

import torch
from lightning import LightningModule
from torch import nn
from torch.optim import Optimizer

from svetlanna.setup import LinearOpticalSetup


class LinearOpticalSetupLightning(LightningModule):
    """PyTorch Lightning adapter for LinearOpticalSetup.
    
    Wraps a LinearOpticalSetup into a LightningModule, providing standard
    training/validation/test interfaces while preserving the optical system's
    structure and functionality.
    
    This adapter handles:
    - Forward pass through the optical system
    - Loss computation and metric tracking
    - Optimizer configuration
    - Learning rate scheduling
    - Logging and checkpointing
    
    Example:
        >>> optical_setup = LinearOpticalSetup([lens, detector])
        >>> lightning_model = LinearOpticalSetupLightning(
        ...     optical_setup=optical_setup,
        ...     loss_fn=nn.CrossEntropyLoss(),
        ...     optimizer_config={'lr': 0.001, 'weight_decay': 1e-5}
        ... )
    """

    def __init__(
        self,
        optical_setup: LinearOpticalSetup,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer_config: dict[str, Any] | None = None,
        scheduler_config: dict[str, Any] | None = None,
        metrics: list[Any] | None = None,
        log_every_n_steps: int = 50
    ):
        """Initialize the Lightning adapter.
        
        Args:
            optical_setup: The optical system to wrap
            loss_fn: Loss function for training
            optimizer_config: Optimizer configuration dict with 'optimizer' and params
            scheduler_config: Learning rate scheduler configuration
            metrics: List of metrics to track during training
            log_every_n_steps: Frequency of logging during training
            
        Optimizer config format:
            {
                'optimizer': 'Adam',  # or torch.optim.Adam class
                'lr': 0.001,
                'weight_decay': 1e-5,
                'betas': (0.9, 0.999)
            }
            
        Scheduler config format:
            {
                'scheduler': 'StepLR',  # or torch.optim.lr_scheduler.StepLR class
                'step_size': 30,
                'gamma': 0.1,
                'interval': 'epoch',  # 'epoch' or 'step'
                'frequency': 1
            }
        """
        super().__init__()

        # Store core components - register as submodule to get parameters
        self.optical_setup = optical_setup
        self.loss_fn = loss_fn
        self.log_every_n_steps = log_every_n_steps

        # Set default optimizer config
        self.optimizer_config = optimizer_config or {
            "optimizer": "Adam",
            "lr": 0.001,
            "weight_decay": 1e-5
        }

        self.scheduler_config = scheduler_config

        # Initialize metrics
        self.train_metrics = nn.ModuleList(metrics or [])
        self.val_metrics = nn.ModuleList([m.clone() for m in self.train_metrics] if metrics else [])
        self.test_metrics = nn.ModuleList([m.clone() for m in self.train_metrics] if metrics else [])

        # Save hyperparameters (excluding non-serializable objects)
        self.save_hyperparameters(ignore=["optical_setup", "loss_fn", "metrics"])

        # Enable automatic optimization by default
        self.automatic_optimization = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the optical system.
        
        Args:
            x: Input tensor (typically a Wavefront or tensor to be converted)
            
        Returns:
            Output tensor from the optical system
        """
        return self.optical_setup(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step for a single batch.
        
        Args:
            batch: Training batch (typically (input, target) tuple)
            batch_idx: Index of the current batch
            
        Returns:
            Loss tensor for this batch
        """
        # Unpack batch
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            inputs, targets = batch
        else:
            raise ValueError(f"Expected batch to be (input, target) tuple, got {type(batch)}")

        # Forward pass
        outputs = self(inputs)

        # Compute loss
        loss = self.loss_fn(outputs, targets)

        # Log loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Compute and log metrics
        for metric in self.train_metrics:
            metric_value = metric(outputs, targets)
            metric_name = metric.__class__.__name__.lower()
            self.log(f"train_{metric_name}", metric_value, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step for a single batch.
        
        Args:
            batch: Validation batch
            batch_idx: Index of the current batch
            
        Returns:
            Loss tensor for this batch
        """
        # Unpack batch
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            inputs, targets = batch
        else:
            raise ValueError(f"Expected batch to be (input, target) tuple, got {type(batch)}")

        # Forward pass
        outputs = self(inputs)

        # Compute loss
        loss = self.loss_fn(outputs, targets)

        # Log loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Compute and log metrics
        for metric in self.val_metrics:
            metric_value = metric(outputs, targets)
            metric_name = metric.__class__.__name__.lower()
            self.log(f"val_{metric_name}", metric_value, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Test step for a single batch.
        
        Args:
            batch: Test batch
            batch_idx: Index of the current batch
            
        Returns:
            Loss tensor for this batch
        """
        # Unpack batch
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            inputs, targets = batch
        else:
            raise ValueError(f"Expected batch to be (input, target) tuple, got {type(batch)}")

        # Forward pass
        outputs = self(inputs)

        # Compute loss
        loss = self.loss_fn(outputs, targets)

        # Log loss
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        # Compute and log metrics
        for metric in self.test_metrics:
            metric_value = metric(outputs, targets)
            metric_name = metric.__class__.__name__.lower()
            self.log(f"test_{metric_name}", metric_value, on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Prediction step for inference.
        
        Args:
            batch: Input batch for prediction
            batch_idx: Index of the current batch
            dataloader_idx: Index of the dataloader (for multiple prediction dataloaders)
            
        Returns:
            Model predictions
        """
        # Handle both single inputs and (input, target) tuples
        if isinstance(batch, (tuple, list)) and len(batch) >= 1:
            inputs = batch[0]
        else:
            inputs = batch

        # Forward pass
        return self(inputs)

    def configure_optimizers(self) -> Optimizer | dict[str, Any]:
        """Configure optimizer and optional learning rate scheduler.
        
        Returns:
            Optimizer or dict with optimizer and scheduler configuration
        """
        # Get optimizer class and parameters
        optimizer_name = self.optimizer_config.pop("optimizer", "Adam")

        if isinstance(optimizer_name, str):
            # Get optimizer class by name
            optimizer_cls = getattr(torch.optim, optimizer_name)
        else:
            # Assume it's already a class
            optimizer_cls = optimizer_name

        # Create optimizer - ensure we have parameters
        params = list(self.parameters())
        if not params:
            # If no trainable parameters, create a dummy parameter
            # This can happen with optical setups that have no trainable components
            self.dummy_param = nn.Parameter(torch.zeros(1))
            params = [self.dummy_param]

        optimizer = optimizer_cls(params, **self.optimizer_config)

        # If no scheduler config, return just the optimizer
        if self.scheduler_config is None:
            return optimizer

        # Configure scheduler
        scheduler_name = self.scheduler_config.pop("scheduler")
        scheduler_params = self.scheduler_config.copy()

        # Extract Lightning-specific params
        interval = scheduler_params.pop("interval", "epoch")
        frequency = scheduler_params.pop("frequency", 1)
        monitor = scheduler_params.pop("monitor", None)

        if isinstance(scheduler_name, str):
            # Get scheduler class by name
            scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)
        else:
            # Assume it's already a class
            scheduler_cls = scheduler_name

        # Create scheduler
        scheduler = scheduler_cls(optimizer, **scheduler_params)

        # Build scheduler configuration
        scheduler_config = {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": frequency,
        }

        # Add monitor for ReduceLROnPlateau
        if monitor is not None:
            scheduler_config["monitor"] = monitor

        # Return optimizer and scheduler configuration
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config
        }

    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        # Reset metrics
        for metric in self.train_metrics:
            if hasattr(metric, "reset"):
                metric.reset()

    def on_validation_epoch_end(self) -> None:
        """Called at the end of each validation epoch."""
        # Reset metrics
        for metric in self.val_metrics:
            if hasattr(metric, "reset"):
                metric.reset()

    def on_test_epoch_end(self) -> None:
        """Called at the end of each test epoch."""
        # Reset metrics
        for metric in self.test_metrics:
            if hasattr(metric, "reset"):
                metric.reset()

    def get_optical_setup(self) -> LinearOpticalSetup:
        """Get the underlying optical setup.
        
        Returns:
            The wrapped LinearOpticalSetup
        """
        return self.optical_setup

    def set_loss_function(self, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        """Update the loss function.
        
        Args:
            loss_fn: New loss function
        """
        self.loss_fn = loss_fn

    def add_metric(self, metric: Any, stage: str = "all") -> None:
        """Add a metric to track during training.
        
        Args:
            metric: Metric to add (should have torchmetrics-like interface)
            stage: Which stage to add metric to ('train', 'val', 'test', or 'all')
        """
        if stage in ("train", "all"):
            self.train_metrics.append(metric)

        if stage in ("val", "all"):
            self.val_metrics.append(metric.clone() if hasattr(metric, "clone") else metric)

        if stage in ("test", "all"):
            self.test_metrics.append(metric.clone() if hasattr(metric, "clone") else metric)
