"""
Logging Utilities

Sets up TensorBoard logging for training metrics.
Kaggle-compatible logging configuration.
"""

import os
from pathlib import Path
from typing import Optional
import torch


class LoggingSetup:
    """Configure TensorBoard logging for training."""
    
    @staticmethod
    def setup_tensorboard(
        log_dir: str = "logs",
        exp_name: str = "stage1_training",
    ):
        """
        Setup TensorBoard logging.
        
        Args:
            log_dir: Directory to save logs
            exp_name: Experiment name (creates subdirectory)
            
        Returns:
            writer: TensorBoard SummaryWriter
        """
        os.makedirs(log_dir, exist_ok=True)
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            log_path = os.path.join(log_dir, exp_name)
            writer = SummaryWriter(log_dir=log_path)
            print(f"✅ TensorBoard logger initialized: {log_path}")
            return writer
        
        except ImportError:
            print("⚠️  TensorBoard not available. Using placeholder.")
            return None
    
    @staticmethod
    def log_scalar(
        writer,
        tag: str,
        value: float,
        global_step: int,
    ) -> None:
        """Log scalar value to TensorBoard."""
        if writer is not None:
            writer.add_scalar(tag, value, global_step)
    
    @staticmethod
    def log_histogram(
        writer,
        tag: str,
        values: torch.Tensor,
        global_step: int,
    ) -> None:
        """Log histogram to TensorBoard."""
        if writer is not None:
            writer.add_histogram(tag, values, global_step)
    
    @staticmethod
    def log_metrics_dict(
        writer,
        metrics: dict,
        global_step: int,
        prefix: str = "",
    ) -> None:
        """Log dictionary of metrics."""
        if writer is None:
            return
        
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            if isinstance(value, (int, float)):
                writer.add_scalar(tag, value, global_step)
            elif isinstance(value, torch.Tensor):
                writer.add_histogram(tag, value, global_step)


class CSVLogger:
    """Optional CSV logging for metrics (lightweight alternative to TensorBoard)."""
    
    def __init__(self, log_file: str = "training_log.csv"):
        """Initialize CSV logger."""
        self.log_file = log_file
        self.initialized = False
    
    def log(self, metrics: dict, step: int) -> None:
        """Log metrics to CSV file."""
        # TODO: Implement CSV logging
        # Useful for Kaggle where TensorBoard visualization may be limited
        pass
