"""
Checkpoint Utilities

Manages saving and loading checkpoints for training and resume functionality.
Ensures Kaggle compatibility with structured checkpoint format.
"""

import os
import torch
from pathlib import Path
from typing import Dict, Optional, Any


class CheckpointManager:
    """Manage checkpoint saving, loading, and resume logic."""
    
    @staticmethod
    def save_checkpoint(
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        checkpoint_dir: str = "checkpoints",
        save_best: bool = False,
        is_best: bool = False,
    ) -> str:
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch number
            model: Model to checkpoint
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
            metrics: Optional metrics dict (loss, accuracy, etc.)
            checkpoint_dir: Directory to save checkpoint
            save_best: Whether to save as best checkpoint
            is_best: Whether this is the best checkpoint so far
            
        Returns:
            checkpoint_path: Path to saved checkpoint
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint["metrics"] = metrics
        
        # Save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, "latest.ckpt")
        torch.save(checkpoint, latest_path)
        print(f"✅ Checkpoint saved: {latest_path}")
        
        # Save best checkpoint if applicable
        if is_best and save_best:
            best_path = os.path.join(checkpoint_dir, "best.ckpt")
            torch.save(checkpoint, best_path)
            print(f"✅ Best checkpoint saved: {best_path}")
        
        # Save periodic checkpoint
        periodic_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.ckpt")
        torch.save(checkpoint, periodic_path)
        
        return latest_path
    
    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optional optimizer to restore state
            scheduler: Optional scheduler to restore state
            device: Device to load checkpoint to ('cpu' or 'cuda')
            
        Returns:
            Checkpoint metadata (epoch, metrics, etc.)
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"✅ Model loaded from: {checkpoint_path}")
        
        # Load optimizer state if available
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"✅ Optimizer state restored")
        
        # Load scheduler state if available
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"✅ Scheduler state restored")
        
        metadata = {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
        }
        
        return metadata
    
    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        """Find latest checkpoint in directory."""
        latest_path = os.path.join(checkpoint_dir, "latest.ckpt")
        if os.path.exists(latest_path):
            return latest_path
        return None
