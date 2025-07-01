import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from models.contact_prediction_model import ContactPredictionModel, ContactPredictionLoss
from preprocessing.contact_map import compute_contact_map  # If needed, import create_contact_mask from here if it exists

class ContactPredictionLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for protein contact prediction.
    """
    
    def __init__(self,
                 esm_embedding_dim: int = 1280,
                 hidden_dim: int = 512,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 pos_weight: float = 10.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0):
        """
        Initialize the Lightning module.
        
        Args:
            esm_embedding_dim: Dimension of ESM2 embeddings
            hidden_dim: Hidden dimension for the network
            num_layers: Number of transformer layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            pos_weight: Weight for positive contacts
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Create model
        self.model = ContactPredictionModel(
            esm_embedding_dim=esm_embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Create loss function
        self.criterion = ContactPredictionLoss()
        
        # Learning rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1_scores = []
        
    def forward(self, 
                embeddings: torch.Tensor,
                structural_contacts: Optional[torch.Tensor] = None,
                attention_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            embeddings: ESM2 embeddings [batch_size, seq_len, esm_embedding_dim]
            structural_contacts: Structural contact maps [batch_size, seq_len, seq_len]
            attention_masks: Attention masks [batch_size, seq_len]
        
        Returns:
            Predicted contact probabilities [batch_size, seq_len, seq_len]
        """
        return self.model(embeddings, structural_contacts, attention_masks)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
        
        Returns:
            Training loss
        """
        embeddings = batch['embeddings']
        contact_maps = batch['contact_maps']
        attention_masks = batch['attention_masks']
        contact_masks = batch['contact_masks']
        
        # Forward pass
        predictions = self(embeddings, contact_maps, attention_masks)
        
        # Apply contact mask
        predictions = predictions * contact_masks.float()
        targets = contact_maps * contact_masks.float()
        
        # Compute loss
        loss = self.criterion(predictions, targets, contact_masks)
        
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_losses.append(loss.item())
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
        
        Returns:
            Dictionary with validation metrics
        """
        embeddings = batch['embeddings']
        contact_maps = batch['contact_maps']
        attention_masks = batch['attention_masks']
        contact_masks = batch['contact_masks']
        
        # Forward pass
        predictions = self(embeddings, contact_maps, attention_masks)
        
        # Apply contact mask
        predictions = predictions * contact_masks.float()
        targets = contact_maps * contact_masks.float()
        
        # Compute loss
        loss = self.criterion(predictions, targets, contact_masks)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, targets, contact_masks)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_precision', metrics['precision'], on_epoch=True, prog_bar=True)
        self.log('val_recall', metrics['recall'], on_epoch=True, prog_bar=True)
        self.log('val_f1_score', metrics['f1_score'], on_epoch=True, prog_bar=True)
        
        # Store for epoch end
        self.val_losses.append(loss.item())
        self.val_precisions.append(metrics['precision'])
        self.val_recalls.append(metrics['recall'])
        self.val_f1_scores.append(metrics['f1_score'])
        
        return {
            'val_loss': loss,
            'val_precision': metrics['precision'],
            'val_recall': metrics['recall'],
            'val_f1_score': metrics['f1_score']
        }
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
        
        Returns:
            Dictionary with test metrics
        """
        embeddings = batch['embeddings']
        contact_maps = batch['contact_maps']
        attention_masks = batch['attention_masks']
        contact_masks = batch['contact_masks']
        
        # Forward pass
        predictions = self(embeddings, contact_maps, attention_masks)
        
        # Apply contact mask
        predictions = predictions * contact_masks.float()
        targets = contact_maps * contact_masks.float()
        
        # Compute loss
        loss = self.criterion(predictions, targets, contact_masks)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, targets, contact_masks)
        
        # Log metrics
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_precision', metrics['precision'], on_epoch=True)
        self.log('test_recall', metrics['recall'], on_epoch=True)
        self.log('test_f1_score', metrics['f1_score'], on_epoch=True)
        
        return {
            'test_loss': loss,
            'test_precision': metrics['precision'],
            'test_recall': metrics['recall'],
            'test_f1_score': metrics['f1_score']
        }
    
    def _compute_metrics(self, 
                        predictions: torch.Tensor, 
                        targets: torch.Tensor, 
                        masks: torch.Tensor) -> Dict[str, float]:
        """
        Compute precision, recall, F1 score, and ROC AUC.
        
        Args:
            predictions: Predicted contact probabilities
            targets: Ground truth contact maps
            masks: Contact masks
        
        Returns:
            Dictionary with metrics
        """
        # Convert to numpy for metric computation
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        masks_np = masks.detach().cpu().numpy()
        
        # Apply masks
        valid_mask = masks_np.astype(bool)
        pred_valid = predictions_np[valid_mask]
        target_valid = targets_np[valid_mask]
        
        # Convert to binary predictions
        pred_binary = (pred_valid > 0.5).astype(np.int64)
        target_binary = target_valid.astype(np.int64)
        
        # Calculate metrics
        tp = np.sum((pred_binary == 1) & (target_binary == 1))
        fp = np.sum((pred_binary == 1) & (target_binary == 0))
        fn = np.sum((pred_binary == 0) & (target_binary == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Compute ROC AUC
        try:
            from sklearn.metrics import roc_auc_score
            # Only compute ROC AUC if there are both positive and negative samples
            if np.unique(target_binary).size > 1:
                roc_auc = roc_auc_score(target_binary, pred_valid)
            else:
                roc_auc = float('nan')
        except ImportError:
            roc_auc = float('nan')
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'roc_auc': roc_auc
        }
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',
        #     factor=0.5,
        #     patience=5,
        #     verbose=True
        # )
        
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler,
        #         'monitor': 'val_loss',
        #         'interval': 'epoch',
        #         'frequency': 1
        #     }
        # }
        return {'optimizer': optimizer}
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Log average training loss for the epoch
        avg_train_loss = np.mean(self.train_losses[-len(self.trainer.train_dataloader):])
        self.log('train_loss_epoch', avg_train_loss, on_epoch=True)
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        # Log average validation metrics for the epoch
        if self.val_losses:
            avg_val_loss = np.mean(self.val_losses[-len(self.trainer.val_dataloaders):])
            avg_val_precision = np.mean(self.val_precisions[-len(self.trainer.val_dataloaders):])
            avg_val_recall = np.mean(self.val_recalls[-len(self.trainer.val_dataloaders):])
            avg_val_f1 = np.mean(self.val_f1_scores[-len(self.trainer.val_dataloaders):])
            
            self.log('val_loss_epoch', avg_val_loss, on_epoch=True)
            self.log('val_precision_epoch', avg_val_precision, on_epoch=True)
            self.log('val_recall_epoch', avg_val_recall, on_epoch=True)
            self.log('val_f1_epoch', avg_val_f1, on_epoch=True)
    
    def plot_predictions(self, 
                        predictions: torch.Tensor,
                        targets: torch.Tensor,
                        save_path: Optional[str] = None) -> None:
        """
        Plot contact map predictions vs targets.
        
        Args:
            predictions: Predicted contact probabilities
            targets: Ground truth contact maps
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot predictions
        im1 = axes[0].imshow(predictions.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
        axes[0].set_title('Predicted Contact Map')
        axes[0].set_xlabel('Residue Index')
        axes[0].set_ylabel('Residue Index')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot targets
        im2 = axes[1].imshow(targets.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth Contact Map')
        axes[1].set_xlabel('Residue Index')
        axes[1].set_ylabel('Residue Index')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close() 