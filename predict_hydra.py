#!/usr/bin/env python3
"""
Hydra-based prediction script for protein contact prediction model.
"""

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import logging
from typing import Optional

from models.lightning_model import ContactPredictionLightningModule
from esm2_embeddings.extract_embeddings import get_embeddings
from preprocessing.contact_map import compute_contact_map
from data.pdb_utils import download_pdb_structure, extract_sequence_from_pdb

# Set up logging
log = logging.getLogger(__name__)

def create_contact_mask(seq_len, min_distance=6):
    mask = np.ones((seq_len, seq_len), dtype=np.float32)
    for i in range(seq_len):
        for j in range(seq_len):
            if abs(i - j) < min_distance:
                mask[i, j] = 0.0
    return mask

def predict_contacts_lightning(model: ContactPredictionLightningModule,
                             sequence: str,
                             device: torch.device,
                             max_length: int = 512,
                             structural_contacts: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Predict contact map for a protein sequence using Lightning model.
    
    Args:
        model: Trained Lightning contact prediction model
        sequence: Protein sequence
        device: Device to run prediction on
        max_length: Maximum sequence length
        structural_contacts: Optional structural contact map for comparison
    
    Returns:
        Predicted contact probabilities [seq_len, seq_len]
    """
    # Truncate sequence if too long
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
        log.info(f"Sequence truncated to {max_length} residues")
    
    seq_len = len(sequence)
    
    # Get ESM2 embeddings
    log.info("Extracting ESM2 embeddings...")
    embeddings, esm_contacts = get_embeddings(sequence)
    
    # Pad embeddings to max_length
    if embeddings.shape[0] < max_length:
        padding = torch.zeros(max_length - embeddings.shape[0], embeddings.shape[1])
        embeddings = torch.cat([embeddings, padding], dim=0)
    else:
        embeddings = embeddings[:max_length]
    
    # Prepare structural contacts if provided
    if structural_contacts is not None:
        if structural_contacts.shape[0] > max_length:
            structural_contacts = structural_contacts[:max_length, :max_length]
        # Pad structural contacts
        padded_structural = np.zeros((max_length, max_length))
        padded_structural[:seq_len, :seq_len] = structural_contacts
        structural_contacts = torch.from_numpy(padded_structural).float().unsqueeze(0).to(device)
    else:
        structural_contacts = None
    
    # Create attention mask
    attention_mask = torch.zeros(max_length, dtype=torch.bool)
    attention_mask[:seq_len] = True
    attention_mask = attention_mask.unsqueeze(0).to(device)
    
    # Prepare embeddings
    embeddings = embeddings.unsqueeze(0).to(device)  # Add batch dimension
    
    # Predict contacts
    log.info("Predicting contacts...")
    with torch.no_grad():
        predictions = model(embeddings, structural_contacts, attention_mask)
    
    # Extract predictions for actual sequence length
    predictions = predictions[0, :seq_len, :seq_len].cpu().numpy()
    
    # Create contact mask to exclude contacts that are too close in sequence
    contact_mask = create_contact_mask(seq_len, min_distance=6).numpy()
    predictions = predictions * contact_mask
    
    return predictions

def plot_contact_maps_lightning(predictions: np.ndarray,
                               targets: Optional[np.ndarray] = None,
                               esm_contacts: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None,
                               title: str = "Contact Map Prediction (Lightning)"):
    """
    Plot contact maps for comparison.
    
    Args:
        predictions: Predicted contact probabilities
        targets: Ground truth contact map (optional)
        esm_contacts: ESM2 contact predictions (optional)
        save_path: Path to save the plot
        title: Plot title
    """
    n_plots = 1
    if targets is not None:
        n_plots += 1
    if esm_contacts is not None:
        n_plots += 1
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot predictions
    im1 = axes[plot_idx].imshow(predictions, cmap='viridis', vmin=0, vmax=1)
    axes[plot_idx].set_title(f'{title}\n(Predicted)')
    axes[plot_idx].set_xlabel('Residue Index')
    axes[plot_idx].set_ylabel('Residue Index')
    plt.colorbar(im1, ax=axes[plot_idx])
    plot_idx += 1
    
    # Plot targets if provided
    if targets is not None:
        im2 = axes[plot_idx].imshow(targets, cmap='viridis', vmin=0, vmax=1)
        axes[plot_idx].set_title('Ground Truth')
        axes[plot_idx].set_xlabel('Residue Index')
        axes[plot_idx].set_ylabel('Residue Index')
        plt.colorbar(im2, ax=axes[plot_idx])
        plot_idx += 1
    
    # Plot ESM contacts if provided
    if esm_contacts is not None:
        im3 = axes[plot_idx].imshow(esm_contacts, cmap='viridis', vmin=0, vmax=1)
        axes[plot_idx].set_title('ESM2 Contacts')
        axes[plot_idx].set_xlabel('Residue Index')
        axes[plot_idx].set_ylabel('Residue Index')
        plt.colorbar(im3, ax=axes[plot_idx])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.info(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def evaluate_predictions_lightning(predictions: np.ndarray,
                                 targets: np.ndarray,
                                 contact_mask: np.ndarray) -> dict:
    """
    Evaluate prediction quality.
    
    Args:
        predictions: Predicted contact probabilities
        targets: Ground truth contact map
        contact_mask: Mask for valid contacts
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Apply mask
    valid_mask = contact_mask.astype(bool)
    pred_valid = predictions[valid_mask]
    target_valid = targets[valid_mask]
    
    # Convert to binary predictions
    pred_binary = (pred_valid > 0.5).astype(np.int64)
    target_binary = target_valid.astype(np.int64)
    
    # Calculate metrics
    tp = np.sum((pred_binary == 1) & (target_binary == 1))
    fp = np.sum((pred_binary == 1) & (target_binary == 0))
    fn = np.sum((pred_binary == 0) & (target_binary == 1))
    tn = np.sum((pred_binary == 0) & (target_binary == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main prediction function using Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    # Override mode to prediction
    cfg.mode = "predict"
    
    # Set device
    if cfg.trainer.accelerator == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif cfg.trainer.accelerator == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    log.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Lightning model
    # log.info(f"Loading model from {cfg.checkpoint_path}")
    model = ContactPredictionLightningModule.load_from_checkpoint(
        cfg.checkpoint_path,
        map_location=device
    )
    model.to(device)
    model.eval()
    
    log.info(f"Model hyperparameters: {model.hparams}")
    
    # Get sequence
    if cfg.sequence:
        sequence = cfg.sequence
    elif cfg.pdb_id:
        log.info(f"Downloading PDB structure {cfg.pdb_id}...")
        pdb_file = download_pdb_structure(cfg.pdb_id)
        sequence = extract_sequence_from_pdb(pdb_file)
        log.info(f"Extracted sequence: {sequence[:50]}...")
    elif cfg.pdb_file:
        sequence = extract_sequence_from_pdb(cfg.pdb_file)
        log.info(f"Extracted sequence: {sequence[:50]}...")
    else:
        log.error("Please provide either sequence, pdb_id, or pdb_file in the configuration")
        return
    
    # Get structural contacts for comparison if PDB is available
    structural_contacts = None
    if cfg.pdb_id or cfg.pdb_file:
        pdb_file = cfg.pdb_file if cfg.pdb_file else download_pdb_structure(cfg.pdb_id)
        try:
            structural_contacts = compute_contact_map(pdb_file)
            log.info("Computed structural contact map for comparison")
        except Exception as e:
            log.error(f"Could not compute structural contact map: {e}")
    
    # Get ESM2 contacts for comparison
    log.info("Getting ESM2 contact predictions...")
    _, esm_contacts = get_embeddings(sequence)
    esm_contacts = esm_contacts.cpu().numpy()
    
    # Predict contacts
    predictions = predict_contacts_lightning(
        model, sequence, device, 
        cfg.data.max_length, structural_contacts
    )
    
    # Save predictions
    predictions_path = output_dir / "predictions.npy"
    np.save(predictions_path, predictions)
    log.info(f"Predictions saved to {predictions_path}")
    
    # Evaluate if structural contacts are available
    if structural_contacts is not None:
        # Create contact mask
        contact_mask = create_contact_mask(len(sequence), min_distance=6).numpy()
        
        # Truncate structural contacts to match predictions
        if structural_contacts.shape[0] > len(sequence):
            structural_contacts = structural_contacts[:len(sequence), :len(sequence)]
        
        # Evaluate predictions
        metrics = evaluate_predictions_lightning(predictions, structural_contacts, contact_mask)
        log.info("Evaluation Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                log.info(f"{metric}: {value:.4f}")
            else:
                log.info(f"{metric}: {value}")
        
        # Save metrics
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Generate plots
    if cfg.plot:
        plot_path = output_dir / "contact_maps.png"
        plot_contact_maps_lightning(
            predictions=predictions,
            targets=structural_contacts,
            esm_contacts=esm_contacts,
            save_path=plot_path,
            title=f"Contact Map Prediction (Hydra) - Length: {len(sequence)}"
        )
    
    log.info("Prediction completed!")

if __name__ == '__main__':
    main() 