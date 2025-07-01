import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

log = logging.getLogger(__name__)

class ContactPredictionModel(nn.Module):
    """
    Original model: ESM2 embedding encoder, transformer, layer norm, and pairwise + structural features for contact prediction.
    """
    def __init__(self, esm_embedding_dim: int = 1280, hidden_dim: int = 512, num_layers: int = 3, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.esm_embedding_dim = esm_embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding_proj = nn.Linear(esm_embedding_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*2, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        # Contact prediction head
        self.contact_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, esm_embeddings: torch.Tensor, structural_contacts: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        log.info(f"[Model] Forward: esm_embeddings shape: {esm_embeddings.shape}")
        x = self.embedding_proj(esm_embeddings)
        x = self.layer_norm(x)
        x = self.transformer(x)
        x = self.dropout(x)
        batch_size, seq_len, hidden_dim = x.shape
        # Pairwise features
        x_i = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
        x_j = x.unsqueeze(1).expand(-1, seq_len, -1, -1)
        pairwise_features = torch.cat([x_i, x_j], dim=-1)  # [batch, seq_len, seq_len, hidden_dim*2]
        # Structural features
        if structural_contacts is not None:
            if structural_contacts.dim() == 3:
                structural_features = structural_contacts.unsqueeze(-1)  # [batch, seq_len, seq_len, 1]
            else:
                structural_features = structural_contacts
        else:
            structural_features = torch.zeros(batch_size, seq_len, seq_len, hidden_dim, device=x.device)
        # Pad or trim structural features to match hidden_dim
        if structural_features.size(-1) < hidden_dim:
            structural_features = F.pad(structural_features, (0, hidden_dim - structural_features.size(-1)))
        elif structural_features.size(-1) > hidden_dim:
            structural_features = structural_features[..., :hidden_dim]
        # Concatenate all features
        all_features = torch.cat([pairwise_features, structural_features], dim=-1)  # [batch, seq_len, seq_len, hidden_dim*3]
        # Predict contact probabilities
        contact_probs = self.contact_head(all_features).squeeze(-1)
        log.info(f"[Model] contact_probs shape: {contact_probs.shape}, min: {contact_probs.min().item()}, max: {contact_probs.max().item()}, NaNs: {torch.isnan(contact_probs).any().item()}")
        return contact_probs

class ContactPredictionLoss(nn.Module):
    """
    Simple binary cross-entropy loss for contact prediction.
    """
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        log.info(f"[Loss] predictions shape: {predictions.shape}, min: {predictions.min().item()}, max: {predictions.max().item()}, NaNs: {torch.isnan(predictions).any().item()}")
        log.info(f"[Loss] targets shape: {targets.shape}, min: {targets.min().item()}, max: {targets.max().item()}, NaNs: {torch.isnan(targets).any().item()}")
        if mask is not None:
            predictions = predictions * mask
            targets = targets * mask
        if torch.isnan(predictions).any() or torch.isnan(targets).any():
            log.error("Predictions or targets contain NaNs.")
            raise ValueError("Predictions or targets contain NaNs.")
        loss = F.binary_cross_entropy(predictions, targets)
        log.info(f"[Loss] BCE loss: {loss.item()}")
        return loss