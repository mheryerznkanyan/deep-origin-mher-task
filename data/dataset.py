import torch
from torch.utils.data import Dataset
import numpy as np
from esm2_embeddings.extract_embeddings import get_embeddings
import pickle

class ProteinContactDataset(Dataset):
    """
    Dataset for protein contact prediction combining ESM2 embeddings and structural data.
    """
    
    def __init__(self, 
                 pkl_files: list,
                 max_length: int = 1024,
                 min_contact_distance: int = 8):
        """
        Initialize the dataset.
        
        Args:
            pkl_files: List of PDB file paths corresponding to sequences (optional)
            max_length: Maximum sequence length (sequences will be truncated/padded)
            min_contact_distance: Minimum sequence distance for contacts
        """
        super().__init__()
        self.pkl_files = pkl_files
        self.max_length = max_length
        self.min_contact_distance = min_contact_distance
    
    def __len__(self):
        return len(self.pkl_files)
    
    def __getitem__(self, idx):
        """Get a single protein sample."""
        with open(self.pkl_files[idx], 'rb') as f:
            data = pickle.load(f)
        contact_map = data['contact_map']
        sequence = data['sequence']
        seq_len = len(sequence)
        # Pad contact map to max_length
        if contact_map.shape[0] < self.max_length:
            padded_contact_map = np.zeros((self.max_length, self.max_length), dtype=np.int64)
            padded_contact_map[:seq_len, :seq_len] = contact_map
            contact_map = padded_contact_map
        else:
            contact_map = contact_map[:self.max_length, :self.max_length]
        # Create attention mask
        attention_mask = torch.zeros(self.max_length, dtype=torch.bool)
        attention_mask[:seq_len] = True
        # Create contact mask (exclude contacts that are too close in sequence)
        contact_mask = torch.ones(self.max_length, self.max_length, dtype=torch.bool)
        for i in range(self.max_length):
            for j in range(self.max_length):
                if abs(i - j) < self.min_contact_distance:
                    contact_mask[i, j] = False
        # Convert to tensors
        contact_map = torch.from_numpy(contact_map).float()
        # Check for NaNs/Infs in contact_map
        if np.isnan(contact_map).any() or np.isinf(contact_map).any():
            raise ValueError(f"NaN or Inf detected in contact_map for file: {self.pkl_files[idx]}")
        # Check for NaNs or invalid characters in sequence
        if sequence is None or not isinstance(sequence, str) or any([c not in 'ACDEFGHIKLMNPQRSTVWY' for c in sequence]):
            raise ValueError(f"Invalid or NaN sequence for file: {self.pkl_files[idx]} (sequence: {sequence})")
        # Get ESM2 embedding
        embedding, _ = get_embeddings(sequence)
        embedding = embedding.squeeze(0)  # [seq_len, embedding_dim]
        # Check for NaNs/Infs in embedding
        if torch.isnan(embedding).any() or torch.isinf(embedding).any():
            raise ValueError(f"NaN or Inf detected in embedding for file: {self.pkl_files[idx]}")
        # Pad embedding to max_length
        if embedding.shape[0] < self.max_length:
            pad = torch.zeros(self.max_length - embedding.shape[0], embedding.shape[1])
            embedding = torch.cat([embedding, pad], dim=0)
        else:
            embedding = embedding[:self.max_length, :]
        return {
            'contact_map': contact_map,
            'attention_mask': attention_mask,
            'contact_mask': contact_mask,
            'sequence_length': seq_len,
            'sequence': sequence,
            'embedding': embedding
        }

def collate_fn(batch):
    """
    Collate function for batching protein data.
    
    Args:
        batch: List of protein samples
    
    Returns:
        Batched data dictionary
    """
    # Stack all tensors
    contact_maps = torch.stack([item['contact_map'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    contact_masks = torch.stack([item['contact_mask'] for item in batch])
    embeddings = torch.stack([item['embedding'] for item in batch])
    # Get sequence lengths
    sequence_lengths = [item['sequence_length'] for item in batch]
    sequences = [item['sequence'] for item in batch]
    
    return {
        'contact_maps': contact_maps,
        'attention_masks': attention_masks,
        'contact_masks': contact_masks,
        'sequence_lengths': sequence_lengths,
        'sequences': sequences,
        'embeddings': embeddings
    }