import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from typing import Optional, List
import os
import glob

from data.dataset import ProteinContactDataset, collate_fn
from data.pdb_utils import extract_sequence_from_pdb

class ProteinContactDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for protein contact prediction.
    """
    
    def __init__(self,
                 sequences: Optional[List[str]] = None,
                 pdb_ids: Optional[List[str]] = None,
                 pdb_files: Optional[List[str]] = None,
                 max_length: int = 512,
                 contact_threshold: float = 8.0,
                 min_contact_distance: int = 6,
                 batch_size: int = 4,
                 num_workers: int = 0,
                 train_split: float = 0.7,
                 val_split: float = 0.2,
                 test_split: float = 0.1,
                 use_sample_data: bool = False,
                 cache_embeddings: bool = True):
        """
        Initialize the data module.
        
        Args:
            sequences: List of protein sequences
            pdb_ids: List of PDB IDs corresponding to sequences
            pdb_files: List of PDB file paths corresponding to sequences
            max_length: Maximum sequence length
            contact_threshold: Distance threshold for contact definition
            min_contact_distance: Minimum sequence distance for contacts
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            use_sample_data: Whether to use sample dataset
            cache_embeddings: Whether to cache ESM2 embeddings
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.sequences = sequences
        self.pdb_ids = pdb_ids
        self.pdb_files = pdb_files
        self.max_length = max_length
        self.contact_threshold = contact_threshold
        self.min_contact_distance = min_contact_distance
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.use_sample_data = use_sample_data
        self.cache_embeddings = cache_embeddings
        
        # Validate splits
        total_split = train_split + val_split + test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Train, validation, and test splits must sum to 1.0, got {total_split}")
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for training, validation, and testing.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        if stage == 'fit' or stage is None:
            # Use precomputed .pkl files for train and test sets
            train_pkl_files = sorted(glob.glob('precomputed/train/*_contact_map.pkl'))
            test_pkl_files = sorted(glob.glob('precomputed/test/*_contact_map.pkl'))
            self.train_dataset = ProteinContactDataset(
                pkl_files=train_pkl_files,
                max_length=self.max_length,
                min_contact_distance=self.min_contact_distance
            )
            self.test_dataset = ProteinContactDataset(
                pkl_files=test_pkl_files,
                max_length=self.max_length,
                min_contact_distance=self.min_contact_distance
            )
            # Optionally, split a small portion for validation
            total_size = len(self.train_dataset)
            val_size = int(self.val_split * total_size)
            train_size = total_size - val_size
            self.train_dataset, self.val_dataset = random_split(
                self.train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
        elif stage == 'test':
            if self.test_dataset is None:
                self.setup('fit')
        elif stage == 'predict':
            pass
    
    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        if self.train_dataset is None:
            self.setup('fit')
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        if self.val_dataset is None:
            self.setup('fit')
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        if self.test_dataset is None:
            self.setup('test')
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create prediction data loader."""
        # For prediction, we might want to use the test dataset
        return self.test_dataloader()
    
    @classmethod
    def from_sample_data(cls, 
                        batch_size: int = 4,
                        max_length: int = 128,
                        **kwargs) -> 'ProteinContactDataModule':
        """
        Create data module from sample data.
        
        Args:
            batch_size: Batch size
            max_length: Maximum sequence length
            **kwargs: Additional arguments
        
        Returns:
            Configured data module
        """
        return cls(
            use_sample_data=True,
            batch_size=batch_size,
            max_length=max_length,
            **kwargs
        )
    
    @classmethod
    def from_sequences(cls,
                      sequences: List[str],
                      pdb_ids: Optional[List[str]] = None,
                      pdb_files: Optional[List[str]] = None,
                      batch_size: int = 4,
                      **kwargs) -> 'ProteinContactDataModule':
        """
        Create data module from sequences.
        
        Args:
            sequences: List of protein sequences
            pdb_ids: List of PDB IDs
            pdb_files: List of PDB file paths
            batch_size: Batch size
            **kwargs: Additional arguments
        
        Returns:
            Configured data module
        """
        return cls(
            sequences=sequences,
            pdb_ids=pdb_ids,
            pdb_files=pdb_files,
            batch_size=batch_size,
            **kwargs
        )

# Import torch for random_split
import torch 