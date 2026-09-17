import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class InterestDataset(Dataset):
    """Dataset for interest prediction
    
    Args:
        data_path: Path to parquet file with features
        mode: 'train', 'val', or 'test'
    """
    
    def __init__(
        self,
        data_path: str,
        mode: str = 'train'
    ):
        self.mode = mode
        self.data_path = Path(data_path)
        
        logger.info(f"Loading {mode} data from {data_path}")
        self.df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(self.df)} samples")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Profile features
        profile = torch.tensor(row['profile_features'], dtype=torch.float32)
        
        # Behavior sequence (actions over time)
        behavior_seq = torch.tensor(row['behavior_sequence'], dtype=torch.float32)
        behavior_mask = torch.tensor(row['behavior_mask'], dtype=torch.bool)
        
        # Social graph (adjacency list or embeddings)
        graph_features = torch.tensor(row['graph_features'], dtype=torch.float32)
        graph_adj = torch.tensor(row['graph_adjacency'], dtype=torch.float32)
        
        # Labels (multi-hot encoding of interests)
        labels = torch.tensor(row['interest_labels'], dtype=torch.float32)
        
        sample = {
            'profile': profile,
            'behavior_seq': behavior_seq,
            'behavior_mask': behavior_mask,
            'graph_features': graph_features,
            'graph_adj': graph_adj,
            'labels': labels,
            'user_id': row['user_id']
        }
        
        return sample


class InterestDataModule:
    """Data module for managing train/val/test datasets
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data (optional)
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
    """
    
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: Optional[str] = None,
        batch_size: int = 256,
        num_workers: int = 4
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self):
        """Load datasets"""
        logger.info("Setting up data module...")
        
        self.train_dataset = InterestDataset(self.train_path, mode='train')
        self.val_dataset = InterestDataset(self.val_path, mode='val')
        
        if self.test_path:
            self.test_dataset = InterestDataset(self.test_path, mode='test')
            
        logger.info("Data module setup complete")
        
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """Get test dataloader"""
        if self.test_dataset is None:
            return None
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_sample_batch(self) -> Dict[str, torch.Tensor]:
        """Get a sample batch for testing"""
        loader = self.train_dataloader()
        return next(iter(loader))
