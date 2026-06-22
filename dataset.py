import torch
from torch.utils.data import Dataset

class MultimodalSportsDataset(Dataset):
    def __init__(self, num_samples=640, seq_len=10, num_modalities=3, feature_dim=256, num_classes=4):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_modalities = num_modalities
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Mock features representing Visual, Acoustic, and Textual modalities
        self.X = torch.randn(num_samples, seq_len, num_modalities, feature_dim)
        
        # Match phase (e.g., normalized time left in match)
        self.phase = torch.rand(num_samples, seq_len, 1)
        
        # Event classes (e.g., 0: Normal, 1: Goal/Score, 2: Foul, 3: Highlight Climax)
        self.labels = torch.randint(0, num_classes, (num_samples, seq_len))
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return self.X[idx], self.phase[idx], self.labels[idx]
