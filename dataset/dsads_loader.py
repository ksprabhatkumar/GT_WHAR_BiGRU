import torch
from torch.utils.data import Dataset, DataLoader

class RealDSADSDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_loso_dataloaders(test_subject_id, batch_size=128):
    """
    Implements Leave-One-Subject-Out Cross Validation.
    test_subject_id: int from 1 to 8.
    """
    x = torch.load("dataset/processed/dsads_x.pt")
    y = torch.load("dataset/processed/dsads_y.pt")
    p = torch.load("dataset/processed/dsads_p.pt")

    # Create boolean masks for train/test splits based on person ID
    test_mask = (p == test_subject_id)
    train_mask = (p != test_subject_id)

    train_dataset = RealDSADSDataset(x[train_mask], y[train_mask])
    test_dataset = RealDSADSDataset(x[test_mask], y[test_mask])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader