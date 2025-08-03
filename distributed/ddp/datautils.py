import torch

from torch.utils.data import Dataset

class DDPTrainDataset(Dataset):
    
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)] 
        
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        
        return self.data[index]