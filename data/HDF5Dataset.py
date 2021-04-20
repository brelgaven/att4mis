import torch.utils.data as data
import h5py
import numpy as np
from PIL import Image as im

class HDF5Dataset(data.Dataset):
    """
    Input params:
        path: Path to the folder containing the dataset (one or multiple HDF5 files).
        transform: PyTorch transform to apply to every data instance (default = None).
    """
    def __init__(self, path, transform = None):
        super().__init__()
        self.transform = transform
        self.main_path = path
        
        reader = h5py.File(path, 'r')

        self.images = reader['images']
        self.labels = reader['labels']
                    
    def __getitem__(self, index):
        x = self.images[index]
        y = self.labels[index]
        
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        
        x = im.fromarray(x)
        y = im.fromarray(y)
        
        if self.transform != None:
            x = self.transform(x)
            y = self.transform(y)
        
        return x, y
    
    def __len__(self):
         return len(self.images)