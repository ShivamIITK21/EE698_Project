from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class HairColorDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.black_hair_images = os.listdir(root + "/black")
        self.blonde_hair_images = os.listdir(root + "/blond")
        self.total_len = max(len(self.black_hair_images), len(self.blonde_hair_images))

    def __len__(self):
        return self.total_len 

    def __getitem__(self, idx):
        """
        Returns a sample and its corresponding label.
        """
        black_file_path = self.black_hair_images[idx%len(self.black_hair_images)]
        blond_file_path = self.blonde_hair_images[idx%len(self.blonde_hair_images)]

        black_img = (Image.open(self.root + "/black/" + black_file_path).convert("RGB"))
        blond_img = (Image.open(self.root + "/blond/" + blond_file_path).convert("RGB"))

        if self.transform:
            black_img = self.transform(black_img)
            blond_img = self.transform(blond_img)

        return black_img, blond_img 
