from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch

class CelebA(Dataset):
    def __init__(self, root, labels, transform=None):
        self.root = root
        self.transform = transform
        self.img_root = root + "/Img/img_align_celeba"
        self.csv_path = root + "/list_attr_celeba.csv" 
        self.num_labels = len(labels)
        self.anno_table = pd.read_csv(self.csv_path)[labels + ["image_id"]]
        

    def __len__(self):
        return len(self.anno_table)

    def __getitem__(self, idx):
        row = self.anno_table.iloc[idx].tolist()
        filepath = self.img_root + "/" +  row[-1]
        img_attr = row[0:-1]
        img = Image.open(filepath).convert("RGB")
        if self.transform:
            img = self.transform(img)
        image_one_hot = [float(x == 1) for x in img_attr]
        return img, torch.FloatTensor(image_one_hot)
