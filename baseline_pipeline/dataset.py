import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from pathlib import Path

class agai_correct_v3(Dataset):
    def __init__(self, csv_file, transform=None):
        csv_file = Path(csv_file)
        if csv_file.exists():
            print("CSV path exists. Continuing...")

        self.icp_data = pd.read_csv(csv_file, encoding="cp1252")
        self.transform = transform

    def __len__(self):
        return len(self.icp_data) # excludes header
    
    def __getitem__(self, idx):
        image = Image.open(self.icp_data["image_path"].iloc[idx]).convert("RGB")

        label = self.icp_data["residue_bin"].iloc[idx] #ground truth bin

        label_to_int = {
            "high":2,
            "medium":1,
            "low":0,
        }
        
        label = label_to_int[label]

        if self.transform:
            image = self.transform(image)

        return image, label
    
