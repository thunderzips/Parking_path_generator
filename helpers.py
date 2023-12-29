import os
import torch
from torch.utils.data import Dataset, DataLoader
import re
import sys
import numpy as np

def extract_number(s):
    # Extract the numeric part from the string using regular expression
    match = re.search(r'\d+', s)
    return int(match.group()) if match else 0

def custom_sort_key(s):
    # Define a custom sorting key using the extracted numeric part
    return extract_number(s)

class TextFileDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = os.listdir(root_dir)
        # self.file_list.sort()

        self.file_list = sorted(self.file_list, key=custom_sort_key)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, file_name)

        with open(file_path, 'r') as file:
            lines = file.readlines()
            for i,v in enumerate(lines):
                lines[i] = v.split()
                lines[i] = [j for j in lines[i] if (j not in "[] ")]
                if lines[i][0][0]=="[":
                    lines[i][0] = lines[i][0][1:]
                if lines[i][1][-1]=="]":
                    lines[i][1] = lines[i][1][:-1]
                lines[i][0], lines[i][1] = float(lines[i][0]), float(lines[i][1])

            data = torch.tensor(lines).flatten()
            # data = np.reshape(lines,(1,-1))


        if self.transform:
            data = self.transform(data)

        return data, file_name