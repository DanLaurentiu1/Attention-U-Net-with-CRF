import os
from idlelib.pyparse import trans

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from functorch.dim import Tensor
from torch.onnx.symbolic_opset9 import tensor
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class BeachLitterDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.images = sorted([root_path + "\\images\\" + i for i in os.listdir(root_path + "\\images\\")])
        self.masks = sorted([root_path + "\\maskpngs\\" + i for i in os.listdir(root_path + "\\maskpngs\\")])

        self.transform_to_image = transforms.Compose([
            transforms.ToPILImage()
        ])

        self.transform_to_tensor = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        """
        mask = self.transform_to_tensor(mask)
        for row in range(512):
            for col in range(512):
                first, second, third = float(str(mask[0][row][col].item()).split('.')[1][:4]), float(str(mask[1][row][col].item()).split('.')[1][:4]), float(
                    str(mask[2][row][col].item()).split('.')[1][:4])
                if (first == 5019.0 and second == 0.0 and third == 5019.0) or (first == 5019.0 and second == 0.0 and third == 0.0):
                    mask[0][row][col], mask[1][row][col], mask[2][row][col] = 0.0, 0.0, 0.0
                else:
                    mask[0][row][col], mask[1][row][col], mask[2][row][col] = 1.0, 1.0, 1.0
        mask = self.transform_to_image(mask)
        """
        return self.transform_to_tensor(img), self.transform_to_tensor(mask)

    def __len__(self):
        return len(self.images)
