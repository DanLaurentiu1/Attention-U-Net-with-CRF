import os

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class BeachLitterDataset(Dataset):
    def __init__(self, partition=True, mode="binary"):
        self.partition = partition
        self.mode = mode
        self.root_path = self.choose_dataset_paths()

        self.images = sorted([self.root_path + "\\images\\" + i for i in os.listdir(self.root_path + "\\images\\")])
        self.masks = sorted([self.root_path + "\\maskpngs\\" + i for i in os.listdir(self.root_path + "\\maskpngs\\")])

        self.transform_mask_to_tensor = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            lambda x: (x * 255).long()])

        self.transform_tensor_to_image = transforms.Compose([
            transforms.ToPILImage()
        ])

        self.transform_img_to_tensor = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])

    def choose_dataset_paths(self):
        if self.mode == "binary":
            if self.partition:
                return "data/beachlitter_dataset_2022_smaller_binary/beachlitter"
            elif not self.partition:
                return "data/beachlitter_dataset_2022_big_binary/beachlitter"
        elif self.mode == "multi-class":
            if self.partition:
                return "data/beachlitter_dataset_2022_smaller_multi_class/beachlitter"
            elif not self.partition:
                return "data/beachlitter_dataset_2022_big_multi_class/beachlitter"

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        mask = Image.open(self.masks[index])
        if self.mode == "multi-class":
            mask = self.transform_mask_to_tensor(mask)
            mask = mask - 1
            return self.transform_img_to_tensor(img), mask
        return self.transform_img_to_tensor(img), self.transform_img_to_tensor(mask)

    def __len__(self):
        return len(self.images)
