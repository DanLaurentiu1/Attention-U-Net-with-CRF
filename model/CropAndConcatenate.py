import torch
import torchvision.transforms.functional
from torch import nn


class CropAndConcatenate(nn.Module):
    def forward(self, expansive_feature_map, contracting_feature_map):
        contracting_feature_map = torchvision.transforms.functional.center_crop(contracting_feature_map, [expansive_feature_map.shape[2], expansive_feature_map.shape[3]])
        expansive_feature_map = torch.cat([expansive_feature_map, contracting_feature_map], dim=1)
        return expansive_feature_map
