import torch
from PIL import Image
from matplotlib import pyplot as plt
from networkx.algorithms.operators.binary import intersection

from beachlitter_dataset import BeachLitterDataset
from model.UNet import UNet
from torchvision import transforms

SMOOTH = 1e-6


def binary_iou(prediction, ground_truth):
    # prediction -> H x W (int64)
    # ground_truth -> H x W (int64)

    intersection = (prediction.bool() & ground_truth.bool()).float().sum()
    union = (prediction.bool() | ground_truth.bool()).float().sum()

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.item()


def multi_class_iou(prediction, ground_truth):
    # prediction -> H x W (int64)
    # ground_truth -> H x W (int64)
    targets = torch.unique(ground_truth)
    iou = 0
    for target in targets:
        prediction_mask = prediction == target
        ground_truth_mask = ground_truth == target
        intersection = (prediction_mask & ground_truth_mask).sum().item()
        union = (prediction_mask | ground_truth_mask).sum().item()
        iou += (intersection + SMOOTH) / (union + SMOOTH)

    iou /= len(targets)

    return iou


def binary_single_image_evaluation(image_mask_index, model_pth, device):
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    # C x H x W
    img, mask = BeachLitterDataset(partition=False, mode="binary").__getitem__(image_mask_index)

    img = img.unsqueeze(0)  # B x C x H x W
    mask = mask.squeeze(0).cpu().detach()  # H x W
    mask = 1.0 - mask

    pred_mask = model(img)  # B x C x H x W
    pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().detach()  # H x W

    pred_mask = torch.where(pred_mask < -0.25, torch.tensor(1.0), torch.tensor(0.0))

    return binary_iou(pred_mask, mask)


def multi_class_single_image_evaluation(image_mask_index, model_pth, device):
    model = UNet(in_channels=3, out_channels=8).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    img, mask = BeachLitterDataset(partition=False, mode="multi-class").__getitem__(image_mask_index)

    img = img.unsqueeze(0)
    mask = mask.squeeze(0)

    pred_mask = model(img)  # B x C x H x W
    pred_mask = pred_mask.squeeze(0).cpu().detach()  # C x H x W
    pred_mask = pred_mask.permute(1, 2, 0)  # H x W x C  -> just to be consistent with prediction
    pred_class = torch.argmax(pred_mask, dim=-1)
    return multi_class_iou(pred_class, mask)


if __name__ == "__main__":
    BINARY_MODEL_PATH = "model/model_final_parameters/model_final_binary.pth"
    MULTI_CLASS_MODEL_PATH = "model/model_final_parameters/model_final_multi_class.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    binary_dict = {}
    multi_class_dict = {}
    for i in range(3500):
        print(f"index = {i}")
        binary_iou_score = binary_single_image_evaluation(image_mask_index=i, model_pth=BINARY_MODEL_PATH, device=device)
        multi_class_iou_score = multi_class_single_image_evaluation(image_mask_index=i, model_pth=MULTI_CLASS_MODEL_PATH, device=device)
        binary_dict[i] = binary_iou_score
        multi_class_dict[i] = multi_class_iou_score

        print(f"binary is: {binary_iou_score}")
        print(f"multi-class is: {multi_class_iou_score}\n\n")

    binary_dict = sorted(binary_dict.items(), key=lambda x: x[1], reverse=True)
    multi_class_dict = sorted(multi_class_dict.items(), key=lambda x: x[1], reverse=True)
