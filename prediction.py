import torch

from beachlitter_dataset import BeachLitterDataset
from model.UNet import UNet
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import numpy as np


def apply_dense_crf(image, probs, sxy_gaussian=3, compat_gaussian=3):
    height, width, n_classes = image.shape
    probs = probs.permute(2, 0, 1).numpy()

    dense_crf = dcrf.DenseCRF2D(width, height, n_classes)

    unary = unary_from_softmax(probs)
    dense_crf.setUnaryEnergy(unary)
    pairwise_energy = create_pairwise_bilateral(sdims=(10, 10), schan=(0.01,), img=image, chdim=2)
    dense_crf.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    dense_crf.addPairwiseEnergy(pairwise_energy)

    Q = dense_crf.inference(10)
    res = np.argmax(Q, axis=0).reshape((height, width))

    return res


def single_image_inference_binary(image_mask_index, model_pth, device):
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    img, mask = BeachLitterDataset(partition=False, mode="binary").__getitem__(image_mask_index)  # C x H x W

    img = img.unsqueeze(0)  # B x C x H x W
    mask = mask.unsqueeze(0)  # B x C x H x W

    pred_mask = model(img)
    img = img.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0)  # H x W x C for plotting

    mask = mask.squeeze(0).cpu().detach()
    mask = mask.permute(1, 2, 0)  # H x W x C for plotting

    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = torch.where(pred_mask < -0.25, torch.tensor(0.0), torch.tensor(1.0))
    pred_mask = pred_mask.permute(1, 2, 0)  # H x W x C for plotting
    fig = plt.figure()
    for i in range(1, 4):
        fig.add_subplot(1, 3, i)
        if i == 1:
            plt.imshow(img, cmap="gray")
        elif i == 2:
            plt.imshow(mask, cmap="gray")
        else:
            plt.imshow(pred_mask, cmap="gray")
    plt.show()


def single_image_inference_multi_class(image_mask_index, model_pth, device):
    model = UNet(in_channels=3, out_channels=8).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    img, mask = BeachLitterDataset(partition=False, mode="multi-class").__getitem__(image_mask_index)  # C x H x W]

    img = img.unsqueeze(0)  # B x C x H x W
    mask = mask.unsqueeze(0)  # B x C x H x W

    pred_mask = model(img)  # B x C x H x W
    pred_mask = pred_mask.squeeze(0).cpu().detach()  # C x H x W
    pred_mask = pred_mask.permute(1, 2, 0)  # H x W x C for plotting
    pred_class = np.argmax(pred_mask, axis=-1)

    img = img.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0)  # H x W x C for plotting

    mask = mask.squeeze(0).cpu().detach()
    mask = mask.permute(1, 2, 0)  # H x W x C for plotting

    apply_dense_crf(image=pred_mask, probs=pred_class)
    fig = plt.figure()
    for i in range(1, 4):
        fig.add_subplot(1, 3, i)
        if i == 1:
            plt.imshow(img, cmap="gray")
        elif i == 2:
            plt.imshow(mask, cmap="gray")
        else:
            plt.imshow(pred_class, cmap="tab10")
    plt.show()


if __name__ == "__main__":
    BINARY_MODEL_PATH = "model/model_final_parameters/model_final_binary.pth"
    MULTI_CLASS_MODEL_PATH = "model/model_final_parameters/model_final_multi_class.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    single_image_inference_binary(image_mask_index=144, model_pth=BINARY_MODEL_PATH, device=device)
    single_image_inference_multi_class(image_mask_index=715, model_pth=MULTI_CLASS_MODEL_PATH, device=device)
