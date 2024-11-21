import torch
from PIL import Image

from model.UNet import UNet
from torchvision import transforms

SMOOTH = 1e-6


def iou(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10

    return thresholded


def single_image_inference(image_pth, mask_path, model_pth, device):
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform_to_tensor = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])

    img = transform_to_tensor(Image.open(image_pth)).float().to(device)
    mask = transform_to_tensor(Image.open(mask_path)).float().to(device)

    img = img.unsqueeze(0)

    pred_mask = model(img)

    print(iou(pred_mask, mask))

    # pred_mask = torch.where((pred_mask > -2.4) & (pred_mask < -1.3), torch.tensor(0.0), torch.tensor(1.0))


if __name__ == "__main__":
    SINGLE_IMG_PATH = "C:\\Users\\Lau\\PycharmProjects\\Attention-U-Net-with-CRF\\beachlitter_dataset_2022_smaller\\beachlitter\\images\\000108.jpg"
    SINGLE_MASK_PATH = "C:\\Users\\Lau\\PycharmProjects\\Attention-U-Net-with-CRF\\beachlitter_dataset_2022_smaller\\beachlitter\\maskpngs\\000108.png"
    DATA_PATH = "./data"
    MODEL_PATH = "C:\\Users\\Lau\\PycharmProjects\\Attention-U-Net-with-CRF\\model\\model.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    single_image_inference(SINGLE_IMG_PATH, SINGLE_MASK_PATH, MODEL_PATH, device)
