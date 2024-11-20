import torch

from model.UNet import UNet
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def single_image_inference(image_pth, model_pth, device):
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])

    img = transform(Image.open(image_pth)).float().to(device)
    img = img.unsqueeze(0)

    pred_mask = model(img)

    img = img.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0)

    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)
    pred_mask = torch.where((pred_mask > -2.4) & (pred_mask < -1.3), torch.tensor(0.0), torch.tensor(1.0))
    fig = plt.figure()
    for i in range(1, 3):
        fig.add_subplot(1, 2, i)
        if i == 1:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(pred_mask, cmap="gray")
    plt.show()


if __name__ == "__main__":
    SINGLE_IMG_PATH = "C:\\Users\\Lau\\PycharmProjects\\Attention-U-Net-with-CRF\\beachlitter_dataset_2022_smaller\\beachlitter\\images\\000108.jpg"
    DATA_PATH = "./data"
    MODEL_PATH = "C:\\Users\\Lau\\PycharmProjects\\Attention-U-Net-with-CRF\\model\\model.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device)
