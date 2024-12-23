import torch
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torch.utils import checkpoint
from tqdm import tqdm

from beachlitter_dataset import BeachLitterDataset
from model.UNet import UNet

if __name__ == "__main__":
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    EPOCHS = 2
    DATA_PATH = "C:\\Users\\Lau\\PycharmProjects\\Attention-U-Net-with-CRF\\beachlitter_dataset_2022_big_binary\\beachlitter"
    MODEL_SAVE_PATH = "C:\\Users\\Lau\\PycharmProjects\\Attention-U-Net-with-CRF\\model\\model.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    dataset = BeachLitterDataset(root_path=DATA_PATH)

    generator = torch.Generator().manual_seed(42)


    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = UNet(in_channels=3, out_channels=1).to(device)

    # model = torch.utils.checkpoint.CheckpointFunction.apply(model)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            optimizer.zero_grad()

            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)

        print(f"Train Loss EPOCH {epoch + 1}: {train_loss:.4f}")

    for epoch in tqdm(range(EPOCHS)):
        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = model(img)
                loss = criterion(y_pred, mask)

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)

        print(f"Val Loss EPOCH {epoch + 1}: {val_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
