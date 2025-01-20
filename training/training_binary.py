import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from beachlitter_dataset import BeachLitterDataset
from model.UNet import UNet


def training_binary(learning_rate, batch_size, epochs, model_save_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    dataset = BeachLitterDataset(mode="binary", partition=False)

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=True)

    model = UNet(in_channels=3, out_channels=1).to(device)
    # model.load_state_dict(torch.load("model/model_final_parameters/model_final_binary.pth", map_location=torch.device(device)))

    # model = torch.utils.checkpoint.CheckpointFunction.apply(model)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(epochs)):
        torch.save(model.state_dict(), f"model/model_final_parameters/model_final_binary_{epoch}.pth")
        model.train()
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)  # img is B x C x W x H
            mask = img_mask[1].long().to(device)  # mask is B x C x W x H

            y_pred = model(img)  # y_pred is B x C x W x H
            optimizer.zero_grad()

            loss = criterion(y_pred, mask.float())  # y_pred and mask must match shapes and types
            train_running_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)
        print(f"Train Loss EPOCH {epoch + 1}: {train_loss:.4f}")

    for epoch in tqdm(range(epochs)):
        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].long().to(device)

                y_pred = model(img)
                loss = criterion(y_pred, mask.float())

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)

        print(f"Val Loss EPOCH {epoch + 1}: {val_loss:.4f}")

    torch.save(model.state_dict(), model_save_path)
