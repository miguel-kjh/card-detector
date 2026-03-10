import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.config import TrainConfig


class CardsDataset(Dataset):
    def __init__(self, df, data_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.data_dir, row["filepaths"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["class index"])


def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def get_loaders(cfg: TrainConfig):
    csv_path = os.path.join(cfg.data_dir, "cards.csv")
    df = pd.read_csv(csv_path)
    # Drop corrupted rows (filepaths that don't exist on disk)
    df = df[df["filepaths"].apply(lambda p: os.path.exists(os.path.join(cfg.data_dir, p)))]

    train_df = df[df["data set"] == "train"]
    val_df   = df[df["data set"] == "valid"]
    test_df  = df[df["data set"] == "test"]

    train_tf, val_tf = get_transforms()

    train_loader = DataLoader(
        CardsDataset(train_df, cfg.data_dir, train_tf),
        batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        CardsDataset(val_df, cfg.data_dir, val_tf),
        batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        CardsDataset(test_df, cfg.data_dir, val_tf),
        batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
    )

    splits = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    return train_loader, val_loader, test_loader, splits
