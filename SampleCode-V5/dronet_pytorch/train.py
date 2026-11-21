import os
import csv
import glob
import random
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from model import DroNet

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MultiFlightDroneDataset(Dataset):
    """
    Dataset loader for teach & repeat flights.
    Can load all flights or specific flight directory (e.g., 1st run only).
    """
    def __init__(self, dataset_dir, target_folder=None, is_train=True, img_size=(224, 224), use_intervened_only=False):
        self.is_train = is_train
        self.img_size = img_size
        self.samples = []

        if target_folder:
            # Load specific single flight folder
            full_target = os.path.join(dataset_dir, target_folder) if not target_folder.startswith('/') else target_folder
            flight_dirs = [full_target] if os.path.exists(full_target) else []
        else:
            # Find all flight directories
            flight_dirs = sorted(glob.glob(os.path.join(dataset_dir, "*flight_*")))

        print(f"Target dataset directories ({len(flight_dirs)}): {flight_dirs}")

        total_intervened_count = 0

        for fdir in flight_dirs:
            csv_path = os.path.join(fdir, "commands.csv")
            if not os.path.exists(csv_path):
                print(f"Warning: CSV not found in {fdir}, skipping...")
                continue

            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            for idx, row in enumerate(rows):
                img_name = f"{idx + 1:06d}.jpg"
                img_path = os.path.join(fdir, img_name)

                if not os.path.exists(img_path):
                    img_name_png = f"{idx + 1:06d}.png"
                    img_path_png = os.path.join(fdir, img_name_png)
                    if os.path.exists(img_path_png):
                        img_path = img_path_png
                    else:
                        continue

                # Check intervention flag if present
                is_intervened = int(row.get('is_intervened', 0))
                if is_intervened == 1:
                    total_intervened_count += 1

                if use_intervened_only and is_intervened == 0:
                    continue

                roll = float(row['roll']) / 660.0    # Normalize -660~660 to -1.0~1.0
                pitch = float(row['pitch']) / 660.0  # Normalize -660~660 to -1.0~1.0
                yaw = float(row['yaw']) / 660.0      # Normalize -660~660 to -1.0~1.0

                self.samples.append({
                    'img_path': img_path,
                    'roll': roll,
                    'pitch': pitch,
                    'yaw': yaw
                })

        print(f"Total collected sample frames: {len(self.samples)} (Human Intervention Frames: {total_intervened_count})")

        # Data Augmentation: ColorJitter
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        )
        # ImageNet Normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item['img_path']).convert('RGB')
        image = image.resize(self.img_size, Image.BILINEAR)

        roll = item['roll']
        pitch = item['pitch']
        yaw = item['yaw']

        if self.is_train:
            # 1. ColorJitter
            image = self.color_jitter(image)

            # 2. Random Horizontal Flip with label sign adjustment
            if random.random() > 0.5:
                image = TF.hflip(image)
                roll = -roll  # Invert Roll
                yaw = -yaw    # Invert Yaw
                # Pitch remains unchanged

        image_tensor = TF.to_tensor(image)
        image_tensor = self.normalize(image_tensor)
        target = torch.tensor([roll, pitch, yaw], dtype=torch.float32)

        return image_tensor, target


def train_dronet(dataset_dir, target_folder=None, epochs=10, batch_size=16, lr=1e-3, output_model="dronet_best.pth"):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for training.")

    dataset = MultiFlightDroneDataset(dataset_dir=dataset_dir, target_folder=target_folder, is_train=True)
    if len(dataset) == 0:
        print("ERROR: No valid dataset frames found!")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Initialize fresh DroNet model for 1st run training
    model = DroNet(num_outputs=3).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n==================================================")
    print(f"Starting DroNet Training for {epochs} Epochs")
    print(f"Target Folder: {target_folder if target_folder else 'ALL'}")
    print(f"Total Samples: {len(dataset)} | Batch Size: {batch_size}")
    print(f"==================================================")

    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - MSE Loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }
            torch.save(checkpoint, output_model)

    print(f"\nTraining completed! Best model saved to '{output_model}' (Loss: {best_loss:.6f}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--dataset_dir", type=str, default="/home/ruri/E2E/e2e/SampleCode-V5/bc_teach_repeat/dataset")
    parser.add_argument("--target_folder", type=str, default="", help="Specific flight folder name (e.g. flight_202611325_130442)")
    parser.add_argument("--output_model", type=str, default="dronet_best.pth")
    args = parser.parse_args()

    train_dronet(
        dataset_dir=args.dataset_dir,
        target_folder=args.target_folder if args.target_folder else None,
        epochs=args.epochs,
        batch_size=16,
        lr=1e-3,
        output_model=args.output_model
    )
