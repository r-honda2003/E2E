import os
import csv
import argparse
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from model import DroNet

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DroneDataset(Dataset):
    """
    Dataset loader for DJI Mini 4 Pro E2E Learning.
    Loads RGB images and target labels [Roll, Pitch, Yaw].

    Data Augmentation:
    - ColorJitter (brightness, contrast, saturation, hue)
    - Random Horizontal Flip:
      When flipped horizontally, Roll and Yaw signs are inverted (* -1),
      while Pitch remains unchanged (* 1).
    """
    def __init__(self, csv_file, img_dir, is_train=True, img_size=(224, 224)):
        self.img_dir = img_dir
        self.is_train = is_train
        self.img_size = img_size
        self.data = []

        # Read CSV file using Python standard library for maximum compatibility
        with open(csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Find image filename key
                img_key = None
                for key in ['image_name', 'filename', 'image', 'frame']:
                    if key in row:
                        img_key = key
                        break
                if img_key is None:
                    # Fallback to the first column if no known key found
                    img_key = list(row.keys())[0]

                self.data.append({
                    'image_name': row[img_key],
                    'roll': float(row['roll']),
                    'pitch': float(row['pitch']),
                    'yaw': float(row['yaw'])
                })

        # ColorJitter transform
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05
        )

        # ImageNet normalization (Standard for 224x224 RGB inputs)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item['image_name'])

        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.img_size, Image.BILINEAR)

        # Targets: [Roll, Pitch, Yaw]
        roll = item['roll']
        pitch = item['pitch']
        yaw = item['yaw']

        if self.is_train:
            # 1. ColorJitter
            image = self.color_jitter(image)

            # 2. Random Horizontal Flip with label adjustment
            if random.random() > 0.5:
                image = TF.hflip(image)
                roll = -roll  # Invert Roll
                yaw = -yaw    # Invert Yaw
                # pitch stays unchanged (* 1)

        # Convert to Tensor [3, H, W] in range [0.0, 1.0]
        image_tensor = TF.to_tensor(image)
        # Apply ImageNet normalization
        image_tensor = self.normalize(image_tensor)

        target = torch.tensor([roll, pitch, yaw], dtype=torch.float32)

        return image_tensor, target


def create_dummy_dataset(dummy_dir="dummy_data", num_samples=32):
    """Generates synthetic images and CSV for testing training pipeline when real dataset is not present."""
    os.makedirs(dummy_dir, exist_ok=True)
    img_dir = os.path.join(dummy_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    csv_path = os.path.join(dummy_dir, "plan.csv")

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'roll', 'pitch', 'yaw'])

        for i in range(num_samples):
            filename = f"frame_{i:04d}.png"
            img_path = os.path.join(img_dir, filename)
            # Create dummy random image
            random_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(random_img).save(img_path)

            # Generate dummy targets (-1.0 to 1.0)
            roll = round(random.uniform(-1.0, 1.0), 4)
            pitch = round(random.uniform(-1.0, 1.0), 4)
            yaw = round(random.uniform(-1.0, 1.0), 4)
            writer.writerow([filename, roll, pitch, yaw])

    print(f"Created synthetic dummy dataset at: {dummy_dir} with {num_samples} samples.")
    return csv_path, img_dir


def train_model(model, dataloader, epochs=5, lr=1e-3, device='cpu'):
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\nStarting training for {epochs} epochs on device: {device}...")
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch}/{epochs}] - Loss (MSE): {epoch_loss:.6f}")

    return model


def export_torchscript_for_android(model, output_filename="dronet_android.pt"):
    """
    Exports PyTorch model for PyTorch Mobile 1.13.1 on Android.
    Strict constraints:
    1. Set model to eval mode.
    2. Use torch.jit.trace with dummy input [1, 3, 224, 224].
    3. DO NOT use _save_for_lite_interpreter (.ptl).
    4. Save as standard TorchScript (.pt).
    """
    print(f"\n--- Exporting model for Android (PyTorch Mobile 1.13.1) ---")
    model.eval()
    model.cpu()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    with torch.no_grad():
        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)

        # Save standard TorchScript (.pt) file
        traced_model.save(output_filename)

    print(f"SUCCESS: Model saved as '{output_filename}'")
    print("NOTE: Saved as standard TorchScript (.pt) without using _save_for_lite_interpreter (.ptl).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DroNet and export to Android TorchScript (.pt)")
    parser.add_argument("--csv_path", type=str, default="", help="Path to plan.csv file")
    parser.add_argument("--img_dir", type=str, default="", help="Path to image directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output_pt", type=str, default="dronet_android.pt", help="Output .pt file path")
    args = parser.parse_args()

    set_seed(42)

    # If csv_path or img_dir is not provided or missing, generate dummy dataset for demonstration/testing
    if not args.csv_path or not os.path.exists(args.csv_path) or not args.img_dir or not os.path.exists(args.img_dir):
        print("Dataset path not specified or not found. Generating dummy dataset for pipeline verification...")
        csv_path, img_dir = create_dummy_dataset()
    else:
        csv_path = args.csv_path
        img_dir = args.img_dir

    # Dataset & DataLoader
    dataset = DroneDataset(csv_file=csv_path, img_dir=img_dir, is_train=True, img_size=(224, 224))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # Initialize DroNet Model
    model = DroNet(num_outputs=3)

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train
    model = train_model(model, dataloader, epochs=args.epochs, lr=args.lr, device=device)

    # Export for Android PyTorch Mobile 1.13.1
    export_torchscript_for_android(model, output_filename=args.output_pt)
