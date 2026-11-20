import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import DroneBCDataset
from model import DroneBCModel

def parse_args():
    parser = argparse.ArgumentParser(description="DJI Mini 4 Pro Teach & Repeat (Behavior Cloning) Training")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="flightXX フォルダが格納されたルートディレクトリ")
    parser.add_argument("--val_flights", type=str, default="", help="valに使うflightフォルダ名 (カンマ区切り, 例: flight03,flight04)")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"], help="エンコーダバックボーン")
    parser.add_argument("--num_frames", type=int, default=3, help="入力する過去フレーム数")
    parser.add_argument("--loss", type=str, default="l1", choices=["l1", "mse"], help="損失関数 (l1 or mse)")
    parser.add_argument("--epochs", type=int, default=30, help="エポック数")
    parser.add_argument("--batch_size", type=int, default=16, help="バッチサイズ")
    parser.add_argument("--lr", type=float, default=1e-3, help="学習率")
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="バックボーンを凍結するかどうか")
    parser.add_argument("--unfreeze_epoch", type=int, default=-1, help="指定したエポック以降バックボーンを解凍して微調整 (-1で終始凍結)")
    parser.add_argument("--output_dir", type=str, default="./output", help="モデル・評価グラフの出力ディレクトリ")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 飛行フォルダの検索
    all_flight_dirs = sorted([
        d for d in glob.glob(os.path.join(args.data_dir, "*"))
        if os.path.isdir(d)
    ])

    if not all_flight_dirs:
        raise FileNotFoundError(f"データフォルダが見つかりません: {args.data_dir}")

    # train / val の flight 分割
    if args.val_flights:
        val_names = [v.strip() for v in args.val_flights.split(",")]
        val_dirs = [d for d in all_flight_dirs if os.path.basename(d) in val_names]
        train_dirs = [d for d in all_flight_dirs if os.path.basename(d) not in val_names]
    else:
        # デフォルト: 後方 20% の flight を val に設定
        val_count = max(1, int(len(all_flight_dirs) * 0.2))
        train_dirs = all_flight_dirs[:-val_count]
        val_dirs = all_flight_dirs[-val_count:]

    print(f"Total flights found: {len(all_flight_dirs)}")
    print(f"Train flights ({len(train_dirs)}): {[os.path.basename(d) for d in train_dirs]}")
    print(f"Val flights   ({len(val_dirs)}): {[os.path.basename(d) for d in val_dirs]}")

    # Dataset & DataLoader の作成
    train_dataset = DroneBCDataset(train_dirs, num_frames=args.num_frames, is_train=True)
    val_dataset = DroneBCDataset(val_dirs, num_frames=args.num_frames, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Device 設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # モデルの定義
    model = DroneBCModel(
        backbone_type=args.backbone,
        num_frames=args.num_frames,
        freeze_backbone=args.freeze_backbone
    ).to(device)

    # 損失関数の選択 (L1 or MSE)
    if args.loss == "l1":
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        # 微調整 (Unfreeze) のチェック
        if args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch:
            print(f"Unfreezing backbone encoder at epoch {epoch}...")
            model.unfreeze_encoder()
            # 学習率を下げる
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1

        # Train loop
        model.train()
        train_loss = 0.0
        for frames, targets in train_loader:
            frames, targets = frames.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * frames.size(0)

        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, targets in val_loader:
                frames, targets = frames.to(device), targets.to(device)
                outputs = model(frames)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * frames.size(0)

        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Train Loss ({args.loss.upper()}): {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ベストモデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, save_path)
            print(f"--> Best model saved to {save_path} (Val Loss: {val_loss:.4f})")

    # 最終モデルの保存
    last_save_path = os.path.join(args.output_dir, "last_model.pth")
    torch.save(model.state_dict(), last_save_path)

    # 学習曲線のプロット・保存
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, label=f"Train Loss ({args.loss.upper()})")
    plt.plot(range(1, args.epochs + 1), val_losses, label=f"Val Loss ({args.loss.upper()})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Behavior Cloning Training Curve")
    plt.legend()
    plt.grid(True)
    curve_path = os.path.join(args.output_dir, "learning_curve.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"Learning curve saved to {curve_path}")

if __name__ == "__main__":
    main()
