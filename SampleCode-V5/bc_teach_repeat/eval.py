import os
import argparse
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import DroneBCDataset
from model import DroneBCModel

def parse_args():
    parser = argparse.ArgumentParser(description="DJI Mini 4 Pro Behavior Cloning Evaluation")
    parser.add_argument("--model_path", type=str, default="./output/best_model.pth", help="学習済みモデル重み (.pth)")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="データセットディレクトリ")
    parser.add_argument("--val_flights", type=str, default="", help="評価するflightフォルダ (指定なしの場合は末尾20%)")
    parser.add_argument("--output_dir", type=str, default="./output/eval", help="評価結果プロット保存先")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    # Checkpoint ロード
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)

    # 引数の引き継ぎ (checkpoint から引数を復元)
    train_args = checkpoint.get('args', {})
    backbone = train_args.get('backbone', 'mobilenet_v3_small')
    num_frames = train_args.get('num_frames', 3)

    print(f"Loading model: {args.model_path}")
    print(f"Backbone: {backbone}, Num frames: {num_frames}")

    model = DroneBCModel(backbone_type=backbone, num_frames=num_frames, freeze_backbone=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Val flights 設定
    all_flight_dirs = sorted([d for d in glob.glob(os.path.join(args.data_dir, "*")) if os.path.isdir(d)])
    if args.val_flights:
        val_names = [v.strip() for v in args.val_flights.split(",")]
        val_dirs = [d for d in all_flight_dirs if os.path.basename(d) in val_names]
    else:
        val_count = max(1, int(len(all_flight_dirs) * 0.2))
        val_dirs = all_flight_dirs[-val_count:]

    print(f"Evaluating on flights: {[os.path.basename(d) for d in val_dirs]}")

    val_dataset = DroneBCDataset(val_dirs, num_frames=num_frames, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    preds_norm = []
    gts_norm = []

    with torch.no_grad():
        for frames, targets in val_loader:
            frames = frames.to(device)
            outputs = model(frames)
            preds_norm.append(outputs.cpu().numpy())
            gts_norm.append(targets.numpy())

    preds_norm = np.vstack(preds_norm) # [N, 3] (-1 ~ 1)
    gts_norm = np.vstack(gts_norm)     # [N, 3] (-1 ~ 1)

    # 660 を掛けて実スケール ( -660 ~ 660 ) に戻す
    preds_cmd = np.round(preds_norm * 660.0)
    gts_cmd = np.round(gts_norm * 660.0)

    # MSE & MAE の計算 (正規化スケールおよび実数スケール)
    mse_norm = np.mean((preds_norm - gts_norm) ** 2, axis=0)
    mae_norm = np.mean(np.abs(preds_norm - gts_norm), axis=0)

    mse_cmd = np.mean((preds_cmd - gts_cmd) ** 2, axis=0)
    mae_cmd = np.mean(np.abs(preds_cmd - gts_cmd), axis=0)

    print("\n================ EVALUATION METRICS ================")
    print("--- Normalized Scale (-1.0 to 1.0) ---")
    print(f"Roll  - MSE: {mse_norm[0]:.4f}, MAE: {mae_norm[0]:.4f}")
    print(f"Pitch - MSE: {mse_norm[1]:.4f}, MAE: {mae_norm[1]:.4f}")
    print(f"Yaw   - MSE: {mse_norm[2]:.4f}, MAE: {mae_norm[2]:.4f}")
    print(f"Mean MSE: {np.mean(mse_norm):.4f}")

    print("\n--- Real Command Scale (-660 to 660) ---")
    print(f"Roll  - MSE: {mse_cmd[0]:.2f}, MAE: {mae_cmd[0]:.2f}")
    print(f"Pitch - MSE: {mse_cmd[1]:.2f}, MAE: {mae_cmd[1]:.2f}")
    print(f"Yaw   - MSE: {mse_cmd[2]:.2f}, MAE: {mae_cmd[2]:.2f}")
    print(f"Mean MSE: {np.mean(mse_cmd):.2f}")
    print("====================================================\n")

    # 時系列プロット (時系列での 予測 vs 実コマンド)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    channel_names = ["Roll (Left/Right)", "Pitch (Forward/Backward)", "Yaw (Turn)"]

    for i in range(3):
        axes[i].plot(gts_cmd[:, i], label="Ground Truth", color="black", alpha=0.8, linewidth=1.5)
        axes[i].plot(preds_cmd[:, i], label="Prediction", color="red", alpha=0.7, linestyle="--", linewidth=1.5)
        axes[i].set_ylabel(channel_names[i])
        axes[i].set_ylim(-700, 700)
        axes[i].grid(True, linestyle=":", alpha=0.6)
        axes[i].legend(loc="upper right")

    axes[2].set_xlabel("Frame Index")
    plt.suptitle("Validation: Predictions vs Ground Truth (Commands -660~660)")
    plt.tight_layout()
    ts_plot_path = os.path.join(args.output_dir, "timeseries_val.png")
    plt.savefig(ts_plot_path, dpi=150)
    plt.close()
    print(f"Saved timeseries plot: {ts_plot_path}")

    # 散布図プロット (Ground Truth vs Prediction)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        axes[i].scatter(gts_cmd[:, i], preds_cmd[:, i], alpha=0.4, s=15, color="blue")
        axes[i].plot([-660, 660], [-660, 660], 'r--', label="Ideal")
        axes[i].set_title(f"{channel_names[i]}")
        axes[i].set_xlabel("Ground Truth (-660 to 660)")
        axes[i].set_ylabel("Prediction (-660 to 660)")
        axes[i].set_xlim(-700, 700)
        axes[i].set_ylim(-700, 700)
        axes[i].grid(True)
        axes[i].legend()

    plt.suptitle("Scatter Plot: Ground Truth vs Prediction")
    plt.tight_layout()
    scatter_plot_path = os.path.join(args.output_dir, "scatter_val.png")
    plt.savefig(scatter_plot_path, dpi=150)
    plt.close()
    print(f"Saved scatter plot: {scatter_plot_path}")

if __name__ == "__main__":
    main()
