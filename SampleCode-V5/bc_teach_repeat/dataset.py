import os
import re
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

class DroneBCDataset(Dataset):
    """
    DJI Mini 4 Pro 模倣学習用データセット
    - 画像: [flight_dir]/*.jpg または *.png
    - コマンド: [flight_dir]/commands.csv (列: timestamp, roll, pitch, yaw, throttle)
    - 正規化: roll, pitch, yaw を 660 で割って [-1, 1] に正規化
    - 直近 num_frames フレームを過去方向にスタック
    """
    def __init__(self, flight_dirs, num_frames=3, transform=None, is_train=True):
        super().__init__()
        self.num_frames = num_frames
        self.is_train = is_train
        self.transform = transform

        # ImageNet の平均・標準偏差 (フレーム毎に適用)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.samples = [] # 各要素: (flight_id, frame_idx_list, label_vector)
        self.flight_images = {} # flight_id -> list of image_paths

        for f_dir in flight_dirs:
            if not os.path.exists(f_dir):
                print(f"Warning: Directory not found: {f_dir}")
                continue

            flight_id = os.path.basename(os.path.normpath(f_dir))

            # 画像ファイルの検索 (*.jpg, *.jpeg, *.png)
            img_paths = sorted(
                glob.glob(os.path.join(f_dir, "*.jpg")) +
                glob.glob(os.path.join(f_dir, "*.png")) +
                glob.glob(os.path.join(f_dir, "*.jpeg")) +
                glob.glob(os.path.join(f_dir, "images", "*.jpg")) +
                glob.glob(os.path.join(f_dir, "images", "*.png"))
            )

            # コマンド CSV の探索
            csv_path = os.path.join(f_dir, "commands.csv")
            if not os.path.exists(csv_path):
                # なければ plan.csv 等も探す
                alt_csv = glob.glob(os.path.join(f_dir, "*.csv"))
                if alt_csv:
                    csv_path = alt_csv[0]
                else:
                    print(f"Warning: CSV not found in {f_dir}")
                    continue

            df_cmd = pd.read_csv(csv_path)

            # 列名の小文字化・クリーンアップ
            df_cmd.columns = [c.strip().lower() for c in df_cmd.columns]

            if img_paths and not df_cmd.empty:
                self.flight_images[flight_id] = img_paths

                # 各画像に対応する最新のコマンドを取得
                for img_idx, img_p in enumerate(img_paths):
                    # 画像タイムスタンプの取得 (ファイル名中の数値 or 10Hz仮定)
                    t_img = self._extract_timestamp(img_p, img_idx)

                    # timestamp列がある場合は画像の時刻より前の最新コマンドを照合
                    if 'timestamp' in df_cmd.columns:
                        matched = df_cmd[df_cmd['timestamp'] <= t_img]
                        if matched.empty:
                            matched_row = df_cmd.iloc[0]
                        else:
                            matched_row = matched.iloc[-1]
                    else:
                        row_idx = min(img_idx, len(df_cmd) - 1)
                        matched_row = df_cmd.iloc[row_idx]

                    # roll, pitch, yaw を取得して 660 で割って正規化 (-1 ~ 1)
                    r = float(matched_row['roll']) / 660.0
                    p = float(matched_row['pitch']) / 660.0
                    y = float(matched_row['yaw']) / 660.0

                    target = np.array([r, p, y], dtype=np.float32)
                    target = np.clip(target, -1.0, 1.0)

                    # 直近 num_frames フレームのインデックスリストを作成 (不足分は先頭フレーム複製)
                    frame_indices = []
                    for k in range(self.num_frames - 1, -1, -1):
                        past_idx = max(0, img_idx - k)
                        frame_indices.append(past_idx)

                    self.samples.append((flight_id, frame_indices, target))

    def _extract_timestamp(self, img_path, default_idx):
        fname = os.path.splitext(os.path.basename(img_path))[0]
        # ファイル名の中の浮動小数点または整数を検索
        numbers = re.findall(r"\d+\.?\d*", fname)
        if numbers:
            val = float(numbers[-1])
            # 小さい連番値の可能性が高い場合は 10Hz (0.1秒) に換算
            if val < 10000 and '.' not in fname:
                return val * 0.1
            return val
        return default_idx * 0.1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flight_id, frame_indices, target = self.samples[idx]
        img_paths = self.flight_images[flight_id]

        # Nフレーム画像の読み込み
        frames = []
        for f_idx in frame_indices:
            img = Image.open(img_paths[f_idx]).convert('RGB')
            img = img.resize((224, 224), Image.BILINEAR)
            frames.append(img)

        # データ拡張 (学習時)
        # 色変化 (明るさ・コントラスト)
        if self.is_train and random.random() < 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            frames = [TF.adjust_brightness(f, brightness_factor) for f in frames]
            frames = [TF.adjust_contrast(f, contrast_factor) for f in frames]

        # 水平反転 (左右反転時は roll と yaw の符号を反転)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        if self.is_train and random.random() < 0.5:
            frames = [TF.hflip(f) for f in frames]
            # roll: target[0], pitch: target[1], yaw: target[2]
            # 左右反転で roll (左右傾き) と yaw (左右旋回) の符号反転
            target_tensor[0] = -target_tensor[0]
            target_tensor[2] = -target_tensor[2]

        # テンソル変換 & 正規化
        tensor_frames = []
        for f in frames:
            t_img = TF.to_tensor(f) # [3, H, W] in [0, 1]
            t_img = self.normalize(t_img)
            tensor_frames.append(t_img)

        # [num_frames, 3, 224, 224]
        stacked_frames = torch.stack(tensor_frames, dim=0)

        return stacked_frames, target_tensor
