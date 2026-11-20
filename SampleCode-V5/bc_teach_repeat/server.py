import os
import sys
import time
import socket
import struct
import json
import argparse
from collections import deque

import numpy as np
import torch
from PIL import Image
import io
import torchvision.transforms as T

from model import DroneBCModel

class DroneBCPredictor:
    """
    画像フレーム履歴を保持し、推論コマンドを計算・安全制御する予測器
    """
    def __init__(self, model_path, num_frames=3, device=None):
        self.num_frames = num_frames
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # モデルロード
        print(f"Loading checkpoint from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        train_args = checkpoint.get('args', {})
        backbone = train_args.get('backbone', 'mobilenet_v3_small')

        self.model = DroneBCModel(backbone_type=backbone, num_frames=num_frames, freeze_backbone=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 画像正規化トランスフォーム
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # 直近 N フレームのリングバッファ
        self.frame_buffer = deque(maxlen=num_frames)
        self.last_receive_time = time.time()

    def reset(self):
        """バッファのリセット"""
        self.frame_buffer.clear()
        self.last_receive_time = time.time()

    def predict(self, jpeg_bytes):
        """
        JPEGバイトデータから直近Nフレームバッファを更新し、操縦コマンド [roll, pitch, yaw, throttle] を予測
        """
        self.last_receive_time = time.time()

        # JPEG デコード
        img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)

        # Tensor 変換 & 正規化
        t_img = T.functional.to_tensor(img) # [3, 224, 224]
        t_img = self.normalize(t_img)

        # バッファへの追加
        self.frame_buffer.append(t_img)

        # バッファが不足している場合は先頭フレームでパディング
        while len(self.frame_buffer) < self.num_frames:
            self.frame_buffer.appendleft(t_img)

        # [1, N_frames, 3, 224, 224]
        input_tensor = torch.stack(list(self.frame_buffer), dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_norm = self.model(input_tensor).cpu().numpy()[0] # [roll, pitch, yaw] in [-1.0, 1.0]

        # 安全制御 1: クリップ処理 (-1.0 ~ 1.0)
        output_norm = np.clip(output_norm, -1.0, 1.0)

        # 660 を掛けて四捨五入し、-660 ~ 660 の整数に逆正規化
        roll_cmd = int(np.round(output_norm[0] * 660.0))
        pitch_cmd = int(np.round(output_norm[1] * 660.0))
        yaw_cmd = int(np.round(output_norm[2] * 660.0))

        # 安全制御 2: 出力値の上限クリップ (-660 ~ 660)
        roll_cmd = int(np.clip(roll_cmd, -660, 660))
        pitch_cmd = int(np.clip(pitch_cmd, -660, 660))
        yaw_cmd = int(np.clip(yaw_cmd, -660, 660))
        throttle_cmd = 0 # 高度一定の前提

        return {
            "roll": roll_cmd,
            "pitch": pitch_cmd,
            "yaw": yaw_cmd,
            "throttle": throttle_cmd
        }

    def get_safe_hover_command(self):
        """安全制御 3: 一定時間受信がない場合のゼロ出力 (ホバリング)"""
        return {
            "roll": 0,
            "pitch": 0,
            "yaw": 0,
            "throttle": 0
        }

def run_server(args):
    predictor = DroneBCPredictor(model_path=args.model_path, num_frames=args.num_frames)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((args.host, args.port))
    server_socket.listen(1)

    print(f"==================================================")
    print(f"Drone Teach & Repeat Inference Server running!")
    print(f"Listening on {args.host}:{args.port}")
    print(f"Timeout for safe hover: {args.timeout} seconds")
    print(f"==================================================")

    while True:
        try:
            print("\nWaiting for Android client connection...")
            client_socket, client_address = server_socket.accept()
            print(f"Client connected from: {client_address}")
            predictor.reset()
            client_socket.settimeout(args.timeout)

            while True:
                try:
                    # 1. 画像データの長さ (4バイト Big-Endian 整数) を受信
                    length_bytes = recv_exact(client_socket, 4)
                    if not length_bytes:
                        print("Client disconnected.")
                        break

                    img_len = struct.unpack(">I", length_bytes)[0]

                    # 2. JPEG 画像本体を受信
                    jpeg_bytes = recv_exact(client_socket, img_len)
                    if not jpeg_bytes:
                        print("Failed to receive full JPEG data.")
                        break

                    # 3. 操縦コマンドの予測
                    cmd = predictor.predict(jpeg_bytes)

                    # 4. JSON 形式で応答を送信
                    response_json = json.dumps(cmd) + "\n"
                    client_socket.sendall(response_json.encode('utf-8'))
                    print(f"Recv Image ({img_len} bytes) -> Send Cmd: {cmd}")

                except socket.timeout:
                    # 受信タイムアウト時の安全対応: ホバリングコマンドを出力
                    print(f"[SAFETY WARNING] Timeout ({args.timeout}s)! Sending Safe Hover Command...")
                    hover_cmd = predictor.get_safe_hover_command()
                    response_json = json.dumps(hover_cmd) + "\n"
                    try:
                        client_socket.sendall(response_json.encode('utf-8'))
                    except Exception:
                        break

        except Exception as e:
            print(f"Error handling connection: {e}")
        finally:
            client_socket.close()

def recv_exact(sock, n_bytes):
    """指定バイト数を確実に受信するヘルパー関数"""
    data = bytearray()
    while len(data) < n_bytes:
        packet = sock.recv(n_bytes - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)

def parse_args():
    parser = argparse.ArgumentParser(description="DJI Mini 4 Pro Behavior Cloning Inference Server")
    parser.add_argument("--model_path", type=str, default="./output/best_model.pth", help="学習済みモデル重み (.pth)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="バインドするIPアドレス")
    parser.add_argument("--port", type=int, default=8000, help="バインドするポート")
    parser.add_argument("--num_frames", type=int, default=3, help="入力フレーム数")
    parser.add_argument("--timeout", type=float, default=0.5, help="受信タイムアウト時間 (秒) - 超過時はホバリング出力")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_server(args)
