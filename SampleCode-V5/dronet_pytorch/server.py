import os
import sys
import time
import socket
import struct
import json
import argparse

import numpy as np
import torch
from PIL import Image
import io
import torchvision.transforms as T

from model import DroNet

class DroNetPredictor:
    """
    PC Inference Server using DroNet (ResNet-8 backbone).
    Predicts Roll, Pitch, Yaw from real-time camera frames received from Android app.
    """
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading DroNet model weights from: {model_path}")
        self.model = DroNet(num_outputs=3).to(self.device)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("Successfully loaded model weights.")
        else:
            print(f"WARNING: Model path '{model_path}' not found! Operating with uninitialized model.")

        self.model.eval()

        # ImageNet normalization transform
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def predict(self, jpeg_bytes):
        """
        Decodes JPEG frame, passes through DroNet, and returns control commands (-660 ~ 660).
        """
        # Decode JPEG image
        img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)

        # Tensor conversion & normalization -> [1, 3, 224, 224]
        t_img = T.functional.to_tensor(img)
        t_img = self.normalize(t_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_norm = self.model(t_img).cpu().numpy()[0]  # [roll, pitch, yaw] normalized in [-1.0, 1.0]

        # Safety control 1: Clip to [-1.0, 1.0]
        output_norm = np.clip(output_norm, -1.0, 1.0)

        # Denormalize to joystick range (-660 ~ 660)
        roll_cmd = int(np.round(output_norm[0] * 660.0))
        pitch_cmd = int(np.round(output_norm[1] * 660.0))
        yaw_cmd = int(np.round(output_norm[2] * 660.0))

        # Safety control 2: Clip commands to [-660, 660]
        roll_cmd = int(np.clip(roll_cmd, -660, 660))
        pitch_cmd = int(np.clip(pitch_cmd, -660, 660))
        yaw_cmd = int(np.clip(yaw_cmd, -660, 660))
        throttle_cmd = 0  # Fixed throttle for level flight

        return {
            "roll": roll_cmd,
            "pitch": pitch_cmd,
            "yaw": yaw_cmd,
            "throttle": throttle_cmd
        }

    def get_safe_hover_command(self):
        """Safe hover command on timeout or connection loss"""
        return {"roll": 0, "pitch": 0, "yaw": 0, "throttle": 0}


def recv_exact(sock, n_bytes):
    """Helper to receive exact number of bytes from socket"""
    data = bytearray()
    while len(data) < n_bytes:
        packet = sock.recv(n_bytes - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def run_server(args):
    predictor = DroNetPredictor(model_path=args.model_path)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((args.host, args.port))
    server_socket.listen(1)

    print(f"\n==================================================")
    print(f" DroNet Real-Time PC Inference Server Started!")
    print(f" Listening on {args.host}:{args.port}")
    print(f" Model: {args.model_path}")
    print(f"==================================================\n")

    while True:
        try:
            print("Waiting for connection from Android App (Repeat Mode)...")
            client_socket, client_address = server_socket.accept()
            print(f"CONNECTED: Android Client at {client_address}")
            client_socket.settimeout(args.timeout)

            while True:
                try:
                    # 1. Read 4-byte header (image length)
                    length_bytes = recv_exact(client_socket, 4)
                    if not length_bytes:
                        print("Client disconnected.")
                        break

                    img_len = struct.unpack(">I", length_bytes)[0]

                    # 2. Read JPEG image data
                    jpeg_bytes = recv_exact(client_socket, img_len)
                    if not jpeg_bytes:
                        print("Failed to receive full frame data.")
                        break

                    # 3. DroNet Inference
                    cmd = predictor.predict(jpeg_bytes)

                    # 4. Return JSON response
                    response_json = json.dumps(cmd) + "\n"
                    client_socket.sendall(response_json.encode('utf-8'))
                    print(f"Recv ({img_len:6d} B) -> DroNet Output: Roll={cmd['roll']:4d}, Pitch={cmd['pitch']:4d}, Yaw={cmd['yaw']:4d}")

                except socket.timeout:
                    print(f"[SAFETY WARNING] Frame timeout ({args.timeout}s). Sending Hover Command...")
                    hover_cmd = predictor.get_safe_hover_command()
                    client_socket.sendall((json.dumps(hover_cmd) + "\n").encode('utf-8'))

        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            client_socket.close()


def parse_args():
    parser = argparse.ArgumentParser(description="DroNet PC Real-Time Inference Server")
    parser.add_argument("--model_path", type=str, default="dronet_best.pth", help="Path to trained DroNet model (.pth)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--timeout", type=float, default=0.5, help="Timeout in seconds for safe hover")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_server(args)
