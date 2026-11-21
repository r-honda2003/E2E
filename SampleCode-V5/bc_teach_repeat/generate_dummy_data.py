import os
import csv
import numpy as np
from PIL import Image

def generate_dummy_dataset(output_dir="./dataset", num_flights=3, frames_per_flight=50):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating dummy dataset in {output_dir}...")

    for f_idx in range(1, num_flights + 1):
        flight_name = f"flight{f_idx:02d}"
        flight_dir = os.path.join(output_dir, flight_name)
        os.makedirs(flight_dir, exist_ok=True)

        csv_path = os.path.join(flight_dir, "commands.csv")

        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "roll", "pitch", "yaw", "throttle"])

            for i in range(frames_per_flight):
                t = i * 0.1 # 10Hz (0.1秒間隔)

                # 画像生成 (224x224 のダミーノイズ/グラデーション画像)
                img_array = np.uint8(np.random.rand(224, 224, 3) * 255)
                img = Image.fromarray(img_array)
                img_path = os.path.join(flight_dir, f"{i:06d}.jpg")
                img.save(img_path)

                # コマンド生成 (-660 ~ 660 の範囲)
                roll = int(200 * np.sin(t))
                pitch = int(300 * np.cos(t))
                yaw = int(100 * np.sin(t * 0.5))
                throttle = 0

                writer.writerow([f"{t:.2f}", roll, pitch, yaw, throttle])

        print(f"Created {flight_name} with {frames_per_flight} frames.")

    print("Dummy dataset generation complete.")

if __name__ == "__main__":
    generate_dummy_dataset()
