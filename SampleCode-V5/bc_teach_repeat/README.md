# DJI Mini 4 Pro Teach & Repeat (End-to-End 行動模倣 / Behavior Cloning)

DJI Mini 4 Pro の Teach 飛行データ（手動操縦ログ）から「カメラ画像 → 操縦コマンド (roll, pitch, yaw)」を回帰予測する End-to-End の Behavior Cloning 実装です。

---

## 📁 ディレクトリ構成

```text
bc_teach_repeat/
├── dataset.py              # データセットクラス (DroneBCDataset)
├── model.py                # モデル定義 (MobileNetV3-small / ResNet18 + MLP)
├── train.py                # 学習スクリプト (flight単位のTrain/Val分割, Loss選択, 拡張)
├── eval.py                 # 評価スクリプト (MSE/MAE算出, 時系列 & 散布図プロット)
├── server.py               # TCP 推論サーバ (Androidクライアント受信用 & 安全ホバリング機能)
├── generate_dummy_data.py  # 動作確認用ダミーデータ生成スクリプト
└── README.md               # 実行手順・説明書
```

---

## 🛠️ 主な仕様

1. **データセット (`dataset.py`)**
   - コマンド: `timestamp, roll, pitch, yaw, throttle` (-660 ～ 660)
   - 出力対象: `roll`, `pitch`, `yaw` (3次元)。学習時に 660 で割って `-1.0 ~ 1.0` に正規化。
   - 時刻照合: 各画像のタイムスタンプ以前の最新コマンドをマッチング (10Hz想定)。
   - フレームスタック: 直近 3 フレーム (過去足りない場合は先頭複製パディング)。
   - データ拡張: 色変化 (明るさ・コントラスト) + 左右反転 (左右反転時は roll と yaw の符号を反転)。

2. **モデル (`model.py`)**
   - エンコーダ: `MobileNetV3-small` または `ResNet18` (ImageNet事前学習)。
   - マルチフレーム統合: 各フレームを共通エンコーダに通して特徴量を抽出・連結。
   - 最初はバックボーンを凍結し、必要に応じて微調整 (Unfreeze) 可能。
   - ヘッド: MLP (`Linear -> ReLU -> Linear -> Tanh`)。

3. **学習 & 評価 (`train.py`, `eval.py`)**
   - **flight 単位**で Train / Val を分割。
   - L1 Loss または MSE Loss の切り替えに対応。
   - 学習曲線 (`learning_curve.png`)、時系列プロット (`timeseries_val.png`)、散布図 (`scatter_val.png`) を自動出力。

4. **推論サーバ (`server.py`)**
   - TCP ソケットサーバ (デフォルトポート 8000)。
   - 受信: 4バイト Big-Endian (JPEGバイト長) + JPEG画像バイナリ。
   - 応答: JSON フォーマット (`{"roll": 120, "pitch": -50, "yaw": 0, "throttle": 0}\n`)。
   - **安全制御**:
     - 出力の上下限クリップ (-660 ～ 660)。
     - 0.5 秒以上画像受信がない場合に自動で**ホバリングコマンド (全0)** を出力。

---

## 🚀 実行手順・コマンド例

### 1. ダミーデータの生成 (動作確認用)

リアルデータが無い場合、テスト用のダミーデータセットを作成できます。

```bash
python3 generate_dummy_data.py
```
`./dataset/flight01`, `./dataset/flight02`, `./dataset/flight03` が生成されます。

### 2. モデルの学習 (`train.py`)

#### (A) MobileNetV3-small, L1 Loss (デフォルト)
```bash
python3 train.py --data_dir ./dataset --epochs 30 --batch_size 16 --loss l1
```

#### (B) ResNet18, MSE Loss, 特定 flight を val に指定
```bash
python3 train.py --data_dir ./dataset --backbone resnet18 --loss mse --val_flights flight03 --epochs 30
```

#### (C) バックボーンの微調整 (10エポック目以降解凍)
```bash
python3 train.py --data_dir ./dataset --unfreeze_epoch 10 --epochs 30
```

### 3. モデルの評価 (`eval.py`)

学習済みモデル (`best_model.pth`) を使って評価指標 (MSE/MAE) を計算し、プロット画像を出力します。

```bash
python3 eval.py --model_path ./output/best_model.pth --data_dir ./dataset
```
`./output/eval/timeseries_val.png` と `./output/eval/scatter_val.png` が生成されます。

### 4. 推論サーバの起動 (`server.py`)

Android アプリからの接続を待ち受ける TCP サーバを起動します。

```bash
python3 server.py --model_path ./output/best_model.pth --port 8000 --timeout 0.5
```
