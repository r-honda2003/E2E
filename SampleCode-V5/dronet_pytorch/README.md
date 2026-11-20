# DroNet (ResNet-8) PyTorch Training & Android Export

DJI Mini 4 Pro の E2E 模倣学習（T&R: Target & Regulation）用 DroNet モデルの学習・エクスポート用ディレクトリです。

## 📁 ファイル構成

- **`model.py`**: Android PyTorch Mobile 1.13.1 互換の DroNet (ResNet-8) モデル定義
- **`train_and_export.py`**: CSV/画像読み込み、データ拡張（ColorJitter, 左右反転 + Roll/Yaw反転）、学習および `.pt` 保存スクリプト
- **`test_export.py`**: エクスポートされた `.pt` ファイルの動作および出力シェイプ (`[1, 3]`) の検証スクリプト

---

## 🚀 使い方

### 1. 学習とエクスポート

実データセット（CSV と画像フォルダ）を指定して実行します：

```bash
python3 train_and_export.py \
  --csv_path /path/to/plan.csv \
  --img_dir /path/to/images \
  --epochs 10 \
  --batch_size 16 \
  --lr 0.001 \
  --output_pt dronet_android.pt
```

※ 引数を指定せずに `python3 train_and_export.py` を実行した場合、自動的に合成ダミーデータセットを作成して学習・エクスポートの動作確認を行います。

### 2. エクスポートされた TorchScript (`.pt`) の検証

```bash
python3 test_export.py dronet_android.pt
```

出力結果例：
```text
--- Verifying Exported TorchScript Model: dronet_android.pt ---
1. torch.jit.load() successful.
2. Input shape prepared: torch.Size([1, 3, 224, 224])
3. Forward pass successful. Output: tensor([[ 0.0941, -0.0959, -0.0862]])
4. Checking output shape... Output shape is torch.Size([1, 3]), Expected is torch.Size([1, 3])

=======================================================
 VERIFICATION PASSED: Model is ready for Android (PyTorch Mobile 1.13.1)!
=======================================================
```

---

## ⚙️ 仕様・必須制約への準拠

1. **入出力仕様**
   - **入力**: `[1, 3, 224, 224]` (RGB, ImageNet正規化: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
   - **出力**: `[1, 3]` (Roll, Pitch, Yaw の連続値回帰)

2. **データ拡張**
   - **ColorJitter**: 明るさ・コントラストの変更
   - **Random Horizontal Flip**: 左右反転時に `Roll` と `Yaw` の符号を反転 (`* -1`)、`Pitch` はそのまま (`* 1`)

3. **Android (PyTorch Mobile 1.13.1) 互換性**
   - SIGABRT 回避のため `Conv2d`, `BatchNorm2d`, `ReLU`, `MaxPool2d`, `AvgPool2d`, `Linear`, `Add` などの基本演算子のみを使用。
   - `_save_for_lite_interpreter` (.ptl) は使わず、`model.eval()` 後に `torch.jit.trace` から `traced.save("dronet_android.pt")` で標準 TorchScript (`.pt`) として出力。
