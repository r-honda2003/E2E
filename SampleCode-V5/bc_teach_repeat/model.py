import torch
import torch.nn as nn
import torchvision.models as models

class DroneBCModel(nn.Module):
    """
    DJI Mini 4 Pro 模倣学習モデル
    - バックボーン: MobileNetV3-small または ResNet18 (ImageNet事前学習)
    - 共通エンコーダで直近 N フレームの特徴量を抽出・連結
    - ヘッド: MLP (Linear -> ReLU -> Linear -> Tanh)
    - 出力: [roll, pitch, yaw] (正規化済み -1.0 ~ 1.0)
    """
    def __init__(self, backbone_type='mobilenet_v3_small', num_frames=3, hidden_dim=128, freeze_backbone=True):
        super().__init__()
        self.num_frames = num_frames
        self.backbone_type = backbone_type.lower()

        if 'resnet18' in self.backbone_type:
            weights = models.ResNet18_Weights.DEFAULT
            resnet = models.resnet18(weights=weights)
            # 最終分類層 (fc) を除外して特徴量抽出器として利用
            feat_dim = resnet.fc.in_features
            resnet.fc = nn.Identity()
            self.encoder = resnet
        else:
            # デフォルト: MobileNetV3-small
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            mobilenet = models.mobilenet_v3_small(weights=weights)
            feat_dim = mobilenet.classifier[0].in_features
            mobilenet.classifier = nn.Identity()
            self.encoder = mobilenet

        # 最初はバックボーンを凍結
        if freeze_backbone:
            self.freeze_encoder()

        # MLP ヘッド定義
        combined_feat_dim = feat_dim * num_frames
        self.head = nn.Sequential(
            nn.Linear(combined_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 3),
            nn.Tanh() # 出力を -1.0 ~ 1.0 に収める
        )

    def freeze_encoder(self):
        """エンコーダの重みを凍結"""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """エンコーダの重みを微調整用に解凍"""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        x: [B, N_frames, 3, 224, 224]
        """
        B, N, C, H, W = x.shape
        # [B * N, C, H, W] に変形して一括でエンコーダに通す
        x_flat = x.view(B * N, C, H, W)
        feats_flat = self.encoder(x_flat) # [B * N, feat_dim]

        # [B, N * feat_dim] に変換
        feat_dim = feats_flat.shape[-1]
        feats = feats_flat.view(B, N * feat_dim)

        # MLP ヘッドでコマンド予測 [B, 3] (roll, pitch, yaw)
        out = self.head(feats)
        return out
