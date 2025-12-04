import torch
import torch.nn as nn

class TrafficCNN_TinyTransformer(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()

        # ============================
        # CNN BACKBONE (2-layer tiny)
        # ============================
        # Conv block 1: 28x28 -> 14x14
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv block 2: 14x14 -> 7x7
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        # CNN output shape → (B, 64, 7, 7)
        self.d_model = 128         # Transformer embedding dim
        self.h = 7
        self.w = 7
        self.seq_len = self.h * self.w  # 49 tokens

        # ============================
        # LEARNABLE POS EMBEDDING
        # ============================
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.seq_len, self.d_model)
        )

        # ============================
        # TRANSFORMER ENCODER
        # ============================
        self.num_heads = 4            # works: 64 / 4 = 16 per head
        self.ff_dim = 256
        self.num_layers = 2

        self.norm_in = nn.LayerNorm(self.d_model)
        self.norm_out = nn.LayerNorm(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        # ============================
        # CLASSIFIER HEAD
        # ============================
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(self.d_model, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # ---- CNN ----
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # x: (B, 64, 7, 7)

        # ---- flatten to sequence ----
        x = x.flatten(2)         # (B, 64, 49)
        x = x.permute(0, 2, 1)   # (B, 49, 64)

        # ---- positional embeddings ----
        x = x + self.pos_embedding[:, :x.size(1), :]

        x = self.norm_in(x)
        x = self.transformer(x)
        x = self.norm_out(x)

        # ---- global average pool tokens ----
        x = x.mean(dim=1)        # (B, 64)

        # ---- classifier ----
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x