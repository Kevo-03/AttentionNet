import torch
import torch.nn as nn

class TrafficCNN_Transformer(nn.Module):
    def __init__(self, num_classes=12):
        super(TrafficCNN_Transformer, self).__init__()
        
        # =======================
        # CNN FEATURE EXTRACTOR
        # =======================
        # Input: (B, 1, 28, 28)

        # Conv block 1: 28x28 -> 28x28 -> 14x14
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            padding=1           # keep 28x28
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        
        # Conv block 2: 14x14 -> 14x14 -> 7x7
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1           # keep 14x14
        )
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7

        # Conv block 3: 7x7 -> 7x7 (no pooling)
        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            padding=1           # keep 7x7
        )
        self.bn3 = nn.BatchNorm2d(256)

        # Activations
        self.relu = nn.ReLU(inplace=True)   # CNN uses ReLU
        self.gelu = nn.GELU()               # Transformer & FC use GELU
        
        # After convs and pools, shape is:
        #  Input:  (B, 1, 28, 28)
        #  block1: (B, 64, 14, 14)
        #  block2: (B, 128, 7, 7)
        #  block3: (B, 256, 7, 7)
        #
        # We'll treat (7 * 7) = 49 as sequence length and 256 as d_model.
        self.d_model = 256
        self.h = 7
        self.w = 7
        self.seq_len = self.h * self.w     # 49 tokens

        # =======================
        # TRANSFORMER BLOCK
        # =======================
        # Learnable positional embeddings: (1, 49, 256)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.seq_len, self.d_model)
        )

        # You can tweak these if you want:
        self.num_heads = 4                  # 256 / 4 = 64
        self.num_layers = 2
        self.ff_dim = 512                   # dim_feedforward

        self.norm_in = nn.LayerNorm(self.d_model)
        self.norm_out = nn.LayerNorm(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=0.1,
            activation='gelu',              # <-- GELU inside transformer FFN
            batch_first=True,               # (B, L, d_model)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        # =======================
        # CLASSIFIER HEAD
        # =======================
        # After transformer:
        #   x: (B, 49, 256) -> mean over 49 -> (B, 256)
        self.dropout1 = nn.Dropout(0.5)     # stronger dropout before first FC
        self.dropout2 = nn.Dropout(0.3)     # milder before final FC
        self.fc1 = nn.Linear(self.d_model, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # -------------
        # CNN backbone (ReLU)
        # -------------
        # x: (B, 1, 28, 28)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)                   # (B, 64, 14, 14)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)                   # (B, 128, 7, 7)
        
        x = self.relu(self.bn3(self.conv3(x)))
        # (B, 256, 7, 7)

        # -------------
        # To sequence
        # -------------
        # (B, 256, 7, 7) -> (B, 256, 49)
        x = x.flatten(2)
        # (B, 256, 49) -> (B, 49, 256)
        x = x.permute(0, 2, 1)

        # -------------
        # Transformer (GELU inside)
        # -------------
        x = self.norm_in(x)

        # Add positional embeddings (broadcast along batch)
        # x: (B, 49, 256), pos_embedding: (1, 49, 256)
        #x = x + self.pos_embedding[:, :x.size(1), :]

        x = self.transformer(x)             # (B, 49, 256)
        x = self.norm_out(x)

        # -------------
        # Pool over tokens
        # -------------
        x = x.mean(dim=1)                   # (B, 256)

        # -------------
        # Classifier (GELU)
        # -------------
        x = self.dropout1(x)
        x = self.gelu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)                     # (B, num_classes)

        return x