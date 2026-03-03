"""
ConvE Decoder — unchanged from the training notebook.
Scores (head, relation) against ALL entity embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.CONV_DH = cfg.CONV_DH
        self.CONV_DW = cfg.CONV_DW
        self.EMBED_DIM = cfg.EMBED_DIM

        self.conv = nn.Conv2d(1, 32, (3, 3), padding=0)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(cfg.EMBED_DIM)

        self.drop = nn.Dropout(cfg.DROPOUT)

        # Output size after conv (no padding, 3×3 kernel)
        h_out = (2 * cfg.CONV_DH) - 2
        w_out = cfg.CONV_DW - 2
        self.flat_sz = 32 * h_out * w_out

        self.fc = nn.Linear(self.flat_sz, cfg.EMBED_DIM)

    def forward(
        self,
        h: torch.Tensor,    # [B, EMBED_DIM]
        r: torch.Tensor,    # [B, EMBED_DIM]
        E: torch.Tensor,    # [num_entities, EMBED_DIM]
    ) -> torch.Tensor:      # [B, num_entities]
        B = h.size(0)

        # Reshape embeddings to images
        h = h.view(B, 1, self.CONV_DH, self.CONV_DW)
        r = r.view(B, 1, self.CONV_DH, self.CONV_DW)

        # Stack along height → [B, 1, 2*DH, DW]
        x = torch.cat([h, r], dim=2)

        x = self.bn0(x)
        x = self.conv(x)     # [B, 32, H', W']
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop(x)

        x = x.view(B, -1)
        x = self.fc(x)
        x = self.drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Dot product against all entity embeddings
        scores = x @ E.t()   # [B, num_entities]
        return scores
