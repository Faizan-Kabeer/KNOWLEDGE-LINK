"""
ConvE Decoder — rebuilt to exactly match link-prediction-gatce-final.ipynb.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEDecoder(nn.Module):
    def __init__(self, num_entities: int, cfg):
        super().__init__()
        self.CONV_DH   = cfg.CONV_DH
        self.CONV_DW   = cfg.CONV_DW
        self.EMBED_DIM = cfg.EMBED_DIM

        # 2-D convolution layers
        self.conv = nn.Conv2d(1, 32, (3, 3), padding=0)
        self.bn0  = nn.BatchNorm2d(1)
        self.bn1  = nn.BatchNorm2d(32)
        self.bn2  = nn.BatchNorm1d(cfg.EMBED_DIM)

        # Three separate dropouts exactly matching the notebook
        self.input_drop       = nn.Dropout(0.2)
        self.feature_map_drop = nn.Dropout2d(0.2)
        self.hidden_drop      = nn.Dropout(0.3)

        # Output size after conv (no padding, 3×3 kernel)
        h_out = (2 * cfg.CONV_DH) - 2
        w_out = cfg.CONV_DW - 2
        self.flat_sz = 32 * h_out * w_out

        self.fc = nn.Linear(self.flat_sz, cfg.EMBED_DIM)

        # Entity bias for final scoring (new in final notebook)
        self.entity_bias = nn.Parameter(torch.zeros(num_entities))

    def forward(
        self,
        h: torch.Tensor,    # [B, EMBED_DIM]
        r: torch.Tensor,    # [B, EMBED_DIM]
        E: torch.Tensor,    # [num_entities, EMBED_DIM]
    ) -> torch.Tensor:      # [B, num_entities]
        # Reshape 1-D embeddings into 2-D maps
        h = h.view(-1, 1, self.CONV_DH, self.CONV_DW)
        r = r.view(-1, 1, self.CONV_DH, self.CONV_DW)

        # Concatenate head and relation maps vertically → [B, 1, 2*DH, DW]
        inputs = torch.cat([h, r], dim=2)

        x = self.bn0(inputs)
        x = self.input_drop(x)       # dropout on input (0.2)

        x = self.conv(x)             # [B, 32, H', W']
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x) # spatial dropout on feature maps (0.2)

        x = x.flatten(1)
        x = self.fc(x)
        x = self.hidden_drop(x)      # dropout after linear projection (0.3)
        
        # Safety for inference on single samples (BatchNorm fails on batch size 1)
        if x.size(0) > 1 or not self.training:
            x = self.bn2(x)
        
        x = F.relu(x)

        # Dot product with all entity embeddings + learnable bias
        scores = (x @ E.t()) + self.entity_bias
        return scores
