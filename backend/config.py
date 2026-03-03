"""
Central configuration for KnowledgeLink.
All hyperparameters here MUST match what was used during training.
"""
import os
import torch


class Config:
    # ── Data ────────────────────────────────────────────────
    data_dir: str = "data"          # folder with train.txt / valid.txt / test.txt
    checkpoint_path: str = "checkpoint.pth"

    # ── Model (overridden automatically from checkpoint) ────────────────
    # These defaults must match your training run ONLY if you do NOT save
    # them in the checkpoint.  When the checkpoint contains embed_dim,
    # num_layers and num_heads (as shown in the save snippet), PredictService
    # reads them directly from the file and these defaults are ignored.
    EMBED_DIM: int = 200
    NUM_LAYERS: int = 6
    NUM_HEADS: int = 4
    DROPOUT: float = 0.3

    # ConvE reshape  (EMBED_DIM must be divisible by CONV_DW)
    CONV_DW: int = 10

    @property
    def CONV_DH(self) -> int:
        return self.EMBED_DIM // self.CONV_DW

    # ── Runtime ─────────────────────────────────────────────
    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Singleton instance used across the app
cfg = Config()
