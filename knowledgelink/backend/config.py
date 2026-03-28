"""
Central configuration for KnowledgeLink.
All hyperparameters here MUST match what was used during training.
"""
import os
import torch

# Root of the knowledgelink/ project — two levels up from this file
#   backend/config.py → backend/ → knowledgelink/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    models_dir: str      = os.path.join(_PROJECT_ROOT, "models")

    # ── Model (must match training hyperparameters) ─────────
    EMBED_DIM: int  = 200
    NUM_LAYERS: int = 2
    NUM_HEADS: int  = 4
    DROPOUT: float  = 0.1

    # ConvE reshape (EMBED_DIM must be divisible by CONV_DW)
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
