"""
GATCE model: wires GATEncoder + ConvEDecoder together.
Exposes both a standard forward() and a forward_explain() that
returns per-layer attention maps for the explainability endpoints.
"""
import torch
import torch.nn as nn

from .encoder import GATEncoder
from .decoder import ConvEDecoder


class GATCE(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, cfg):
        super().__init__()
        self.enc = GATEncoder(num_entities, num_relations, cfg)
        self.dec = ConvEDecoder(num_entities, cfg)   # num_entities needed for entity_bias

    # ── Standard inference ───────────────────────────────────────────────
    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        h_idx: torch.Tensor,
        r_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Returns score logits [B, num_entities]."""
        H, R = self.enc(edge_index, edge_type)
        return self.dec(H[h_idx], R[r_idx], H)

    # ── Explainability inference ─────────────────────────────────────────
    def forward_explain(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        h_idx: torch.Tensor,
        r_idx: torch.Tensor,
    ):
        """
        Returns:
            scores    : [B, num_entities]
            H         : full entity embeddings after encoding
            R         : relation embeddings
            attn_maps : list[Tensor[E]] – one per GATLayer
        """
        H, R, attn_maps = self.enc(edge_index, edge_type, return_attention=True)
        scores = self.dec(H[h_idx], R[r_idx], H)
        return scores, H, R, attn_maps


