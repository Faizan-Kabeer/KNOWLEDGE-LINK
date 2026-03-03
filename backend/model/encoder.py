"""
GATH Encoder: GATHLayer and GATHEncoder.

GATHLayer is modified to return per-edge attention weights so the
explainability service can surface which neighbors drove each prediction.
"""
import torch
import torch.nn as nn


class GATHLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Projections
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)

        # Attention scorer
        self.att_proj = nn.Linear(self.head_dim, 1)

        # Relation integration
        self.W_rel = nn.Linear(dim, dim)

        # Output
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        H: torch.Tensor,
        R: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ):
        """
        Returns:
            H_out  : updated node embeddings  [N, dim]
            attn_w : mean attention per edge  [E]      (for explainability)
        """
        src, dst = edge_index

        # [N, heads, head_dim]
        Q = self.Wq(H).view(-1, self.num_heads, self.head_dim)
        K = self.Wk(H).view(-1, self.num_heads, self.head_dim)
        V = self.Wv(H).view(-1, self.num_heads, self.head_dim)

        K_src = K[src]          # [E, heads, head_dim]
        V_src = V[src]
        Q_dst = Q[dst]

        # Relation embedding per edge
        r_emb = self.W_rel(R[edge_type]).view(-1, self.num_heads, self.head_dim)

        # Attention: Hadamard score then sigmoid to [0, 1]
        K_combined = K_src + r_emb
        attn_score = self.att_proj(Q_dst * K_combined)   # [E, heads, 1]
        attn_weights = torch.sigmoid(attn_score)          # [E, heads, 1]

        # Messages
        V_combined = V_src + r_emb
        message = attn_weights * V_combined               # [E, heads, head_dim]

        # Scatter-add messages to destination nodes
        out = torch.zeros_like(Q)
        out.index_add_(0, dst, message)

        out = out.reshape(-1, self.dim)
        out = self.dropout(self.out_proj(out))

        # Mean attention over heads → scalar per edge for explainability
        attn_mean = attn_weights.squeeze(-1).mean(dim=1)  # [E]

        return self.norm(H + self.act(out)), attn_mean


class GATHEncoder(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, cfg):
        super().__init__()
        self.ent = nn.Embedding(num_entities, cfg.EMBED_DIM)
        self.rel = nn.Embedding(num_relations, cfg.EMBED_DIM)

        self.layers = nn.ModuleList([
            GATHLayer(cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.DROPOUT)
            for _ in range(cfg.NUM_LAYERS)
        ])

        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Args:
            return_attention: if True, also returns per-layer attention maps

        Returns:
            H            : entity embeddings  [N, dim]
            R            : relation embeddings [R, dim]
            attn_maps    : list[Tensor[E]] – one per layer (only if return_attention)
        """
        H = self.ent.weight
        R = self.rel.weight

        attn_maps = []
        for layer in self.layers:
            H, attn = layer(H, R, edge_index, edge_type)
            if return_attention:
                attn_maps.append(attn.detach())

        if return_attention:
            return H, R, attn_maps
        return H, R
