"""
GATCE Encoder: GATLayer and GATEncoder — rebuilt to match link-prediction-gatce-final.ipynb.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class GATLayer(MessagePassing):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__(aggr='add', flow='source_to_target', node_dim=0)
        self.dim      = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.dropout   = dropout

        # Projections for Query, Key, Value
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)

        # Attention scorer
        self.att_proj = nn.Linear(self.head_dim, 1)

        # Relation integration projection
        self.W_rel = nn.Linear(dim, dim)

        # Output transformations
        self.out_proj = nn.Linear(dim, dim)
        self.norm     = nn.LayerNorm(dim)
        self.act      = nn.LeakyReLU(0.2)
        self.drop     = nn.Dropout(dropout)

        # Saved attention weights for explainability (set during propagate)
        self._last_attn: torch.Tensor = None

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
        # Compute Q, K, V for all nodes  [N, heads, head_dim]
        Q = self.Wq(H).view(-1, self.num_heads, self.head_dim)
        K = self.Wk(H).view(-1, self.num_heads, self.head_dim)
        V = self.Wv(H).view(-1, self.num_heads, self.head_dim)

        # Relation embedding per edge  [E, heads, head_dim]
        r_emb = self.W_rel(R[edge_type]).view(-1, self.num_heads, self.head_dim)

        # Start PyG message passing — propagate calls message() then aggregates
        out = self.propagate(edge_index, Q=Q, K=K, V=V, r_emb=r_emb)

        # Final linear projection, dropout, residual + norm
        out = out.reshape(-1, self.dim)
        out = self.drop(self.out_proj(out))

        # Mean attention over heads → scalar per edge [E] for explainability
        attn_mean = self._last_attn  # stored inside message()

        return self.norm(H + self.act(out)), attn_mean

    def message(self, Q_i, K_j, V_j, r_emb, index, ptr, size_i):
        """
        Q_i : query at destination  [E, heads, head_dim]
        K_j : key at source         [E, heads, head_dim]
        V_j : value at source       [E, heads, head_dim]
        """
        # Incorporate relation embedding into Key
        K_combined = K_j + r_emb

        # Attention score
        attn_score = self.att_proj(Q_i * K_combined)        # [E, heads, 1]
        e = F.leaky_relu(attn_score, negative_slope=0.2)

        # Neighbourhood-normalised softmax (PyG)
        attn_weights = softmax(e, index, ptr, size_i)        # [E, heads, 1]
        attn_weights = F.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        # Store mean attention (over heads) for explainability
        self._last_attn = attn_weights.squeeze(-1).mean(dim=1).detach()  # [E]

        # Incorporate relation embedding into Value and weight by attention
        V_combined = V_j + r_emb
        return attn_weights * V_combined                     # [E, heads, head_dim]


class GATEncoder(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, cfg):
        super().__init__()
        self.ent = nn.Embedding(num_entities, cfg.EMBED_DIM)
        self.rel = nn.Embedding(num_relations, cfg.EMBED_DIM)

        self.layers = nn.ModuleList([
            GATLayer(cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.DROPOUT)
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
            H         : entity embeddings   [N, dim]
            R         : relation embeddings [R, dim]
            attn_maps : list[Tensor[E]] — one per layer (only if return_attention)
        """
        H = self.ent.weight
        R = self.rel.weight

        attn_maps = []
        for layer in self.layers:
            H, attn = layer(H, R, edge_index, edge_type)
            if return_attention:
                attn_maps.append(attn)

        if return_attention:
            return H, R, attn_maps
        return H, R
