"""
PredictService — loads the GATH checkpoint and handles:
  • predict()  → top-K tail entities with confidence scores
  • explain()  → per-layer attention weights on the head entity's neighbors
"""
import os
from typing import List, Dict, Any

import torch

from backend.model.gath import GATH


class PredictService:
    _instance = None

    def __init__(self, cfg, data_svc):
        self.cfg      = cfg
        self.data_svc = data_svc
        self.device   = cfg.device
        self.model: GATH = None

    # ── Singleton ────────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls, cfg=None, data_svc=None):
        if cls._instance is None:
            if cfg is None or data_svc is None:
                raise RuntimeError("PredictService not initialised yet.")
            cls._instance = cls(cfg, data_svc)
        return cls._instance

    # ── Model loading ────────────────────────────────────────────────────
    def load_model(self):
        if self.model is not None:
            return

        ckpt = torch.load(self.cfg.checkpoint_path, map_location=self.device, weights_only=False)

        # ── 1. Override hyperparams from checkpoint ───────────────────────
        # Values saved at training time are authoritative — no need to edit config.py manually.
        if "embed_dim"  in ckpt: self.cfg.EMBED_DIM  = ckpt["embed_dim"]
        if "num_layers" in ckpt: self.cfg.NUM_LAYERS = ckpt["num_layers"]
        if "num_heads"  in ckpt: self.cfg.NUM_HEADS  = ckpt["num_heads"]

        # ── 2. Seed entity / relation mappings from checkpoint ────────────
        # This means data/ files are only needed for the adjacency graph;
        # the vocabulary is guaranteed to match training exactly.
        ds = self.data_svc
        if "ent2id" in ckpt and "rel2id" in ckpt:
            ds.seed_from_checkpoint(ckpt["ent2id"], ckpt["rel2id"])
            print(
                f"[PredictService] Mappings loaded from checkpoint: "
                f"{ds.num_entities:,} entities, {ds.num_relations:,} relations."
            )

        # ── 3. Build model with correct dimensions ────────────────────────
        self.model = GATH(ds.num_entities, ds.num_relations, self.cfg).to(self.device)

        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state)
        self.model.eval()
        print(
            f"[PredictService] GATH loaded — "
            f"dim={self.cfg.EMBED_DIM}, layers={self.cfg.NUM_LAYERS}, heads={self.cfg.NUM_HEADS}."
        )

    @property
    def ready(self) -> bool:
        return self.model is not None

    # ── Predict ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, head_id: int, rel_id: int, topk: int = 10) -> List[Dict[str, Any]]:
        ds     = self.data_svc
        device = self.device

        h = torch.tensor([head_id], device=device)
        r = torch.tensor([rel_id],  device=device)

        H, R = self.model.enc(ds.edge_index, ds.edge_type)
        scores = self.model.dec(H[h], R[r], H)[0]          # [num_entities]
        probs  = torch.sigmoid(scores).cpu().tolist()

        known = ds.hr2t.get((head_id, rel_id), set())

        # Take more candidates to account for filtering
        candidates = scores.topk(min(topk + len(known) + 10, ds.num_entities)).indices.cpu().tolist()

        results = []
        for tid in candidates:
            results.append({
                "entity_id":   tid,
                "entity_name": ds.id2ent[tid],
                "score":       round(probs[tid], 5),
                "is_known":    tid in known,
            })
            if len(results) >= topk:
                break

        return results

    # ── Explain ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def explain(
        self,
        head_id: int,
        rel_id: int,
        tail_id: int,
        max_neighbors: int = 25,
    ) -> Dict[str, Any]:
        ds     = self.data_svc
        device = self.device

        h = torch.tensor([head_id], device=device)
        r = torch.tensor([rel_id],  device=device)

        scores, H, R, attn_maps = self.model.forward_explain(
            ds.edge_index, ds.edge_type, h, r
        )

        # ── Find edges that flow INTO head_id (dst == head_id) ────────────
        dst_mask   = (ds.edge_index[1] == head_id)
        edge_idxs  = dst_mask.nonzero(as_tuple=True)[0].cpu().tolist()[:max_neighbors]
        src_ids    = ds.edge_index[0][edge_idxs].cpu().tolist()
        rel_ids    = ds.edge_type[edge_idxs].cpu().tolist()

        # ── Per-layer attention for those edges ───────────────────────────
        # attn_maps: list[Tensor[E]] – one tensor per layer
        layer_attns: List[List[float]] = []
        for layer_attn in attn_maps:
            vals = [float(layer_attn[eidx].item()) for eidx in edge_idxs]
            layer_attns.append(vals)

        # ── Aggregate across layers for a summary importance score ────────
        num_layers  = len(attn_maps)
        aggregated  = [0.0] * len(edge_idxs)
        for layer_vals in layer_attns:
            for i, v in enumerate(layer_vals):
                aggregated[i] += v
        if num_layers > 0:
            aggregated = [v / num_layers for v in aggregated]

        # Sort by importance descending
        ranked = sorted(
            zip(src_ids, rel_ids, aggregated),
            key=lambda x: x[2], reverse=True
        )

        neighbors_info = [
            {
                "entity_id":     src,
                "entity_name":   ds.id2ent[src],
                "relation_name": ds.id2rel[rel],
                "importance":    round(imp, 5),
            }
            for src, rel, imp in ranked
        ]

        return {
            "head":             {"id": head_id, "name": ds.id2ent[head_id]},
            "relation":         ds.id2rel[rel_id],
            "predicted_tail":   {"id": tail_id, "name": ds.id2ent[tail_id]},
            "neighbors":        neighbors_info,
            "attention_layers": layer_attns,    # [num_layers][num_neighbors] raw
            "num_layers":       num_layers,
            "num_neighbors":    len(edge_idxs),
        }
