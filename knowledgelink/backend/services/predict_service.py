"""
PredictService — loads the GATCE checkpoint and handles:
  • predict()  → top-K tail entities with confidence scores
  • explain()  → per-layer attention weights on the head entity's neighbors

Updated to match link-prediction-gatce-final.ipynb:
  • Checkpoint (best_gatce_model.pth) is a plain state_dict — no bundled metadata
  • Entity/relation mappings are loaded from a separate mappings.pkl file
  • num_relations passed to GATCE is original_num_relations * 2 (bidirectional graph)
"""
import os
import pickle
from typing import List, Dict, Any

import torch

from backend.model.gatce import GATCE
from torch_geometric.utils import k_hop_subgraph


class PredictService:
    _instance = None

    def __init__(self, cfg, data_svc):
        self.cfg      = cfg
        self.data_svc = data_svc
        self.device   = cfg.device
        self.model: GATCE = None
        self.active_model_name = "fb15k-237"

    # ── Singleton ────────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls, cfg=None, data_svc=None):
        if cls._instance is None:
            if cfg is None or data_svc is None:
                raise RuntimeError("PredictService not initialised yet.")
            cls._instance = cls(cfg, data_svc)
        return cls._instance

    # ── Model Management ─────────────────────────────────────────────────
    def list_models(self) -> List[str]:
        models = []
        if os.path.exists(self.cfg.models_dir):
            for item in os.listdir(self.cfg.models_dir):
                if os.path.isdir(os.path.join(self.cfg.models_dir, item)):
                    models.append(item)
        # Ensure fb15k-237 is first if it exists
        if "fb15k-237" in models:
            models.remove("fb15k-237")
            models.insert(0, "fb15k-237")
        return models

    def switch_model(self, model_name: str):
        if model_name == self.active_model_name and self.model is not None:
            return # Already loaded
            
        print(f"[PredictService] Switching model to: '{model_name}'")
        self.model = None # Clear old model
        self.active_model_name = model_name
        self.load_model()

    def load_model(self):
        if self.model is not None:
            return

        ds = self.data_svc
        
        # Determine paths based on active model
        model_dir = os.path.join(self.cfg.models_dir, self.active_model_name)
        mappings_path = os.path.join(model_dir, "mappings.pkl")
        ckpt_path = os.path.join(model_dir, "best_model.pth")

        # ── 1. Load mappings from separate pkl file ───────────────────────
        if os.path.exists(mappings_path):
            with open(mappings_path, "rb") as f:
                mappings = pickle.load(f)
            
            # Rebuild DataService structures dynamically based on new model
            ds.reload_for_model(self.active_model_name, mappings["ent2id"], mappings["rel2id"])
                 
            print(
                f"[PredictService] Mappings loaded from '{mappings_path}': "
                f"{ds.num_entities:,} entities, "
                f"{ds.original_num_relations:,} original relations "
                f"({ds.num_relations:,} with inverses)."
            )
        else:
            print(
                f"[PredictService] WARNING: '{mappings_path}' not found. "
                "Using mappings derived from data files."
            )

        # ── 2. Build model with correct dimensions ────────────────────────
        # num_entities  = from mappings
        # num_relations = original * 2  (bidirectional graph)
        self.model = GATCE(ds.num_entities, ds.num_relations, self.cfg).to(self.device)

        # ── 3. Load plain state_dict from checkpoint ──────────────────────
        state_dict = torch.load(
            ckpt_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(
            f"[PredictService] GATCE '{self.active_model_name}' loaded from '{ckpt_path}' — "
            f"dim={self.cfg.EMBED_DIM}, layers={self.cfg.NUM_LAYERS}, "
            f"heads={self.cfg.NUM_HEADS}."
        )

    @property
    def ready(self) -> bool:
        return self.model is not None

    # ── Predict ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, head_id: int, rel_id: int, topk: int = 10) -> List[Dict[str, Any]]:
        self.model.eval()  # Ensure eval mode to avoid BatchNorm errors on batch size 1
        ds     = self.data_svc
        device = self.device

        h = torch.tensor([head_id], device=device)
        r = torch.tensor([rel_id],  device=device)

        H, R = self.model.enc(ds.edge_index, ds.edge_type)
        scores = self.model.dec(H[h], R[r], H)[0]          # [num_entities]
        probs  = torch.sigmoid(scores).cpu().tolist()

        known = ds.hr2t.get((head_id, rel_id), set())

        # Take more candidates to account for filtering
        candidates = scores.topk(
            min(topk + len(known) + 10, ds.num_entities)
        ).indices.cpu().tolist()

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
        """
        Calculates Graph Attention Rollout for a 2-layer GATCE.
        - L2 Influence: How much a 1-hop neighbor V influences `head_id` directly in Layer 2.
        - L1 Influence: How much a 2-hop neighbor U influences V in Layer 1, scaled by V's importance to `head_id`.
        """
        self.model.eval()
        ds     = self.data_svc
        device = self.device

        h = torch.tensor([head_id], device=device)
        r = torch.tensor([rel_id],  device=device)

        # Get attention maps
        scores, H, R, attn_maps = self.model.forward_explain(
            ds.edge_index, ds.edge_type, h, r
        )

        L1_attn = attn_maps[0].cpu().tolist() if attn_maps else []
        L2_attn = attn_maps[-1].cpu().tolist() if len(attn_maps) > 1 else L1_attn

        edge_index = ds.edge_index.cpu()
        sources = edge_index[0].tolist()
        targets = edge_index[1].tolist()
        rel_types = ds.edge_type.cpu().tolist()

        # Build adjacency maps for fast lookup
        # target -> list of (source, edge_index, rel_id)
        incoming_edges = {}
        for idx, (s, t, rel) in enumerate(zip(sources, targets, rel_types)):
            if t not in incoming_edges:
                incoming_edges[t] = []
            incoming_edges[t].append((s, idx, rel))

        # --- Layer 2 Influence (1-hop) ---
        # Edges V -> head_id
        node_l2_inf = {}
        edge_relations = {}
        
        l2_edges = incoming_edges.get(head_id, [])
        for v, e_idx, rel in l2_edges:
            w2 = L2_attn[e_idx]
            node_l2_inf[v] = node_l2_inf.get(v, 0.0) + w2
            # Save the direct relation for display
            if v not in edge_relations or w2 > edge_relations[v][1]:
                edge_relations[v] = (rel, w2)

        # --- Layer 1 Influence (2-hop) ---
        # Edges U -> V -> head_id
        node_l1_inf = {}
        for v, e_idx_2, _ in l2_edges:
            w2 = L2_attn[e_idx_2]
            l1_edges = incoming_edges.get(v, [])
            for u, e_idx_1, rel in l1_edges:
                w1 = L1_attn[e_idx_1]
                path_weight = w1 * w2
                node_l1_inf[u] = node_l1_inf.get(u, 0.0) + path_weight
                
                # If U is purely a 2-hop neighbor, we can assign a representative relation
                if u not in edge_relations or path_weight > edge_relations[u][1]:
                    edge_relations[u] = (rel, path_weight)

        # --- Aggregate Influence ---
        # Normalize maps to sum to 1 so they are comparable
        sum_l1 = sum(node_l1_inf.values()) or 1.0
        sum_l2 = sum(node_l2_inf.values()) or 1.0

        all_nodes = set(node_l1_inf.keys()).union(set(node_l2_inf.keys()))
        
        node_scores = []
        for n in all_nodes:
            i1 = node_l1_inf.get(n, 0.0) / float(sum_l1)
            i2 = node_l2_inf.get(n, 0.0) / float(sum_l2)
            total_inf = i1 + i2
            node_scores.append((n, total_inf, i1, i2))

        # Rank and take top
        node_scores.sort(key=lambda x: x[1], reverse=True)
        top_nodes = node_scores[:max_neighbors]

        def rel_display(rid: int) -> str:
            if rid < ds.original_num_relations:
                return ds.id2rel[rid]
            base = ds.id2rel.get(rid - ds.original_num_relations, f"rel_{rid}")
            return f"{base} (inverse)"

        neighbors_info = []
        attention_data = []

        for n, imp, i1, i2 in top_nodes:
            rel = edge_relations.get(n, (0, 0))[0]
            neighbors_info.append({
                "entity_id":     n,
                "entity_name":   ds.id2ent[n],
                "relation_name": rel_display(rel),
                "importance":    round(imp, 5),
            })
            attention_data.append([round(i1, 5), round(i2, 5)])

        return {
            "head":           {"id": head_id, "name": ds.id2ent[head_id]},
            "relation":       ds.id2rel.get(rel_id, f"rel_{rel_id}"),
            "predicted_tail": {"id": tail_id, "name": ds.id2ent[tail_id]},
            "neighbors":      neighbors_info,
            "attention":      attention_data
        }
