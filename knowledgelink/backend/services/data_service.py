"""
DataService — loads FB15k-237 triples, builds entity/relation mappings,
optionally enriches entity names via entities_labels.tsv, and provides
graph query helpers (search, neighborhood).

Updated to match link-prediction-gatce-final.ipynb:
  • Bidirectional graph: for each (h, r, t) an inverse edge (t, r_inv, h) is added
  • num_relations is doubled (original_num_relations * 2) to accommodate inverse rels
  • hr2t includes both forward and inverse lookups (for filtered ranking compatibility)

This is a lazy singleton: call .load() once at startup.
"""
import os
from collections import defaultdict
from typing import List, Dict, Any

import pandas as pd
import torch


class DataService:
    _instance = None

    def __init__(self, cfg):
        self.cfg = cfg
        self._loaded = False
        # Optional human-readable label lookup: freebase_id -> display name
        self._fb2name: Dict[str, str] = {}

        # Populated by load()
        self.ent2id: Dict[str, int] = {}
        self.id2ent: Dict[int, str] = {}
        self.rel2id: Dict[str, int] = {}
        self.id2rel: Dict[int, str] = {}
        self.num_entities: int        = 0
        self.num_relations: int       = 0           # original count (before doubling)
        self.original_num_relations: int = 0

        self.train_ids: List = []           # (h_id, r_id, t_id) — rebuilt after seed
        self.train_triples: List = []       # (h_str, r_str, t_str) — raw strings preserved

        self.edge_index: torch.Tensor = None
        self.edge_type: torch.Tensor  = None

        self.adj: Dict[int, List]     = defaultdict(list)   # h -> [(r, t)]
        self.hr2t: Dict               = defaultdict(set)    # (h,r) -> {t}

    # ── Singleton ────────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls, cfg=None):
        if cls._instance is None:
            if cfg is None:
                raise RuntimeError("DataService not initialised yet.")
            cls._instance = cls(cfg)
        return cls._instance

    # ── Load ─────────────────────────────────────────────────────────────
    def load(self, model_name: str = "fb15k-237"):
        if self._loaded:
            return

        data_dir = os.path.join(self.cfg.models_dir, model_name, "data")
        cols = ["head", "relation", "tail"]

        try:
            df_train = pd.read_csv(f"{data_dir}/train.txt", sep="\t", header=None, names=cols)
        except Exception as e:
            print(f"[DataService] Could not load train.txt from {data_dir}: {e}")
            return

        entities  = set(df_train["head"]) | set(df_train["tail"])
        relations = set(df_train["relation"])

        self.ent2id = {e: i for i, e in enumerate(sorted(entities))}
        self.id2ent = {i: e for e, i in self.ent2id.items()}
        self.rel2id = {r: i for i, r in enumerate(sorted(relations))}
        self.id2rel = {i: r for r, i in self.rel2id.items()}
        self.num_entities           = len(self.ent2id)
        self.original_num_relations = len(self.rel2id)
        self.num_relations          = self.original_num_relations * 2

        # Store raw triples as strings so we can re-index after checkpoint seed
        self.train_triples = [
            (row["head"], row["relation"], row["tail"])
            for _, row in df_train.iterrows()
        ]

        self.train_ids = [
            (self.ent2id[h], self.rel2id[r], self.ent2id[t])
            for h, r, t in self.train_triples
        ]

        self._build_graph_structures()

        self._loaded = True
        print(
            f"[DataService] Loaded {self.num_entities:,} entities, "
            f"{self.original_num_relations:,} original relations "
            f"({self.num_relations:,} with inverses). "
            f"Graph edges: {self.edge_index.shape[1]:,}."
        )

        # Apply human-readable names if mapping file exists
        mapping_path = os.path.join(data_dir, "entities_labels.tsv")
        if os.path.exists(mapping_path):
            self._load_wiki_mapping(mapping_path)
        else:
            print("[DataService] No entities_labels.tsv found — using raw Freebase IDs.")

    # ── Wiki mapping (human-readable names) ──────────────────────────────
    def _load_wiki_mapping(self, path: str):
        """
        Read entity mapping file (ID, Name, ...) and override id2ent display names.
        Always uses the first two columns. Ignores additional columns and headers.
        """
        try:
            # Read first two columns, no header assumed initially
            df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1], engine="python")
            
            # If the first row looks like a header (contains 'id', 'name', 'label' etc.), skip it
            first_col_val = str(df.iloc[0, 0]).lower()
            if first_col_val in ["id", "entity", "freebase_id", "guid", "entity_id"]:
                df = df.iloc[1:].reset_index(drop=True)
            
            # Ensure missing labels become empty strings
            df[1] = df[1].fillna("").astype(str)
            self._fb2name = dict(zip(df[0], df[1]))

            # Overwrite id → display name for every entity that has a mapping
            mapped = 0
            for fid, eid in self.ent2id.items():
                label = self._fb2name.get(fid)
                if label:
                    self.id2ent[eid] = label
                    mapped += 1

            print(f"[DataService] Applied readable names for {mapped:,} / {self.num_entities:,} entities from {os.path.basename(path)}.")
        except Exception as e:
            print(f"[DataService] Failed to load mapping from {path}: {str(e)}")

    # ── Graph structure builder ──────────────────────────────────────────
    def _build_graph_structures(self):
        """
        (Re)build edge_index, edge_type, adj, hr2t from self.train_triples
        using the current ent2id / rel2id / original_num_relations.
        Triples whose head, relation, or tail are not in the current mappings
        are silently skipped.
        """
        device   = self.cfg.device
        src_list = []
        dst_list = []
        rel_list = []

        self.adj   = defaultdict(list)
        self.hr2t  = defaultdict(set)
        self.train_ids = []

        for h_str, r_str, t_str in self.train_triples:
            h = self.ent2id.get(h_str)
            r = self.rel2id.get(r_str)
            t = self.ent2id.get(t_str)
            if h is None or r is None or t is None:
                continue

            self.train_ids.append((h, r, t))

            # Forward edge
            src_list.append(h); dst_list.append(t); rel_list.append(r)
            # Inverse edge
            r_inv = r + self.original_num_relations
            src_list.append(t); dst_list.append(h); rel_list.append(r_inv)

            # Adjacency (forward only, for neighbourhood display)
            self.adj[h].append((r, t))

            # Filtered-ranking helper
            self.hr2t[(h, r)].add(t)
            self.hr2t[(t, r_inv)].add(h)

        self.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
        self.edge_type  = torch.tensor(rel_list,             dtype=torch.long, device=device)

    # ── Checkpoint seeding ────────────────────────────────────────────────
    def seed_from_checkpoint(self, ent2id: Dict[str, int], rel2id: Dict[str, int]):
        """
        Override entity/relation mappings with those saved in the checkpoint pickle,
        then rebuild all graph structures (adj, edge_index, edge_type, hr2t) so that
        entity IDs are consistent throughout.
        Note: rel2id here contains original (non-doubled) relations; num_relations is
        set to len(rel2id) * 2 to match the bidirectional graph.
        """
        self.ent2id = ent2id
        self.id2ent = {i: e for e, i in ent2id.items()}
        self.rel2id = rel2id
        self.id2rel = {i: r for r, i in rel2id.items()}
        self.num_entities           = len(ent2id)
        self.original_num_relations = len(rel2id)
        self.num_relations          = len(rel2id) * 2

        # Rebuild adj / edge_index / edge_type / hr2t with new IDs
        self._build_graph_structures()
        print(f"[DataService] Graph structures rebuilt after checkpoint seed: "
              f"{self.edge_index.shape[1]:,} edges.")

        # Re-apply readable names if already loaded
        if self._fb2name:
            mapped = 0
            for fid, eid in self.ent2id.items():
                label = self._fb2name.get(fid)
                if label:
                    self.id2ent[eid] = label
                    mapped += 1
            print(f"[DataService] Re-applied readable names for {mapped:,} entities after checkpoint seed.")

    def reload_for_model(self, model_name: str, ent2id: Dict[str, int], rel2id: Dict[str, int]):
        """
        When switching to a user-trained model, we need to load its specific train.txt
        triples to construct the correct graph for inference.
        """
        model_data_dir = os.path.join(self.cfg.models_dir, model_name, "data")
        train_path = os.path.join(model_data_dir, "train.txt")
        
        # Reload actual triples for this new model
        try:
            df_train = pd.read_csv(train_path, sep="\t", header=None, names=["head", "relation", "tail"])
            self.train_triples = [(row["head"], row["relation"], row["tail"]) for _, row in df_train.iterrows()]
        except Exception as e:
            print(f"[DataService] Error reloading train.txt for model {model_name}: {e}")

        # Check for model-specific entity names mapping
        mapping_path = os.path.join(model_data_dir, "entities_labels.tsv")
        self._fb2name = {} # Reset mapping
        if os.path.exists(mapping_path):
            self._load_wiki_mapping(mapping_path)
            print(f"[DataService] Loaded model-specific mapping from {mapping_path}")
        else:
            # If no model-specific mapping, check the default one
            global_mapping_path = os.path.join(self.cfg.models_dir, "fb15k-237", "data", "entities_labels.tsv")
            if os.path.exists(global_mapping_path):
                self._load_wiki_mapping(global_mapping_path)

        self.seed_from_checkpoint(ent2id, rel2id)

    def search_entities(self, query: str, topk: int = 15) -> List[Dict[str, Any]]:
        """
        Case-insensitive substring match against readable display names.
        After wiki mapping is applied, id2ent holds human-readable labels
        (e.g. 'France', 'Christopher Nolan') instead of raw Freebase IDs.
        """
        q = query.lower()
        
        starts_with = []
        contains = []
        
        for eid, name in self.id2ent.items():
            name_str = str(name)
            name_lower = name_str.lower()
            if name_lower.startswith(q):
                starts_with.append({"id": eid, "name": name_str})
            elif q in name_lower:
                contains.append({"id": eid, "name": name_str})
                
        # Optional: alphabetize both groups primarily to look elegant 
        starts_with.sort(key=lambda x: x["name"].lower())
        contains.sort(key=lambda x: x["name"].lower())
        
        results = starts_with + contains
        return results[:topk]

    def get_neighbors(self, entity_id: int, max_neighbors: int = 50) -> List[Dict[str, Any]]:
        """Return 1-hop outgoing edges for an entity."""
        neighbors = []
        for rel_id, tail_id in self.adj.get(entity_id, [])[:max_neighbors]:
            neighbors.append({
                "source":        entity_id,
                "source_name":   self.id2ent[entity_id],
                "relation_id":   rel_id,
                "relation_name": self.id2rel[rel_id],
                "target":        tail_id,
                "target_name":   self.id2ent[tail_id],
            })
        return neighbors

    def get_all_relations(self) -> List[Dict[str, Any]]:
        """Return only original (non-inverse) relations for the UI."""
        return [
            {"id": rid, "name": self.id2rel[rid]}
            for rid in sorted(self.id2rel)
            if rid < self.original_num_relations
        ]
