"""
DataService — loads FB15k-237 triples, builds entity/relation mappings,
optionally enriches entity names via fb_wiki_mapping.tsv, and provides
graph query helpers (search, neighborhood).

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
        self.num_entities: int = 0
        self.num_relations: int = 0

        self.train_ids: List = []
        self.valid_ids: List = []
        self.test_ids: List  = []

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
    def load(self):
        if self._loaded:
            return

        data_dir = self.cfg.data_dir
        cols = ["head", "relation", "tail"]

        df_train = pd.read_csv(f"{data_dir}/train.txt", sep="\t", header=None, names=cols)
        df_valid = pd.read_csv(f"{data_dir}/valid.txt", sep="\t", header=None, names=cols)
        df_test  = pd.read_csv(f"{data_dir}/test.txt",  sep="\t", header=None, names=cols)

        entities = (
            set(df_train["head"]) | set(df_train["tail"]) |
            set(df_valid["head"]) | set(df_valid["tail"]) |
            set(df_test["head"])  | set(df_test["tail"])
        )
        relations = (
            set(df_train["relation"]) | set(df_valid["relation"]) | set(df_test["relation"])
        )

        self.ent2id = {e: i for i, e in enumerate(sorted(entities))}
        self.id2ent = {i: e for e, i in self.ent2id.items()}
        self.rel2id = {r: i for i, r in enumerate(sorted(relations))}
        self.id2rel = {i: r for r, i in self.rel2id.items()}
        self.num_entities  = len(self.ent2id)
        self.num_relations = len(self.rel2id)

        def to_ids(df):
            return [
                (self.ent2id[row["head"]], self.rel2id[row["relation"]], self.ent2id[row["tail"]])
                for _, row in df.iterrows()
            ]

        self.train_ids = to_ids(df_train)
        self.valid_ids = to_ids(df_valid)
        self.test_ids  = to_ids(df_test)

        # Graph tensors
        device = self.cfg.device
        src_list = [h for h, r, t in self.train_ids]
        dst_list = [t for h, r, t in self.train_ids]
        rel_list = [r for h, r, t in self.train_ids]

        self.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
        self.edge_type  = torch.tensor(rel_list,             dtype=torch.long, device=device)

        # Adjacency for neighbourhood queries
        for h, r, t in self.train_ids:
            self.adj[h].append((r, t))

        # Filtered-ranking helper
        for h, r, t in self.train_ids + self.valid_ids + self.test_ids:
            self.hr2t[(h, r)].add(t)

        self._loaded = True
        print(
            f"[DataService] Loaded {self.num_entities:,} entities "
            f"and {self.num_relations:,} relations."
        )

        # Apply human-readable names if mapping file exists
        mapping_path = f"{self.cfg.data_dir}/fb_wiki_mapping.tsv"
        if os.path.exists(mapping_path):
            self._load_wiki_mapping(mapping_path)
        else:
            print("[DataService] No fb_wiki_mapping.tsv found — using raw Freebase IDs.")

    # ── Wiki mapping (human-readable names) ──────────────────────────────
    def _load_wiki_mapping(self, path: str):
        """
        Read fb_wiki_mapping.tsv (freebase_id, wikidata_id, label) and
        override id2ent display names with the human-readable label.
        Freebase IDs in ent2id are preserved so model indices stay correct.
        """
        df = pd.read_csv(path, sep="\t", header=0,
                         names=["freebase_id", "wikidata_id", "label"],
                         usecols=["freebase_id", "label"])
        self._fb2name = dict(zip(df["freebase_id"], df["label"]))

        # Overwrite id → display name for every entity that has a mapping
        mapped = 0
        for fid, eid in self.ent2id.items():
            label = self._fb2name.get(fid)
            if label:
                self.id2ent[eid] = label
                mapped += 1

        print(f"[DataService] Applied readable names for {mapped:,} / {self.num_entities:,} entities.")

    # ── Checkpoint seeding ────────────────────────────────────────────────
    def seed_from_checkpoint(self, ent2id: Dict[str, int], rel2id: Dict[str, int]):
        """
        Override entity/relation mappings with those saved in the checkpoint.
        Re-applies the wiki mapping if it was already loaded.
        """
        self.ent2id = ent2id
        self.id2ent = {i: e for e, i in ent2id.items()}
        self.rel2id = rel2id
        self.id2rel = {i: r for r, i in rel2id.items()}
        self.num_entities  = len(ent2id)
        self.num_relations = len(rel2id)

        # Re-apply readable names if already loaded
        if self._fb2name:
            mapped = 0
            for fid, eid in self.ent2id.items():
                label = self._fb2name.get(fid)
                if label:
                    self.id2ent[eid] = label
                    mapped += 1
            print(f"[DataService] Re-applied readable names for {mapped:,} entities after checkpoint seed.")


    def search_entities(self, query: str, topk: int = 15) -> List[Dict[str, Any]]:
        """
        Case-insensitive substring match against readable display names.
        After wiki mapping is applied, id2ent holds human-readable labels
        (e.g. 'France', 'Christopher Nolan') instead of raw Freebase IDs.
        """
        q = query.lower()
        results = [
            {"id": eid, "name": name}
            for eid, name in self.id2ent.items()
            if q in name.lower()
        ]
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
        return [{"id": rid, "name": self.id2rel[rid]} for rid in sorted(self.id2rel)]
