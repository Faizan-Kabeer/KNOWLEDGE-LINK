"""
Modal-based remote GPU training service for KnowledgeLink.

This module defines a Modal serverless app that:
1. Provisions an A10G GPU container with the full PyTorch + PyG stack.
2. Accepts raw knowledge graph triples as a string payload.
3. Trains a GATCE model for 10 epochs, yielding live progress.
4. Returns the trained model weights and entity/relation mappings as bytes.
"""
import os
from typing import AsyncGenerator

import modal

# ── Modal App + Image ─────────────────────────────────────────────────────────
# The image installs the exact packages needed for the training environment.
# This image is built once and cached in Modal's infrastructure.

app = modal.App("knowledgelink-gatce")

# Simple image: torch from CUDA wheels, then torch_geometric from PyPI.
# The optional torch_scatter/torch_sparse extensions are NOT required for
# MessagePassing + softmax which is all our GATCE model uses.
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install("torch_geometric", "pandas", "numpy")
)


# ── The Remote Training Function ──────────────────────────────────────────────
@app.function(
    image=training_image,
    gpu="A10G",
    timeout=3600,      # 1-hour max (plenty for 10 epochs)
)
def run_training(
    triples_txt: str,
    embed_dim: int = 200,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    conv_dw: int = 10,
    epochs: int = 2,
    batch_size: int = 1024,
):
    """
    Trains a GATCE model on the provided triples string.
    Yields JSON-serializable progress dicts while running.
    Returns {"model_bytes": ..., "mappings_bytes": ...} when done.
    """
    import io
    import json
    import pickle
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import softmax
    import pandas as pd

    # ── 1. Parse data ─────────────────────────────────────────────────────
    print("Device:", "GPU" if torch.cuda.is_available() else "CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from io import StringIO
    df = pd.read_csv(StringIO(triples_txt), sep="\t", header=None,
                     names=["head", "relation", "tail"], engine="python")

    entities = sorted(set(df["head"]) | set(df["tail"]))
    relations = sorted(set(df["relation"]))

    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}
    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}

    original_num_relations = len(rel2id)
    num_relations = original_num_relations * 2
    num_entities = len(ent2id)
    CONV_DH = embed_dim // conv_dw

    print(f"Entities: {num_entities}, Relations: {original_num_relations}")

    # Build triples list & graph edges (with inverse relations)
    triples_ids = [
        (ent2id[r["head"]], rel2id[r["relation"]], ent2id[r["tail"]])
        for _, r in df.iterrows()
        if r["head"] in ent2id and r["tail"] in ent2id and r["relation"] in rel2id
    ]

    src, dst, rel = [], [], []
    for h, r, t in triples_ids:
        src.append(h); dst.append(t); rel.append(r)
        src.append(t); dst.append(h); rel.append(r + original_num_relations)

    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_type = torch.tensor(rel, dtype=torch.long, device=device)

    # ── 2. Model Definitions ──────────────────────────────────────────────
    class GATLayer(MessagePassing):
        def __init__(self, dim, n_heads, drop):
            super().__init__(aggr="add", flow="source_to_target", node_dim=0)
            self.dim = dim
            self.num_heads = n_heads
            self.head_dim = dim // n_heads
            self.dropout = drop
            self.Wq = nn.Linear(dim, dim)
            self.Wk = nn.Linear(dim, dim)
            self.Wv = nn.Linear(dim, dim)
            self.att_proj = nn.Linear(self.head_dim, 1)
            self.W_rel = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.LeakyReLU(0.2)
            self.drop = nn.Dropout(drop)

        def forward(self, H, R, edge_index, edge_type):
            Q = self.Wq(H).view(-1, self.num_heads, self.head_dim)
            K = self.Wk(H).view(-1, self.num_heads, self.head_dim)
            V = self.Wv(H).view(-1, self.num_heads, self.head_dim)
            r_emb = self.W_rel(R[edge_type]).view(-1, self.num_heads, self.head_dim)
            out = self.propagate(edge_index, Q=Q, K=K, V=V, r_emb=r_emb)
            out = out.reshape(-1, self.dim)
            out = self.drop(self.out_proj(out))
            return self.norm(H + self.act(out)), None

        def message(self, Q_i, K_j, V_j, r_emb, index, ptr, size_i):
            K_comb = K_j + r_emb
            e = F.leaky_relu(self.att_proj(Q_i * K_comb), 0.2)
            a = softmax(e, index, ptr, size_i)
            a = F.dropout(a, p=self.dropout, training=self.training)
            return a * (V_j + r_emb)

    class GATEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.ent = nn.Embedding(num_entities, embed_dim)
            self.rel = nn.Embedding(num_relations, embed_dim)
            self.layers = nn.ModuleList(
                [GATLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)]
            )
            nn.init.xavier_uniform_(self.ent.weight)
            nn.init.xavier_uniform_(self.rel.weight)

        def forward(self, ei, et):
            H, R = self.ent.weight, self.rel.weight
            for layer in self.layers:
                H, _ = layer(H, R, ei, et)
            return H, R

    class ConvEDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 32, (3, 3), padding=0)
            self.bn0 = nn.BatchNorm2d(1)
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm1d(embed_dim)
            self.input_drop = nn.Dropout(0.2)
            self.feature_map_drop = nn.Dropout2d(0.2)
            self.hidden_drop = nn.Dropout(0.3)
            h_out = 2 * CONV_DH - 2
            w_out = conv_dw - 2
            self.fc = nn.Linear(32 * h_out * w_out, embed_dim)
            self.entity_bias = nn.Parameter(torch.zeros(num_entities))

        def forward(self, h, r, E):
            h = h.view(-1, 1, CONV_DH, conv_dw)
            r = r.view(-1, 1, CONV_DH, conv_dw)
            x = self.bn0(torch.cat([h, r], 2))
            x = self.conv(self.input_drop(x))
            x = self.feature_map_drop(F.relu(self.bn1(x)))
            x = F.relu(self.bn2(self.hidden_drop(self.fc(x.flatten(1)))))
            return (x @ E.t()) + self.entity_bias

    class GATCE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = GATEncoder()
            self.dec = ConvEDecoder()

        def forward(self, ei, et, h_idx, r_idx):
            H, R = self.enc(ei, et)
            return self.dec(H[h_idx], R[r_idx], H)

    # ── 3. Training Loop ──────────────────────────────────────────────────
    class TripleDataset(Dataset):
        def __init__(self, triples): self.triples = triples
        def __len__(self): return len(self.triples)
        def __getitem__(self, i): return self.triples[i]

    def dropout_edge(ei, p=0.1):
        mask = torch.rand(ei.size(1), device=ei.device) >= p
        return ei[:, mask], mask

    model = GATCE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    loader = DataLoader(TripleDataset(triples_ids), batch_size=batch_size, shuffle=True)

    print("Beginning Training Loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            h_idx, r_idx, targets = batch
            h_idx = h_idx.to(device)
            r_idx = r_idx.to(device)
            targets = targets.to(device)

            drop_ei, mask = dropout_edge(edge_index)
            drop_et = edge_type[mask]

            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    H, R = model.enc(drop_ei, drop_et)
                    scores = model.dec(H[h_idx], R[r_idx], H)
                    loss = loss_fn(scores, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                H, R = model.enc(drop_ei, drop_et)
                scores = model.dec(H[h_idx], R[r_idx], H)
                loss = loss_fn(scores, targets)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        # Yield progress to the caller
        yield {"epoch": epoch, "total_epochs": epochs, "loss": round(avg_loss, 4)}
        print(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}")

    # ── 4. Serialize and return model weights + mappings ──────────────────
    print("Training complete. Serializing model...")
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    model_bytes = buf.getvalue()

    mappings = {"ent2id": ent2id, "rel2id": rel2id,
                "id2ent": id2ent, "id2rel": id2rel}
    mappings_bytes = pickle.dumps(mappings)

    yield {"done": True, "model_bytes": model_bytes, "mappings_bytes": mappings_bytes}


# ── Local Orchestrator ────────────────────────────────────────────────────────
async def train_on_modal(
    triples_txt: str,
    model_name: str,
    models_dir: str,
) -> AsyncGenerator[str, None]:
    """
    Async generator that streams SSE-formatted JSON progress events
    from the Modal remote function, then saves the model files locally.
    """
    import json

    yield f'data: {json.dumps({"status": "Provisioning GPU container on Modal..."})}\n\n'

    model_dir = os.path.join(models_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    try:
        # Enable Modal logs in the local terminal
        modal.enable_output()
        
        # Use Modal's native async support to stream training progress directly.
        async with app.run():
            model_bytes = None
            mappings_bytes = None

            async for item in run_training.remote_gen.aio(triples_txt):
                if "done" in item and item["done"]:
                    model_bytes = item["model_bytes"]
                    mappings_bytes = item["mappings_bytes"]
                elif "epoch" in item:
                    progress_pct = int((item["epoch"] / item["total_epochs"]) * 100)
                    yield f'data: {json.dumps({"epoch": item["epoch"], "total_epochs": item["total_epochs"], "loss": item["loss"], "progress": progress_pct})}\n\n'

        # Save model files locally
        if model_bytes and mappings_bytes:
            with open(os.path.join(model_dir, "best_model.pth"), "wb") as f:
                f.write(model_bytes)
            with open(os.path.join(model_dir, "mappings.pkl"), "wb") as f:
                f.write(mappings_bytes)
            yield f'data: {json.dumps({"status": "done"})}\n\n'
        else:
            yield f'data: {json.dumps({"error": "Training completed but no model bytes received."})}\n\n'

    except Exception as e:
        yield f'data: {json.dumps({"error": f"Modal training failed: {str(e)}"})}\n\n'
