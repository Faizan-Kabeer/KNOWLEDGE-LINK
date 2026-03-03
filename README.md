# KnowledgeLink

A full-stack Knowledge Graph Explorer powered by the **GATH** (Graph Attention with Hadamard + ConvE) model trained on **FB15k-237**.

## Features

| Feature | Description |
|---|---|
| 🔍 **Entity Search** | Instant substring search across 14,541 entities |
| 🕸️ **Graph Neighborhood** | D3.js force-directed graph of 1-hop edges |
| ✦ **Link Prediction** | Top-K tail predictions with confidence bars |
| 🔥 **Explainability** | Per-layer GATH attention heatmap + ranked neighbor influence |
| 🎨 **Premium UI** | Dark glassmorphism, animated predicted edges, responsive |

---

## Project Structure

```
knowledgelink/
├── backend/
│   ├── config.py                  # Hyperparams & paths (must match training)
│   ├── main.py                    # FastAPI app — run this
│   ├── model/
│   │   ├── encoder.py             # GATHLayer + GATHEncoder (returns attention weights)
│   │   ├── decoder.py             # ConvEDecoder
│   │   └── gath.py                # GATH — forward() and forward_explain()
│   ├── services/
│   │   ├── data_service.py        # Load FB15k-237, entity search, adjacency
│   │   └── predict_service.py     # Inference + explainability
│   └── requirements.txt
├── frontend/
│   ├── index.html                 # App shell (3-panel layout)
│   ├── style.css                  # Dark theme + animations
│   └── app.js                     # D3.js graph + all UI logic
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
cd knowledgelink
pip install -r backend/requirements.txt
```

### 2. Place Data

Download **FB15k-237** and place the three split files under `knowledgelink/data/`:

```
knowledgelink/
└── data/
    ├── train.txt
    ├── valid.txt
    └── test.txt
```

> The dataset is available on [Kaggle](https://www.kaggle.com/datasets/groceryheist/fb13k-237) or directly from the original paper repository.

### 3. Save and Place the Trained Checkpoint

At the end of training (in Kaggle), save the model using:

```python
import torch

save_path = '/kaggle/working/gath_model.pth'

checkpoint = {
    'model_state_dict': model.state_dict(),
    'ent2id': ent2id,
    'rel2id': rel2id,
    'embed_dim': EMBED_DIM,
    'num_layers': NUM_LAYERS,
    'num_heads': NUM_HEADS
}

torch.save(checkpoint, save_path)
```

Then download `gath_model.pth` from Kaggle and place it as:

```
knowledgelink/checkpoint.pt
```

> **Note:** The filename in `config.py` is `checkpoint.pt` by default. Either rename the file or update `Config.checkpoint_path = "gath_model.pth"` to match.

The backend automatically reads `ent2id`, `rel2id`, `embed_dim`, `num_layers`, and `num_heads` directly from the checkpoint — no manual config editing required.

---

## Running the App

```bash
# From the knowledgelink/ directory:
uvicorn backend.main:app --reload --port 8000
```

Then open **[http://localhost:8000](http://localhost:8000)** in your browser.

> **No checkpoint?** The app still works for graph exploration and entity search — only the Predict / Explain buttons will be unavailable until a checkpoint is placed.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Status: data loaded, model loaded, counts |
| `GET`  | `/entities/search?q=<query>` | Fuzzy entity search |
| `GET`  | `/relations` | All 237 relation types |
| `GET`  | `/graph/{entity_id}?max_neighbors=40` | 1-hop subgraph |
| `POST` | `/predict` | `{head_id, rel_id, topk}` → ranked predictions |
| `POST` | `/explain` | `{head_id, rel_id, tail_id}` → attention maps |

Interactive docs: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## How Explainability Works

GATH is a **graph attention** model. Every layer computes an attention weight per edge, indicating how much each neighboring entity influenced the central node's embedding update.

`explain()` traces edges flowing **into** the query entity, extracts per-layer sigmoid attention scores, and aggregates them into a single importance score per neighbor.

The UI renders:
1. **Attention Heatmap** — a `num_neighbors × num_layers` canvas grid colored by attention intensity
2. **Subgraph Highlight** — explained neighbors turn gold on the D3 graph
3. **Influence List** — ranked neighbors by mean attention across layers

---

## Configuration

Edit `backend/config.py` to match your training setup:

```python
class Config:
    data_dir       = "data"          # path to FB15k-237 splits
    checkpoint_path = "checkpoint.pt"

    EMBED_DIM  = 200   # must match training
    NUM_LAYERS = 6
    NUM_HEADS  = 4
    DROPOUT    = 0.3
    CONV_DW    = 10    # ConvE reshape width
```
