# KnowledgeLink

A full-stack Knowledge Graph Explorer powered by the **GATCE** (Graph Attention with Hadamard + ConvE) model, initially trained on **FB15k-237**, with support for custom dataset training dynamically via **Modal**.

## Features

| Feature | Description |
|---|---|
| 🔍 **Entity Search** | Instant substring search across entities |
| 🕸️ **Graph Neighborhood** | D3.js force-directed graph of 1-hop edges |
| ✦ **Link Prediction** | Top-K tail predictions with confidence bars |
| 🔥 **Explainability** | Per-layer GATCE attention heatmap (1-hop and 2-hop contributions) + ranked neighbor influence |
| 🚀 **Local Dataset Training** | Upload custom `.txt` files to train models dynamically utilizing Modal |
| 🔁 **Model Management** | Switch between default (`fb15k-237`) and custom trained models, or delete unused ones |
| 🎨 **Premium UI** | Dark glassmorphism, animated predicted edges, responsive |

---

## Project Structure

```text
knowledgelink/
├── backend/
│   ├── config.py                  # Hyperparams, paths & Modal configs
│   ├── main.py                    # FastAPI app — run this
│   ├── model/                     # GATCE model definitions
│   ├── services/                  # Business logic (Predict, Data, Train)
│   └── requirements.txt
├── frontend/
│   ├── index.html                 # App shell (3-panel layout)
│   ├── style.css                  # Core CSS styles
│   ├── css/                       # Modular CSS stylesheets
│   ├── main.js                    # Entry point for the frontend application
│   ├── core/                      # Core JS logic (api, state, etc.)
│   └── features/                  # Distinct feature modules (graph, upload, etc.)
├── models/                        # Saved models, checkpoints, and mappings
│   └── fb15k-237/                 # Default pre-trained model directory
│       ├── data/                  # Original datasets and mapped labels
│       ├── best_model.pth         # Saved weights
│       └── mappings.pkl           # Encoded mappings
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
cd knowledgelink
pip install -r backend/requirements.txt
```

### 2. Available Data and Checkpoints
The system comes with default functionality for `fb15k-237`. You can optionally start the app right away, or prepare your custom data:
- To use the pre-trained `fb15k-237` model, ensure it exists in the `models/fb15k-237/` directory containing `best_model.pth` and `mappings.pkl`.
- To train a new model, you can use the UI to upload a custom edge list (`.txt` file).


### 3. Modal Configuration (For Training)
If you intend to train models using the web GUI, ensure you have set up your Modal token. Training jobs are automatically dispatched to Modal for fast execution.
```bash
python -m modal setup
```

---

## Running the App

```bash
# From the knowledgelink/ directory:
uvicorn backend.main:app --reload --port 8000
```

Then open **[http://localhost:8000](http://localhost:8000)** in your browser.

> **Note:** The app works for graph exploration and entity search on any active model even if you haven't performed a prediction yet.

---

## API Reference

| Method  | Endpoint | Description |
|---------|----------|-------------|
| `GET`   | `/health` | Status: data loaded, models loaded, system health |
| `GET`   | `/models` | Retrieve the list of available trained models |
| `POST`  | `/models/delete/{model_name}` | Delete a custom-trained model |
| `POST`  | `/train` | Upload a dataset and dispatch a training job via Modal |
| `GET`   | `/entities/search?q=<query>&model_name=<name>` | Fuzzy entity search for a given model |
| `GET`   | `/relations?model_name=<name>` | All relations in the active model |
| `GET`   | `/graph/{entity_id}?max_neighbors=40&model_name=<name>` | 1-hop subgraph |
| `POST`  | `/predict` | `{head_id, rel_id, topk, model_name}` → ranked predictions |
| `POST`  | `/explain` | `{head_id, rel_id, tail_id, model_name}` → attention maps and neighbor significance |

Interactive docs: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## How Explainability Works

GATCE is a **graph attention** model. Every layer computes an attention weight per edge, indicating how much each neighboring entity influenced the central node's embedding update.

`explain()` calculates Graph Attention Rollout for a 2-layer GATCE, separating the contribution into:
- **L2 Influence (1-hop):** How much a direct neighbor influences the target directly.
- **L1 Influence (2-hop):** How much a 2-hop neighbor influences a 1-hop neighbor, scaled by that 1-hop neighbor's importance.

The UI renders:
1. **Attention Heatmap** — A visual grid showing the 2-hop (L1) and 1-hop (L2) contributions to the prediction.
2. **Subgraph Highlight** — The most influential neighbor edges turn gold on the D3 graph to visualize the localized contribution path.
3. **Influence List** — Ranked list of neighbors by their mean attention across layers.

---

## Configuration

Edit `backend/config.py` to match your environment variables and training setup if necessary.

```python
class Config:
    models_dir: str      = os.path.join(_PROJECT_ROOT, "models")

    # ── Model (must match training hyperparameters) ─────────
    EMBED_DIM: int  = 200
    NUM_LAYERS: int = 2
    NUM_HEADS: int  = 4
    DROPOUT: float  = 0.1

    # ConvE reshape (EMBED_DIM must be divisible by CONV_DW)
    CONV_DW: int = 10
```
