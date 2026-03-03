"""
FastAPI application — KnowledgeLink backend.

Run with:
    cd knowledgelink
    uvicorn backend.main:app --reload --port 8000
"""
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Make project root importable ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.config import cfg
from backend.services.data_service import DataService
from backend.services.predict_service import PredictService

# ── Singletons ───────────────────────────────────────────────────────────
data_svc    = DataService.get_instance(cfg)
predict_svc = PredictService.get_instance(cfg, data_svc)

# ── App ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="KnowledgeLink API",
    description="GATH-powered Knowledge Graph Explorer with explainability.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    data_svc.load()
    if os.path.exists(cfg.checkpoint_path):
        predict_svc.load_model()
    else:
        print(
            f"[WARN] Checkpoint not found at '{cfg.checkpoint_path}'. "
            "Prediction/Explain endpoints will return 503 until a checkpoint is provided."
        )


# ── Pydantic request schemas ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    head_id: int
    rel_id: int
    topk: int = 10


class ExplainRequest(BaseModel):
    head_id: int
    rel_id: int
    tail_id: int
    max_neighbors: int = 25


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "data_loaded":  data_svc._loaded,
        "model_loaded": predict_svc.ready,
        "num_entities": data_svc.num_entities,
        "num_relations":data_svc.num_relations,
    }


@app.get("/entities/search")
def search_entities(q: str, topk: int = 15):
    """Fuzzy substring search over entity names."""
    if not q.strip():
        return []
    return data_svc.search_entities(q, topk)


@app.get("/relations")
def get_relations():
    """Return all relation types (id + name)."""
    return data_svc.get_all_relations()


@app.get("/graph/{entity_id}")
def get_graph(entity_id: int, max_neighbors: int = 40):
    """
    Return the 1-hop neighbourhood of an entity as a graph payload:
    { nodes: [...], links: [...] }
    """
    if entity_id not in data_svc.id2ent:
        raise HTTPException(status_code=404, detail="Entity not found.")

    neighbors = data_svc.get_neighbors(entity_id, max_neighbors)

    nodes: dict = {
        entity_id: {
            "id":   entity_id,
            "name": data_svc.id2ent[entity_id],
            "type": "center",
        }
    }
    links = []

    for n in neighbors:
        tid = n["target"]
        if tid not in nodes:
            nodes[tid] = {"id": tid, "name": n["target_name"], "type": "neighbor"}
        links.append({
            "source":      entity_id,
            "target":      tid,
            "relation":    n["relation_name"],
            "relation_id": n["relation_id"],
            "type":        "known",
        })

    return {"nodes": list(nodes.values()), "links": links}


@app.post("/predict")
def predict(req: PredictRequest):
    """Top-K tail predictions for a (head, relation) pair."""
    if not predict_svc.ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Place checkpoint.pt in the project root."
        )
    if req.head_id not in data_svc.id2ent:
        raise HTTPException(status_code=404, detail="head_id not found.")
    if req.rel_id not in data_svc.id2rel:
        raise HTTPException(status_code=404, detail="rel_id not found.")

    return predict_svc.predict(req.head_id, req.rel_id, req.topk)


@app.post("/explain")
def explain(req: ExplainRequest):
    """
    Return per-layer attention weights showing which neighbors of head
    contributed most to the prediction of tail.
    """
    if not predict_svc.ready:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    for eid, label in [(req.head_id, "head_id"), (req.tail_id, "tail_id")]:
        if eid not in data_svc.id2ent:
            raise HTTPException(status_code=404, detail=f"{label} not found.")
    if req.rel_id not in data_svc.id2rel:
        raise HTTPException(status_code=404, detail="rel_id not found.")

    return predict_svc.explain(req.head_id, req.rel_id, req.tail_id, req.max_neighbors)


# ── Serve frontend ───────────────────────────────────────────────────────
_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.isdir(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(_frontend_dir, "index.html"))
