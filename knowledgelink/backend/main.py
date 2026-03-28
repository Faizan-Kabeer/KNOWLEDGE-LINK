"""
FastAPI application — KnowledgeLink backend. 

"""

import os
import sys

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Make project root importable ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.config import cfg
from backend.services.data_service import DataService
from backend.services.predict_service import PredictService
from backend.services.modal_training_service import train_on_modal
from fastapi.responses import StreamingResponse

# ── Singletons ───────────────────────────────────────────────────────────
data_svc    = DataService.get_instance(cfg)
predict_svc = PredictService.get_instance(cfg, data_svc)

# ── App ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="KnowledgeLink API",
    description="GATCE-powered Knowledge Graph Explorer with explainability.",
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
    default_model_dir = os.path.join(cfg.models_dir, "fb15k-237")
    ckpt_path = os.path.join(default_model_dir, "best_model.pth")
    if os.path.exists(ckpt_path):
        predict_svc.load_model()
    else:
        print(
            f"[WARN] Default checkpoint not found at '{ckpt_path}'. "
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


class TrainRequest(BaseModel):
    model_name: str


class SwitchModelRequest(BaseModel):
    model_name: str


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


@app.get("/models")
def get_models():
    """Return all available models."""
    return {
        "active_model": predict_svc.active_model_name,
        "models": predict_svc.list_models()
    }


@app.post("/models/switch")
def switch_model(req: SwitchModelRequest):
    """Switch the active model."""
    if req.model_name not in predict_svc.list_models():
        raise HTTPException(status_code=404, detail="Model not found.")
    
    try:
        predict_svc.switch_model(req.model_name)
        return {"status": "success", "active_model": req.model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {str(e)}")


@app.delete("/models/{model_name}")
def delete_model(model_name: str):
    """Delete a custom model. The default fb15k-237 model cannot be deleted."""
    import shutil

    if model_name == "fb15k-237":
        raise HTTPException(status_code=400, detail="Cannot delete the default model.")

    if model_name not in predict_svc.list_models():
        raise HTTPException(status_code=404, detail="Model not found.")

    model_dir = os.path.join(cfg.models_dir, model_name)
    try:
        shutil.rmtree(model_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

    # If the deleted model was active, fall back to the default
    if predict_svc.active_model_name == model_name:
        try:
            predict_svc.switch_model("fb15k-237")
        except Exception:
            pass  # If fallback fails, service will just be unready

    return {"status": "success", "deleted_model": model_name, "models": predict_svc.list_models(), "active_model": predict_svc.active_model_name}



@app.post("/training/upload")
async def upload_dataset(
    model_name: str = Form(...),
    file: UploadFile = File(...),
    mapping_file: UploadFile = File(None)
):
    """Upload a dataset file for a custom model and an optional entity mapping file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    
    # ── Create model directory and data subdirectory ────────
    model_dir = os.path.join(cfg.models_dir, model_name)
    data_dir = os.path.join(model_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Save the triples file as `train.txt`
    train_path = os.path.join(data_dir, "train.txt")
    try:
        content = await file.read()
        with open(train_path, "wb") as f:
            f.write(content)
        
        # Save the optional mapping file as `entities_labels.tsv`
        if mapping_file and mapping_file.filename:
            mapping_path = os.path.join(data_dir, "entities_labels.tsv")
            mapping_content = await mapping_file.read()
            with open(mapping_path, "wb") as f:
                f.write(mapping_content)
            return {
                "status": "success",
                "message": f"Dataset and mapping saved. Dataset: {len(content)} bytes, Mapping: {len(mapping_content)} bytes."
            }

        return {"status": "success", "message": f"Dataset saved. Size: {len(content)} bytes."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.get("/training/stream")
async def stream_training(model_name: str):
    """Stream live PyTorch training progress from Modal GPU."""
    model_dir = os.path.join(cfg.models_dir, model_name)
    train_path = os.path.join(model_dir, "data", "train.txt")
    if not os.path.exists(train_path):
        raise HTTPException(status_code=400, detail="Dataset not uploaded yet. Call /training/upload first.")
    
    with open(train_path, "r", encoding="utf-8") as f:
        triples_txt = f.read()

    return StreamingResponse(
        train_on_modal(triples_txt, model_name, cfg.models_dir),
        media_type="text/event-stream"
    )

# ── Serve frontend ───────────────────────────────────────────────────────
_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.isdir(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(_frontend_dir, "index.html"))
