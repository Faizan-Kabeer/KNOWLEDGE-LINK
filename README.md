# KnowledgeLink: An Explainable Framework Leveraging a Hybrid Graph Attention Encoder and Convolutional Decoder Model for Knowledge Graph Link Prediction

This directory contains the core training implementation and the full-stack web application for the **GATCE (Graph Attention with Hadamard + ConvE)** Knowledge Graph link prediction project.

## Directory Structure

```text
pro-max/
├── link-prediction-gatce-final.ipynb  # Core ML Training Notebook
├── knowledgelink/                    # Full-Stack Web Application
└── README.md                         # This file
```

---

## 1. The ML Model: `link-prediction-gatce-final.ipynb`

This Jupyter Notebook encapsulates the entire research, data processing, and training pipeline for the link prediction model.

### Key Details
- **Architecture**: A bespoke **GATCE (Graph Attention with Hadamard + ConvE)** model. It uses a graph encoder with multi-head attention (GATLayer) to propagate structural edge information, combined with a `ConvE` (2D Convolution) decoder to score (head, relation, tail) triples.
- **Dataset**: Trained and evaluated on the standard **FB15k-237** Knowledge Graph dataset (14,541 entities, 237 original relations). The notebook constructs a bidirectional graph to handle inverse relations.
- **Frameworks**: Built using **PyTorch** and **PyTorch Geometric (PyG)**.
- **Training Strategy**: Uses 1-N multi-hot scoring for efficiency, optimized via AdamW with mixed-precision training (`torch.amp`).
- **Final Metrics**: Achieves a Test MRR of ~0.2883 and Hits@10 of ~45.56%.
- **Outputs**: The notebook exports `best_gatce_model.pth` (the model checkpoint) and `mappings.pkl` (entity and relation string-to-ID mappings) which are consumed by the backend application.

---

## 2. The Application: `knowledgelink/`

**KnowledgeLink** is the productionized full-stack interface built around the trained GATCE model. It allows users to visually interact with the knowledge graph and understand the model's predictions.

### Key Capabilities
- **Explainability**: Dives deeply into *why* the model made a prediction. It visualizes the **Attention Heatmap** and calculates the exact L1 (2-hop) and L2 (1-hop) influence specific neighboring edges had on the final output.
- **Interactive Graph Visualization**: Uses D3.js to render force-directed subgraphs for interactive neighborhood exploration.
- **Custom Dataset Training**: Includes a robust mechanism to upload arbitrary custom `.txt` edge lists through the UI. It dynamically dispatches new training jobs via **Modal** and seamlessly loads the newly trained checkpoints without restarting the server.
- **Multi-Model Support**: Switch seamlessly between the default `fb15k-237` model and your custom uploaded ones.

*For detailed instructions on running the web app, please refer to the `knowledgelink/README.md` file.*
