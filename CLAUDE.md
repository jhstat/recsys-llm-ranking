# Agent Instructions — RecSys LLM Ranking

## Overarching Plan

This project is tracked in [jhstat/research-hub](https://github.com/jhstat/research-hub).
**Always refer to `research-hub/ROADMAP.md` for the overarching timeline, priorities, and deadlines.**

Key context from the roadmap:
- This is **Career Prep Project #1** in the 2026–2027 roadmap
- **Hard deadline: core skills functional by June 1, 2026 (Meta internship start)**
- Team: Meta Verticals IQ ML Core Foundation 1 (ML systems, retrieval & ranking, LLM infra, GPU)
- Time budget: ~5 hrs/wk (Phase 1A) → ~10 hrs/wk (Phase 1B onward)
- Total hours before internship: ~115

## Project Purpose

Build an end-to-end recommendation system with LLM-augmented ranking and ML infrastructure — all in raw PyTorch. The project is designed to build hands-on skills aligned to the Meta ML Core Foundation team stack before the internship starts.

## Learning-First Convention ("Guided Build")

This is a **learning project**, not a delivery project. The agent's default behavior is **teaching**, not writing code for the user.

### The Rule

**If you'd need to debug it, modify it, or explain your design choices to a teammate at Meta, you write it yourself.**

### Who Writes What

| Agent handles (low learning value, high time cost) | User writes (high learning value) |
|---|---|
| Project scaffold, directory structure, configs | `nn.Module` classes — every layer, every forward pass |
| Data downloading/preprocessing scripts | Training loops — optimizer, loss, backward, grad clipping |
| Boilerplate (argparse, logging setup, utils) | Loss functions — contrastive loss, ranking loss |
| Test harness and evaluation metric functions | Feature interaction layers — understand the math |
| Dockerfiles, serving configs, CI/CD | `torch.compile` integration — understand what it changes |
| Documentation templates | LLM embedding extraction — tokenization, pooling |
| Progress report scaffolds | Model architecture decisions and justifications |

### Guided Build Workflow

For each component:

1. **Agent teaches** — explain what we're building, why it's designed this way, what the PyTorch patterns are
2. **User writes** — implement it based on the explanation. Doesn't need to be perfect.
3. **Agent reviews** — point out bugs, anti-patterns, missed optimizations. Explain *why* they matter.
4. **Agent fills gaps** — boilerplate, configs, data utilities, testing harness

### Guide Format

Each component has a guide in `docs/guides/`. Structure:

```
## Concept       — what and why (2-3 paragraphs)
## Architecture  — diagram or pseudocode of what you're building
## Your Task     — specific instructions for what to implement
## Hints         — PyTorch API references you'll need
## Checkpoint    — how to verify your implementation works
```

## Build Order

Refer to `research-hub/ROADMAP.md` for the full phase timeline. Summary:

| Phase | Timeline | Components |
|-------|----------|------------|
| 1A | Now–mid-Mar (~15 hrs) | Scaffold, dataset, data pipeline, start two-tower model |
| 1B | mid-Mar–Apr (~60 hrs) | Two-tower retrieval + training loop, ranking model, offline eval |
| Pre-internship sprint | May (~40 hrs) | LLM cold-start, `torch.compile`, basic serving |
| **June 1** | **~115 hrs total** | **Core skills ready** |
| Post-internship | Aug–Sep (~40 hrs) | Polish with Meta learnings, monitoring, deploy |

## Project Structure

```
recsys-llm-ranking/
├── CLAUDE.md              # These instructions
├── README.md              # Project overview
├── docs/
│   └── guides/            # Learning guides per component
├── src/
│   ├── data/              # Data loading, preprocessing, feature engineering
│   ├── models/            # nn.Module classes (two-tower, ranking, LLM)
│   ├── training/          # Training loop, optimizer configs, scheduling
│   ├── inference/         # Serving, torch.compile, batched inference
│   └── evaluation/        # Metrics (NDCG, recall@K), offline evaluation
├── configs/               # Experiment configs (YAML)
├── scripts/               # Utility scripts (download data, run experiments)
├── tests/                 # Test harness
└── progress/              # Progress reports (synced to research-hub)
```

## Conventions

- **All models in raw PyTorch** — no sklearn, no HuggingFace Trainer, no LightFM wrappers
- **Python 3.10+**, type hints encouraged
- **Configs in YAML** — no hardcoded hyperparameters in training scripts
- **Progress reports** in `progress/` — one per meaningful milestone, linked from research-hub daily logs
- **Commit messages** — imperative mood, reference the component being built (e.g., "Implement two-tower forward pass with cosine similarity")
- **Branch strategy** — work on `main` for now (solo project), branch for experiments if needed

## Tech Stack

- PyTorch (core — models, training, inference)
- FAISS or ScaNN (approximate nearest neighbor retrieval)
- Transformers library (pretrained LLM for cold-start embeddings only — not for model training)
- FastAPI or TorchServe (model serving)
- Weights & Biases or TensorBoard (experiment tracking)
- Docker (containerized serving)

## What NOT to Do

- Don't use high-level wrappers that hide PyTorch internals (PyTorch Lightning, HuggingFace Trainer)
- Don't over-engineer infrastructure before models work — get a training loop running first
- Don't spend time on frontend/UI — this is an ML systems project
- Don't skip the guides — even if the agent could write the code faster, the point is learning
