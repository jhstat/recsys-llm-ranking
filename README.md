# RecSys LLM Ranking

End-to-end recommendation system with LLM-augmented ranking and ML infrastructure, built in raw PyTorch.

## What This Is

A hands-on project covering the full ML systems stack for recommendation:

- **Two-tower retrieval** — embedding-based candidate generation with contrastive learning
- **Deep ranking model** — feature interaction layers for learned ranking
- **LLM cold-start** — pretrained LLM embeddings for items with no interaction history
- **Training pipeline** — custom PyTorch training loop with mixed precision, checkpointing, logging
- **Inference optimization** — `torch.compile`, batched serving, latency profiling
- **Evaluation** — offline metrics (NDCG, recall@K), simulated A/B testing framework

## Stack

PyTorch | FAISS | Transformers (embeddings only) | FastAPI | W&B | Docker

## Project Structure

```
src/
├── data/          # Data loading, preprocessing, feature engineering
├── models/        # nn.Module classes (two-tower, ranking, LLM embeddings)
├── training/      # Training loop, optimizer configs, scheduling
├── inference/     # Serving, torch.compile, batched inference
└── evaluation/    # Metrics, offline evaluation
docs/guides/       # Learning guides per component
configs/           # Experiment configs (YAML)
progress/          # Progress reports
```

## Context

Career prep project for Meta Verticals IQ ML Core Foundation internship (Summer 2026).
Tracked in [jhstat/research-hub](https://github.com/jhstat/research-hub) — see ROADMAP.md for the overarching plan.
