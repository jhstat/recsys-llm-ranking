# Foundations — Settled Decisions & Mastered Patterns

Updated only when fundamental shifts occur. Most sessions leave this unchanged.

---

## Architecture

**Pipeline:** Two-tower retrieval → DLRM ranking → FastAPI serving
**Cold-start:** e5-base-v2 LLM embeddings for items with no interaction history

## Core Decisions

- **Raw PyTorch only** — no wrappers (Lightning, HF Trainer). Must be able to debug at the tensor/autograd level.
- **DLRM for ranking** — Meta's architecture. Sparse embeddings + bottom MLP + pairwise dot-product interactions + top MLP.
- **Amazon Electronics** — dataset with text (for LLM), long-tail (for cold-start), manageable size (for local training).

## Mastered Patterns

(None yet — will be updated as components are built and understood.)

## Key PyTorch APIs by Component

(Will be filled in as each component is implemented. Format: API → what it does → gotchas.)
