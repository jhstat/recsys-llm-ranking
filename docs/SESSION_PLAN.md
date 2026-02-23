# Session Plan — RecSys LLM Ranking

**Created:** 2026-02-23 (Session 1)
**Deadline:** June 1, 2026 (~126 hrs total)

---

## System Overview

```
Retrieval (Two-Tower)  →  Ranking (DLRM)  →  Serving (FastAPI)
     ↓                         ↓                    ↓
  ~1000 candidates       score & reorder        top-K to user
  from millions          with rich features     with latency budget
     ↓
  LLM Cold-Start (e5-base-v2)
  (items with no history
   get embeddings from text)
```

**Dataset:** Amazon Product Reviews 2023 — Electronics category

**Generative recsys thread:** This plan builds the classical retrieve→rank pipeline first (priority #1), while weaving in a strategic thread on **generative recommendation** — the frontier direction where a single model generates item recommendations via semantic IDs, replacing the traditional cascade. This thread flows: R1 (learn the concepts) → build classical baseline → R2 (deep dive) → semantic IDs in DLRM → capstone generative mini-project.

---

## Phase 1A — Foundation (Now through mid-March, ~15 hrs)

### Session 1: Planning & Orientation ✅
- Locked in: dataset (Amazon Electronics), architecture (DLRM), tracking (W&B), LLM (e5-base-v2)
- Created session plan and reading list
- Modified handoff/catchup commands for ML learning project

### Session 2: Data Pipeline
| | |
|---|---|
| **Build** | Dataset download, preprocessing, `torch.utils.data.Dataset`, `DataLoader`, negative sampling |
| **You write** | `Dataset` class, collate function, negative sampling logic |
| **Agent provides** | Download script, preprocessing utils, test harness |

**Pre-reading (before session):**
- [ ] [PyTorch Data Loading Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) — understand `Dataset`, `DataLoader`, `collate_fn`
- [ ] Skim the Amazon Reviews dataset format: what fields exist, how interactions are structured

**Post-reading (after session):**
- [ ] Meta Engineering: *"Recommending items to more than a billion people"* (2015 blog) — Meta's recsys philosophy

---

### Session 3: Two-Tower Model
| | |
|---|---|
| **Build** | User tower, item tower, similarity function, `nn.Module` structure |
| **You write** | Both tower `nn.Module` classes, forward pass, embedding layers |
| **Agent provides** | Guide, shape-checking test harness |

**Pre-reading:**
- [ ] *"Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"* (Yi et al., 2019) — **focus on Section 3** (model architecture). This is Google's two-tower paper.
- [ ] [PyTorch nn.Embedding docs](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) — padding_idx, sparse gradients
- [ ] [PyTorch nn.Module docs](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) — forward(), parameter registration

**Post-reading:**
- [ ] *"Mixed Negative Sampling for Learning Two-Tower Neural Networks in Recommendations"* (Yang et al., 2020) — preview of training considerations for next session

**Revisit after R1:** PyEmma's "Long User Sequence Modeling" — how production systems handle the user sequences your towers encode

---

### Reading Session R1: "Where Recsys Is Going" (~3 hrs)

> **Placement:** Between Phase 1A and 1B. You've built the data pipeline and two-tower model — enough hands-on context to appreciate the frontier. Before diving into training, get the strategic picture.

**Reading list:**
- [ ] PyEmma: [My 2025 Recommendation System Paper Summary](https://pyemma.github.io/) (~20 min) — field overview, the "One-Series" paradigm shift toward unified models
- [ ] PyEmma: [A Random Walk Down Recsys — Part 1](https://pyemma.github.io/A-Random-Walk-Down-Recsys/) (~20 min) — OpenOneRec, OxygenRec, Meta sequential rec, Promise. Covers semantic IDs, multi-stage training, production inference
- [ ] [OpenOneRec paper](https://arxiv.org/abs/2512.24762) Sections 1-3 (~40 min) — architecture, semantic IDs, training methodology. Skim, don't deep-read. Focus on: how does a unified generative model replace retrieve→rank?
- [ ] PyEmma: [Long User Sequence Modeling](https://pyemma.github.io/) (~25 min) — DIN → TransAct → TWIN → DV365 evolution. How user history scaling from 10² to 10⁵ items changes architecture. Directly feeds into your training sessions.

**Reflection prompt:**
> "You just built a two-tower model. OpenOneRec replaces this with a single generative model using semantic IDs. What does the two-tower model do well that a generative approach has to work hard to replicate? What does the generative approach solve that your two-tower can't?"

---

## Phase 1B — Core Training (mid-March through April, ~60 hrs)

### Session 4: Contrastive Loss & Training Loop (Part 1)
| | |
|---|---|
| **Build** | In-batch negatives, contrastive loss, basic training loop skeleton |
| **You write** | Loss function, training step, optimizer setup |
| **Agent provides** | Config YAML template, logging boilerplate |

**Pre-reading:**
- [ ] *"Efficient Training of Retrieval Models using Negative Cache"* (Lindgren et al., 2021) — Sections 1-3 for in-batch negatives intuition
- [ ] [PyTorch Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html) — understand the computation graph

---

### Session 5: Training Loop (Part 2) — Mixed Precision, Checkpointing, Logging
| | |
|---|---|
| **Build** | AMP, gradient clipping, LR scheduling, W&B integration, checkpoint save/load |
| **You write** | Full training loop with all the above |
| **Agent provides** | W&B setup, config schema |

**Pre-reading:**
- [ ] [PyTorch AMP Recipe](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) — GradScaler, autocast
- [ ] [torch.nn.utils.clip_grad_norm_ docs](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- [ ] [PyTorch LR Scheduler docs](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) — CosineAnnealingLR, OneCycleLR

---

### Session 6: Retrieval Evaluation + FAISS
| | |
|---|---|
| **Build** | Recall@K, FAISS index building, ANN retrieval pipeline |
| **You write** | Evaluation loop, FAISS integration code |
| **Agent provides** | Evaluation harness, metric implementations |

**Pre-reading:**
- [ ] [FAISS Getting Started](https://github.com/facebookresearch/faiss/wiki/Getting-started) — index types, add/search API
- [ ] *"Billion-scale similarity search with GPUs"* (Johnson et al., 2019) — Meta's FAISS paper. Skim Sections 1-3: what problem does ANN solve, how does IVF work?

---

### Session 7: DLRM Ranking Model Architecture
| | |
|---|---|
| **Build** | Full DLRM: sparse embeddings, bottom MLP, dot-product feature interactions, top MLP |
| **You write** | Complete ranking `nn.Module` — every layer, every forward pass |
| **Agent provides** | Guide with architecture diagram, shape-checking tests |

**Pre-reading (DLRM deep dive — the most important reading block):**
- [ ] Meta blog: *"DLRM: An Advanced, Open Source Deep Learning Recommendation Model"* (2019) — **read first**, intuition-level with diagrams
- [ ] *"Deep Learning Recommendation Models for Personalization and Recommendation Systems"* (Naumov et al., 2019) — **the DLRM paper**, focus on Sections 2-3: architecture, feature interaction via dot products
- [ ] Browse [`facebookresearch/dlrm`](https://github.com/facebookresearch/dlrm) model code on GitHub — raw PyTorch, close to what you'll write
- [ ] *"Architectural Implications of Embedding Tables for DLRMs"* (Mudigere et al., 2021) — **optional**, covers memory/compute bottlenecks of large embedding tables (relevant to GPU computation on the team)

**Post-reading:**
- [ ] Skim *"TorchRec: A PyTorch Library for Recommendation Systems"* (Meta, 2022) — you won't use TorchRec, but understanding what it abstracts shows you what's hard at scale

**Revisit after R1:** PyEmma's "Recsys 2025 Paper Summary" — SUAN's gated feature interactions as an alternative to dot-product interactions in DLRM

---

### Session 8: Ranking Training — BPR & Binary Cross-Entropy Loss
| | |
|---|---|
| **Build** | Ranking loss functions, pointwise vs pairwise training, training loop for ranker |
| **You write** | Loss function, training loop, data pipeline for ranking stage |
| **Agent provides** | Config, data sampling utilities |

**Pre-reading:**
- [ ] *"BPR: Bayesian Personalized Ranking from Implicit Feedback"* (Rendle et al., 2009) — foundational pairwise ranking paper. Short and readable.
- [ ] Understand the difference: pointwise (BCE on each item) vs pairwise (BPR — compare positive vs negative) vs listwise (LambdaRank)

**Revisit after R1:** PyEmma's "A Random Walk Part 1" — hybrid contrastive + ranking loss patterns used in production generative recsys

---

### Sessions 9-10: Offline Evaluation — NDCG, Full Pipeline
| | |
|---|---|
| **Build** | NDCG@K, MAP, full retrieval→ranking pipeline evaluation, experiment comparison |
| **You write** | Metric implementations, end-to-end eval script |
| **Agent provides** | Evaluation harness, W&B comparison dashboards |

**Pre-reading:**
- [ ] [NDCG on Wikipedia](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) — understand DCG, IDCG, normalization
- [ ] *"A Theoretical Analysis of NDCG Type Ranking Measures"* (Wang et al., 2013) — only if you want the math rigorously

**Post-reading:**
- [ ] Review your own code end-to-end: retrieval → ranking → evaluation. Can you explain every design choice?

---

### Reading Session R2: "Frontier Architectures & Production Patterns" (~3 hrs)

> **Placement:** Between Phase 1B and Pre-Internship Sprint. You have the complete retrieve→rank→eval pipeline working. Now deeply appreciate what modern approaches change, and carry those insights into LLM cold-start, optimization, and the generative capstone.

**Reading list:**
- [ ] PyEmma: [A Random Walk Down Recsys — Part 2](https://pyemma.github.io/) (~20 min) — HyFormer, token-level collaborative alignment, OneMall, sparse attention for long sequences. How LLMs integrate into traditional ranking.
- [ ] PyEmma: [A Random Walk Down Recsys — Part 3](https://pyemma.github.io/) (~20 min) — semantic IDs deep dive: GLASS codebook optimization, QARM V2 (RQ-KMeans + FSQ), end-to-end SID generation, dynamic embeddings. The most directly relevant post for your LLM cold-start work.
- [ ] PyEmma: [Recsys 2025 Paper Summary](https://pyemma.github.io/) (~30 min) — PinFM KV cache optimization, SUAN gated interactions, multi-expert systems, serving patterns, Triton kernel optimization
- [ ] PyEmma: [KDD 2025 Paper Summary](https://pyemma.github.io/) (~25 min) — vector quantization replacing HNSW, MoE for multi-task, knowledge distillation (LLM→BERT), cross-domain contrastive learning
- [ ] PyEmma: [FSDP2 Under the Hood](https://pyemma.github.io/) (~30 min) — PyTorch distributed training internals: DTensor, parameter sharding, CUDA stream management, mixed precision. Prep for torch.compile mindset.

**Reflection prompt:**
> "You've built the classical pipeline. You're about to add LLM cold-start. After reading about semantic IDs, codebook optimization, and generative retrieval — how would you design the cold-start differently if you were building from scratch? Pick one frontier technique (semantic IDs, hybrid losses, gated interactions) that you'd want to add to your pipeline. Why that one?"

---

## Pre-Internship Sprint (May, ~45 hrs)

### Sessions 11-12: LLM Cold-Start + Semantic IDs
| | |
|---|---|
| **Build** | Extract embeddings from item text via e5-base-v2, integrate as features into DLRM for items with no interaction history. Then: quantize embeddings into discrete semantic IDs and compare approaches. |
| **You write** | Tokenization pipeline, pooling strategy, embedding integration into model forward pass. **Also:** k-means quantization of e5 embeddings into discrete cluster IDs, embedding lookup from cluster IDs, A/B evaluation of continuous vs. discrete representations. |
| **Agent provides** | Model loading utils, preprocessing scripts, k-means utility, evaluation harness modifications |

**Part 1 — Continuous embeddings (Session 11):**
Extract e5-base-v2 768-dim embeddings → integrate as dense features into DLRM → evaluate on cold-start items

**Part 2 — Semantic IDs experiment (Session 12):**
Quantize e5 embeddings via k-means into discrete cluster IDs (simplified "semantic IDs") → use cluster IDs as a sparse feature in DLRM (embedding lookup, just like any categorical feature) → compare three setups:
1. Raw 768-dim e5 features (continuous)
2. Discrete semantic IDs only (cluster indices → learned embedding)
3. Both combined

This teaches the core insight from OpenOneRec — representing items as discrete tokens rather than continuous vectors — in a tractable way, and directly sets up the Session 16 generative capstone.

**Pre-reading:**
- [ ] *"DropoutNet: Addressing Cold Start in Recommender Systems"* (Volkovs et al., 2017) — how to handle missing interaction features
- [ ] [Sentence-Transformers docs](https://www.sbert.net/) — understand what `model.encode()` does under the hood
- [ ] [E5 paper: "Text Embeddings by Weakly-Supervised Contrastive Pre-training"](https://arxiv.org/abs/2212.03533) — understand the model you're using

**Revisit from R2:** PyEmma's "A Random Walk Part 3" — codebook optimization techniques (EMA updates, dead code reset) for your semantic ID experiment

---

### Session 13: torch.compile
| | |
|---|---|
| **Build** | Compile models, benchmark before/after, understand graph breaks |
| **You write** | `torch.compile` integration, profiling code |
| **Agent provides** | Benchmarking scaffold |

**Pre-reading:**
- [ ] [PyTorch torch.compile Tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [ ] *"TorchDynamo: An Experiment in Dynamic Python Bytecode Transformation"* — understand what happens under the hood
- [ ] [torch.compile troubleshooting guide](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html) — graph breaks, what to avoid

**Revisit from R2:** PyEmma's "FSDP2 Under the Hood" — CUDA stream management, communication overlap, and GPU optimization context

---

### Sessions 14-15: Serving
| | |
|---|---|
| **Build** | FastAPI endpoint, batched inference, Docker container, latency benchmarking |
| **You write** | Endpoint logic, batching strategy, model loading |
| **Agent provides** | Dockerfile, FastAPI scaffold, load testing script |

**Pre-reading:**
- [ ] [FastAPI First Steps](https://fastapi.tiangolo.com/tutorial/first-steps/) — just the basics
- [ ] Understand: model serialization (`torch.save` / `torch.jit.script` / `torch.export`)

---

### Session 16: Generative Recommendation Mini-Project (Capstone, ~5 hrs)

> **The generative recsys thread culminates here.** You've read about it in R1/R2, experimented with semantic IDs in Sessions 11-12, and now you build a tiny generative recommender from scratch in raw PyTorch.

| | |
|---|---|
| **Build** | A small transformer decoder that takes a user's interaction history (as a sequence of semantic IDs) and autoregressively generates the next semantic ID — retrieving items by *generation* instead of embedding similarity search |
| **You write** | The causal transformer decoder `nn.Module` (4-6 layers, ~10M params), the autoregressive generation loop, beam search over the codebook, Recall@K comparison vs your two-tower + FAISS baseline |
| **Agent provides** | Sequence formatting utils, generation scaffolding, comparison evaluation harness |

**Step-by-step:**
1. **Codebook:** Reuse the k-means semantic IDs from Session 12 (already built)
2. **Data:** Convert user interaction sequences into sequences of semantic ID tokens
3. **Model:** Write a small causal transformer decoder — pure PyTorch, no HuggingFace. Multi-head self-attention, causal mask, positional encoding, vocabulary = your codebook size
4. **Training:** Next-token prediction on semantic ID sequences (cross-entropy loss). This is exactly how GPT works, but over item tokens instead of word tokens
5. **Inference:** Beam search to generate top-K semantic IDs → map back to items
6. **Evaluation:** Compare Recall@K: generative retrieval vs. two-tower + FAISS. Same dataset, same metrics, different paradigm.

**Pre-reading:**
- [ ] Revisit "A Random Walk Part 3" — GLASS retrieval-guided decoding, beam search optimization
- [ ] Revisit OpenOneRec paper Section 3 — how they handle generation over semantic IDs
- [ ] [PyTorch Transformer tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) — if you need a refresher on attention/causal masking

**Why this matters for Meta:**
- Pure PyTorch transformer implementation — high learning value, the kind of code the team writes daily
- Directly mirrors Meta's "Jobs cold-start LLM-based ranking" approach
- Gives you a concrete, experience-backed answer for "how would you use LLMs in recommendations?" in interviews
- You can compare classical vs. generative retrieval on the same dataset — rare hands-on perspective

---

## Supplementary Resources

### OpenOneRec Paper
- [OpenOneRec Technical Report](https://arxiv.org/abs/2512.24762) — unified generative recsys framework. Semantic IDs, 1.7B/8B foundation models, 26.8% Recall@10 improvement on Amazon benchmarks. RecIF-Bench with 96M interactions across 8 tasks.

### PyEmma Blog ([pyemma.github.io](https://pyemma.github.io/))

*Staff SWE, 8 years at Meta + LinkedIn. Recommendation systems and ML infrastructure.*

| Post | Topics | Used in |
|------|--------|---------|
| [My 2025 Recommendation System Paper Summary](https://pyemma.github.io/) | Field overview, "One-Series" paradigm, unified models | R1 |
| [A Random Walk Down Recsys — Part 1](https://pyemma.github.io/A-Random-Walk-Down-Recsys/) | OpenOneRec, OxygenRec, Meta sequential rec, semantic IDs | R1 |
| [A Random Walk Down Recsys — Part 2](https://pyemma.github.io/) | HyFormer, token-level alignment, sparse attention, OneMall | R2 |
| [A Random Walk Down Recsys — Part 3](https://pyemma.github.io/) | Semantic ID deep dive: GLASS, QARM V2, codebook optimization | R2, Sessions 11-12, 16 |
| [Long User Sequence Modeling](https://pyemma.github.io/) | DIN → TransAct → TWIN → DV365, sequence scaling | R1, Session 3 |
| [FSDP2 Under the Hood](https://pyemma.github.io/) | PyTorch distributed training, DTensor, CUDA streams | R2, Session 13 |
| [Recsys 2025 Paper Summary](https://pyemma.github.io/) | PinFM, SUAN, multi-expert, serving, Triton kernels | R2, Session 7 |
| [KDD 2025 Paper Summary](https://pyemma.github.io/) | Vector quantization, MoE, knowledge distillation, SIDs | R2 |

---

## Reading Priority Guide

If you're short on time, prioritize readings in this order:

1. **Always read:** PyTorch API docs for the session's topic (these are short and essential)
2. **High priority:** The primary paper for each session (marked in bold above)
3. **High priority:** R1 and R2 reading sessions — these are dedicated sessions, not skippable
4. **Medium priority:** PyEmma blog posts and Meta blog posts (good intuition, faster than papers)
5. **Low priority:** Optional/post-reading papers, "revisit" cross-references (only if you have extra time or curiosity)

Estimated reading time per coding session: **30-60 minutes** of pre-reading.
Reading sessions R1 and R2: **~3 hours each** (dedicated, no coding).

## Updated Timeline

| Phase | Sessions | Hours | Notes |
|-------|----------|-------|-------|
| 1A — Foundation | 1-3 | ~15 | Data pipeline, two-tower model |
| **R1 — "Where Recsys Is Going"** | **Reading** | **~3** | **Generative recsys concepts** |
| 1B — Core Training | 4-10 | ~60 | Training, DLRM, eval |
| **R2 — "Frontier Architectures"** | **Reading** | **~3** | **Semantic IDs, production patterns** |
| Sprint | 11-15 | ~40 | LLM cold-start + semantic IDs, torch.compile, serving |
| **16 — Generative Rec Capstone** | **Capstone** | **~5** | **Transformer decoder over semantic IDs** |
| **Total** | **18 sessions** | **~126** | **June 1 deadline** |
