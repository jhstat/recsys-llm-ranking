# Learning Log — Recent Sessions

---
### 2026-02-23 — Planning & orientation

**Concepts covered:**
- End-to-end recsys pipeline architecture (retrieval → ranking → serving) — understood at design level, not yet implemented
- DLRM architecture overview (sparse embeddings → bottom MLP → dot-product interactions → top MLP) — understood conceptually, deep paper reading assigned for Session 7
- Two-tower retrieval concept (separate user/item encoders, ANN search at serving time) — understood why towers are separate (enables FAISS retrieval)

**PyTorch patterns learned:**
- None yet (planning session — first code is Session 2)

**Design decisions:**
- **Amazon Electronics over MovieLens/Criteo:** Need text features for LLM cold-start, need long-tail for cold-start scenario. Electronics has both. Single category (~20M reviews) is enough to train locally.
- **DLRM over DCN-v2:** Meta's own architecture. Dot-product feature interactions are simpler than cross layers but the systems challenges (large embedding tables, memory bottlenecks) are what the team actually works on.
- **W&B over TensorBoard:** W&B teaches run comparison, tagging, artifact versioning — same discipline as Meta's internal tools. TensorBoard is a viewer, not an experiment management system.
- **e5-base-v2 over MiniLM:** 768-dim vs 384-dim. Better embedding quality. Extra compute is negligible since we only run inference.
- **`learning/` over `thinking/`:** Renamed the intellectual persistence layer. Tracks concepts, PyTorch patterns, design decisions, and Meta team connections instead of research hypotheses and PI feedback.

**Understanding gaps:**
- DLRM math not yet studied — assigned deep reading for Session 7 (5 papers + open-source repo)
- Don't yet understand how in-batch negatives work for contrastive loss — assigned reading for Session 4
- Unclear on practical FAISS index selection (IVF vs HNSW vs flat) — deferred to Session 6

**Dead ends:**
- None yet

**Meta team connections:**
- Every component in the build plan maps to the team's research areas (documented in CLAUDE.md)
- DLRM is the most direct connection — it's Meta's production ranking architecture
- LLM cold-start directly mirrors "Jobs cold-start LLM-based ranking" on the team's project list

**Seeds:**
- Could add a second Amazon category late in the project to test generalization across domains
- TorchRec (Meta's recsys library) is worth skimming to understand what production-scale problems look like, even though we won't use it

---
### 2026-02-23 — Resource integration & generative recsys planning

**Concepts covered:**
- Generative recommendation paradigm — understood at high level. A single model generates item recommendations via semantic IDs (discrete tokens from quantized embeddings), replacing the retrieve→rank cascade.
- Semantic IDs (SIDs) — understood conceptually. Items are quantized into discrete tokens via RQ-KMeans or FSQ. Enables treating recommendations as a sequence generation problem.
- OpenOneRec architecture — skimmed. 1.7B/8B foundation model, multi-stage training (SFT, knowledge distillation, GRPO), 26.8% Recall@10 improvement on Amazon benchmarks.

**PyTorch patterns learned:**
- None yet (planning session — first code is Session 2)

**Design decisions:**
- **Dedicated reading sessions over supplementary reading:** Chose 2 dedicated ~3hr reading sessions (R1, R2) rather than adding blog posts as optional per-session extras. Rationale: generative recsys is substantial enough to merit focused attention, and the blog posts build on each other.
- **Semantic ID experiment is required, not optional:** In Sessions 11-12, comparing continuous e5 embeddings vs. discrete semantic IDs vs. combined is now a required part of the LLM cold-start work. This bridges classical and generative approaches hands-on.
- **Session 16 generative capstone added:** A tiny transformer decoder (~10M params) over semantic IDs, trained with next-token prediction, evaluated with Recall@K against the two-tower baseline. Adds ~5 hrs to the plan. Chosen over just reading about generative recsys — implementation teaches more than reading.
- **R1 placement after Session 3, R2 after Session 10:** Reading sessions are placed in phase gaps, never interrupting the build flow. R1 gives strategic context before training; R2 gives frontier depth before the sprint.

**Understanding gaps:**
- Don't yet understand the full semantic ID pipeline: how RQ-KMeans differs from simple k-means, what FSQ (Finite Scalar Quantization) is, how codebook optimization (EMA, dead code reset) works — assigned to R1 and R2 reading
- Unclear on how beam search over a codebook maps items back efficiently at inference time — will revisit when implementing Session 16
- Haven't read the OpenOneRec paper deeply yet — only know the high-level architecture from blog summaries

**Dead ends:**
- None

**Meta team connections:**
- **Generative recommendation** directly maps to team's "Jobs cold-start LLM-based ranking" work — this is literally what they're building
- **Semantic IDs** are the bridge concept: the team is likely using some form of item tokenization for their LLM-based ranking
- **PyEmma's blog** written by someone with 8 years at Meta + LinkedIn doing exactly recsys + ML infra — high-quality signal on what matters in production

**Seeds:**
- PyEmma's blog may publish Part 4+ of "A Random Walk Down Recsys" before June — worth checking periodically
- The OpenOneRec benchmark (RecIF-Bench, 96M interactions, 8 tasks) could be an interesting evaluation target if time allows
- After Session 16, could explore replacing k-means with RQ-VAE for hierarchical semantic IDs — more aligned with OpenOneRec but significantly more complex
