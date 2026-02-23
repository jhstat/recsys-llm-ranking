Stop all current work. Context is getting full — time to preserve this session's value.

This project has two memory systems. You must update BOTH:
- `./progress/` — technical chain (what was built, what's next)
- `./learning/` — learning chain (what was understood, what's still fuzzy, design decisions)

---

## Step 1: Save pending work

- Save unsaved edits
- `git add -A && git status`

## Step 2: Roadmap check

Compare what was accomplished this session against the build order in `CLAUDE.md`.
Note which phase we're in and whether we're on track for the June 1 deadline.

## Step 3: Write progress report (technical layer)

Create `./progress/YYYYMMDD_topic.md` following the existing chain format:

```markdown
**Previous Report:** YYYYMMDD_previous.md
**Phase:** [1A / 1B / Pre-internship sprint]

## Status
| Component | Status | Notes |
|-----------|--------|-------|
[fill from session work — use component names from CLAUDE.md build order]

## What Was Built
[specific code written, files created/modified, with paths]
[note: who wrote what — user-written model code vs agent-provided boilerplate]

## Key Learnings
[PyTorch patterns encountered, concepts that clicked, things that were harder than expected]

## Remaining Work / Next Steps
1. [priority 1]
2. [priority 2]
3. [priority 3]

## Files to Read First
[3-5 key files for the next agent to orient quickly]
```

## Step 4: Append to learning/recent.md (learning layer — MOST IMPORTANT)

Review the ENTIRE conversation. Extract everything that lives only in our dialogue.
Append to `./learning/recent.md`:

```markdown
---
### YYYY-MM-DD — [component being built]

**Concepts covered:**
- [concept] — [understood / partially understood / needs revisit]
  e.g. "Contrastive loss with in-batch negatives — understood the math,
  but still fuzzy on why temperature scaling affects gradient magnitude"

**PyTorch patterns learned:**
- [pattern] — [where it's used, why it matters]
  e.g. "nn.Embedding with padding_idx — used in user/item ID embeddings,
  zeroes out gradient for padding token"

**Design decisions:**
- [decision]: Chose [X] over [Y] because [Z].
  [capture the reasoning — the next agent must understand WHY, not just WHAT]
  e.g. "Separate towers with shared embedding dim (128) over concatenated input:
  enables ANN retrieval at serving time, which is how Meta does two-tower"

**Understanding gaps:**
- [topic] — what's still unclear, what to revisit
  [be specific — "don't fully understand X" is useless; "unclear why X behaves
  differently when Y" is useful]

**Dead ends:**
- Tried [X], abandoned because [Y]. [saves next agent from repeating mistakes]

**Meta team connections:**
- [how this session's work maps to the team's stack]
  e.g. "The cold-start embedding approach we're building directly mirrors
  'Jobs cold-start LLM-based ranking' on the team's project list"

**Seeds:**
- [ideas raised but not pursued — optimization ideas, extensions, things to try later]
```

## Step 5: Check learning/foundations.md

Read `./learning/foundations.md`. Did this session produce a FUNDAMENTAL shift?
- Core architectural decision made or changed?
- Major PyTorch pattern mastered that unlocks future components?
- Understanding breakthrough that changes how we approach remaining work?

If yes: update foundations.md. If no (most sessions): leave it alone.

## Step 6: Rotate if needed

Count entries in `./learning/recent.md`. If more than 10 entries:
1. Keep the 5 most recent entries
2. Condense older entries into `./learning/archive/YYYY-MM.md` by month — preserve design decisions, understanding gaps, dead ends; drop resolved items unless the resolution was important
3. Remove archived entries from recent.md

## Step 7: Commit and confirm

```bash
git add -A
git commit -m "session: [1-line summary of what was learned + built]"
git push
```

Report:
- ✅ Progress report: `progress/YYYYMMDD_topic.md` (linked to previous: [filename])
- ✅ Learning entry appended ([N] total in recent.md)
- ✅ foundations.md [updated with: X / unchanged]
- ✅ Roadmap: Phase [X], [on track / behind / ahead]
- 🔑 This session's top learning or open question
- Say: **"Safe to /clear. Next session: /catchup"**
