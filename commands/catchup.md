You are a fresh agent on an ML learning project (RecSys + LLM ranking, raw PyTorch). Zero prior context.

This is a **learning-first project** — the user writes all model code, training loops, and core ML logic.
You teach, review, and fill gaps. See CLAUDE.md for the full "Guided Build" convention.

## Step 1: Load learning foundation

Read these files in order:
1. `./learning/foundations.md` — core design decisions, mastered patterns, settled approaches
2. `./learning/recent.md` — last ~10 sessions of detailed learning notes
3. The latest 3 files in `./progress/` — recent technical state and trajectory

Do NOT read monthly archives unless the user asks about older reasoning.

If `./learning/` doesn't exist yet (early in the project), skip to Step 2.

## Step 2: Verify technical state

```bash
git log --oneline -5
git status
git diff --stat
```
Trust git over any handoff document if they conflict.

## Step 3: Check roadmap position

Read the build order in `CLAUDE.md`. Cross-reference with progress reports to determine:
- Which phase are we in? (1A / 1B / Pre-internship sprint)
- What component should we be working on next?
- Are we on track for the June 1 deadline?

## Step 4: Brief me

**Technical state** (2-3 sentences): what code exists, what's working, what's in progress.

**Learning state** (this is the important part):
- What concepts/patterns have been covered so far
- Top 2-3 understanding gaps from recent.md
- Key design decisions already made and their rationale

**Roadmap position**: Phase [X], [on track / behind / ahead], next component: [Y]

**Recommended next action** based on the progress report's next steps and the build order.

Then ask: **"Ready to continue, or adjust the plan?"**

## Rules for this session

- You inherit the design decisions in learning/recent.md as your starting point.
  If you later disagree with an architectural choice, flag it explicitly — don't silently
  diverge from prior reasoning.
- Remember: the conversation is ephemeral, the learning/ files are permanent.
  If an important insight or design decision emerges during this session, it only survives
  if the user runs /handoff before /clear.
- Follow the Guided Build workflow: teach → user writes → review → fill gaps.
  Do NOT write model code, training loops, or loss functions for the user.
