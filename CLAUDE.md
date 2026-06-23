# CLAUDE.md — HybridMTLNet project guide

Read this first. It lets a fresh Claude (any machine) continue the work with the
right context. The **active task is finishing the conference paper**
(`paper/main.tex`), not the code.

---

## 1. What this project is

**HybridMTLNet** — a unified, **box-free** multi-task network that does **instance
segmentation + pose (keypoint) estimation of laboratory mice in one forward pass**.
The deliverable is a **Springer LNCS conference paper for DSAI 2026** (4th Int.
Conf. on Data Science & AI; UTM Kuala Lumpur; proceedings in Springer LNCS).
Paper is **exactly 14 pages = the LNCS limit** — re-check page count after every edit.

`paper/main.tex` is the **single source of truth** for all reported numbers.
(`README.md` is outdated — its 0.8105 / 0.9542 are from an old single run; ignore.)

---

## 2. ⚠️ CORE INTEGRITY PRINCIPLES (do not violate)

The paper's honesty is the whole point. Every claim must be verifiable against
code/data; no fabrication; baselines must be fair.

- **The contribution is unification + efficiency, NOT a multi-task accuracy gain.**
  The ablation explicitly does **not** find synergy. The intro says so verbatim:
  *"our goal is not a multi-task accuracy gain—which our ablation does not find—but
  a single network that preserves segmentation accuracy at only a small cost in
  pose while roughly halving the computation."* Keep that framing.
- The paper **openly reports a small negative transfer on pose** (−0.0114). Do not
  hide or inflate it.
- **Two comparison frames — never conflate them:**
  - **(A) vs EXTERNAL baselines** (Table 1): HybridMTLNet *matches or surpasses*
    them (+0.023 mIoU vs Mask R-CNN, +0.073 PCK vs YOLO26-Pose).
  - **(B) vs OUR OWN single-task ablation** (Table 3): joint training is *slightly
    worse on pose* (the −0.0114 negative transfer) — same architecture.
  "matching or surpassing dedicated single-task baselines" = frame A. The honest
  "small cost in pose" = frame B.

---

## 3. Method (key technical facts)

- **Encoder:** shared **ResNet-34** (ImageNet-pretrained), features `x0–x4`.
- **Seg decoder:** U-Net style, skip connections, outputs a **2-channel mask**
  (one channel per mouse), predicted at full 256×256 (no RoI grid → crisp edges).
- **Pose head:** 3 deconv layers + 1×1 conv → **6-channel heatmap** = **3 keypoints
  × 2 instances** (keypoints: nose, shoulder, tail base). Inference = argmax per channel.
- **N = 2 cap** comes from the **fixed-width output heads** (2-ch mask, 6-ch heatmap),
  NOT from Hungarian matching. Hungarian is general (any N×M). To scale to K
  instances: widen heads to K-ch mask + 3K heatmaps.
- **Loss:** `L = λ_seg·L_seg + λ_pose·L_pose`, with **λ_seg = 1, λ_pose = 500**.
  - `L_seg = BCE + Dice` (Dice smoothing ε = 1 in numerator and denominator).
  - `L_pose = MSE` over all K=6 channels and H×W pixels.
  - λ_pose=500 because heatmaps are sparse → L_pose is ~2–3 orders smaller than L_seg.
- **Order-Invariant Instance Matching (§3.4):** at each training step, build cost
  matrix `C[i,j] = 1 − IoU(pred_i, gt_j)`, solve with Hungarian
  (`scipy.optimize.linear_sum_assignment`), reorder GT to match output channels,
  then compute loss. The assignment is **detached / non-differentiable** (gradients
  flow through the loss, not the matching).
- **Data:** 3,710 manually annotated images, 8 mouse behavioral tests; 64.4%
  single-mouse, 35.6% two-mouse; split train 2,968 / val 371 / test 371 (80/10/10);
  mixed source resolution → all train/eval at 256×256.
- **Augmentation (train only):** h/v flip, rotation ±30°, color jitter, p=0.5.
- **Training:** Adam lr 1e-4, ReduceLROnPlateau (halve, patience 5), 100 epochs,
  batch 16, single RTX 5060 (8 GB), cuDNN deterministic. **3 seeds: 42, 43, 44.**
  Run-to-run non-determinism ≈ 0.01 PCK.

---

## 4. Key numbers (paper = source of truth; 3-seed mean ± std)

| Setting | mIoU | PCK@0.05 |
|---|---|---|
| **Joint MTL (main result)** | **0.8083 ± 0.0012** | **0.9605 ± 0.0027** |
| Seg-only ceiling | 0.8089 ± 0.0009 | — |
| Pose-only ceiling | — | 0.9719 ± 0.0018 |
| + Cross-task guidance (`joint_bi`) | 0.8081 ± 0.0018 | 0.9667 ± 0.0034 |

- **Negative transfer on pose** = 0.9605 − 0.9719 = **−0.0114** (real, > seed std).
- Cross-task bidirectional guidance recovers ~half the gap (0.9667) → **partial,
  reported as future work**, not a fix.
- **Loss-weight sweep (Table 2, single run seed 42):** 1:1 → 0.8105/0.7511;
  1:100 → 0.8061/0.7960; 1:400 → 0.8114/0.9617; **1:500 → 0.8097/0.9678**;
  1:1000 → 0.8074/0.9694. (Table 2 is a single representative run; Table 1/3 are
  3-seed means — captions explain the difference. Fig. 3 is a *validation* curve,
  peak val PCK 0.9595, ≠ the test numbers.)
- **External baselines (Table 1):** Mask R-CNN (w/ aug, seed 42) mIoU 0.7854;
  YOLO26m-Pose PCK 0.8872. These live in the `\mrcnnmiou` / `\mrcnndelta` macros
  in the main.tex preamble.

Raw results: `models/multiseed_results.json` (keys: seg_only, pose_only, joint,
joint_mg, joint_pg, joint_bi) and `models/sweep_results.json`. **These files are
gitignored — see §6.**

---

## 5. Repo layout

```
paper/main.tex          THE paper (LNCS). Compile target. references.bib, llncs.cls, splncs04.bst, figures/
config.py               DATA_CFG / MODEL_CFG / TRAIN_CFG dataclasses (task, seed, loss weights, flags)
src/models/mtl_net.py   HybridMTLNet architecture
src/training/trainer.py Trainer (incl. _combine_loss, uncertainty weighting flag)
src/evaluation/         evaluator.py, metrics.py (mIoU, PCK), benchmark.py
src/data/dataset.py     MouseMTLDataset (augment=(split=="train"))
baselines/maskrcnn/maskrcnn.py   Mask R-CNN baseline (has `multiseed` subcommand)
baselines/yolo/         YOLO seg/pose baselines
app/run_multiseed.py    3-seed runner (seg_only/pose_only/joint/joint_bi…)
app/run_lossweight_sweep.py   loss-weight sweep → models/sweep_results.json
app/plot_training_curves.py   Fig. 3 (reads models/checkpoints/sweep_1_500/train_log.csv)
app/summarize_results.py      summarize JSONs
```

---

## 6. ⚠️ What is NOT in git (gitignored — must transfer separately)

A fresh clone can **edit/compile the paper** but **cannot retrain or reproduce
numbers** without these:

- `/data/` — the 3,710 images + masks + annotations (~9k files).
- `/models/` — checkpoints (`*.pth`) **and the results JSONs** that back Tables
  1/2/3 (`multiseed_results.json`, `sweep_results.json`, `ablation_results.json`,
  `experiment_results.json`).
- `venv/`, `weights/`.

To do experiments on another machine: copy `data/` and `models/` over manually.
To only continue paper writing: nothing extra needed — `paper/` is fully committed.

---

## 7. Building the paper

Use any TeX distribution's `pdflatex` + `bibtex`. On the original machine MiKTeX was at
`C:/Users/usEr/AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe`.

```
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

For text-only edits (no new citations) two `pdflatex` passes suffice. **Confirm
"Output written on main.pdf (14 pages…)" after every change** — going to 15 pages
means trimming float-free prose or references (trimming text near floats does NOT
reduce pages).

---

## 8. Working conventions (learned the hard way)

- **Verify claims against the main.tex SOURCE (grep), never against PDF text
  extraction.** PDF extraction mangles brackets, subscripts, `·`, and em-dashes and
  produces *false* "bug" reports. ~95% of extracted-PDF review findings were artifacts.
- **Triage review feedback into must-fix vs polish. Do NOT reflexively apply every
  suggestion; verify a claimed bug first.** The user pushed back on over-editing.
- Keep edits small and surgical; recompile and re-check page count each time.
- LNCS `\orcidID{...}` renders the iD as a superscript `[number]` — that is correct
  template behaviour, not a bug.
- **Always respond in Thai.** User requested this on 2026-06-23.
- **Gemini-assisted workflow:** user sometimes pastes a Gemini-rewritten paragraph
  and asks for review. Always check for: (1) Gemini citation artifacts like `[cite: N]`
  — remove all of them; (2) hallucinated or garbled phrases; (3) implementation-specific
  terms that should stay framework-agnostic (e.g. "PyTorch computation graph" → just
  "computation graph"); (4) missing real LaTeX citations that must be restored.
- **MiKTeX not installed** on `C:/Users/Thana` machine. Compile at original machine
  (`C:/Users/usEr/...`) or via Overleaf. All text-only edits in this session have not
  been verified for page count — **compile and confirm 14 pages before submission.**

---

## 9. Status & pending work

- **Abstract + Introduction + Related Work + Fig. 1 caption** — reviewed line-by-line
  with the user and finalized. Intro reuses the advisor-approved abstract's wording.
- **DSAI 2026 submission:** abstract registered on EasyChair (**Submission 25**).
  **Full paper PDF due ~29 June 2026.** Check whether review is double-blind (the
  current PDF shows author names; anonymize if required). Source files (`.tex/.bib/
  .cls/.bst/figures`) only needed at camera-ready.
- **Co-author / advisor:** Praisan Padungweang (Khon Kaen University) — must approve
  before submission. ORCIDs already in `\author{...}`.

### Session summary (2026-06-23 → 2026-06-24)
The following edits have been applied to `paper/main.tex`:
- **Terminology RESOLVED:** "behavioral tests" now used consistently everywhere
  (was mixed with "paradigms" and "test paradigms" in §3.1, §5.2).
- **§3.1 Dataset** — full prose rewrite (improved sentence structure, removed
  clunky "Crucially..." sentence → replaced with concise box-free annotation note).
- **§3.2 Architecture** — full prose rewrite (fixed grammar, removed informal
  language, "U-Net-style" → "dense per-pixel decoder" to match abstract).
- **§3.3 Loss Function** — full prose rewrite (fixed redundancies, moved ε=1
  definition next to equation, tightened each paragraph).
- **§3.4 Order-Invariant Instance Matching** — sentence-by-sentence review done;
  Gemini version obtained and reviewed. **NOT YET APPLIED.** Three fixes needed
  before applying:
  1. Remove all `[cite: 240]` Gemini artifacts
  2. Fix "it from the network may place..." → "it may place..." (Gemini error)
  3. Remove "PyTorch" → "computation graph"
  After applying, restore `\cite{kuhn1955hungarian}` in Step 2.
  Good things to keep from Gemini version: separate `\noindent\textbf{}` paragraphs
  per step, `\hat{M}_i`/`M_j` notation with definition line, `\operatorname{IoU}`.
- **Several minor fixes** across the paper: "L² distance" removed from §3.3,
  pose head grammar fixed in §3.2, "laboratory subjects" → "laboratory mice" in
  Conclusion.

### Open items
- **§3.4** — apply Gemini version with 3 fixes above (next task).
- **§3.5 Data Augmentation** — not yet reviewed.
- **§4 Experimental Setup, §5 Results, §6 Conclusion** — not yet reviewed in
  this round (were reviewed in a prior session; may need another pass).
- **Compile PDF** — not done on this machine (no MiKTeX). Must verify 14 pages
  after all edits are applied. Use original machine or Overleaf.
- **Baseline numbers** (Mask R-CNN 0.7854, YOLO26-Seg 0.5451, YOLO26-Pose 0.8872,
  maDLC 0.4098) need verification against baseline run logs — not reproducible
  from this repo alone (baselines trained elsewhere).
