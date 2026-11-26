# Proposal: results, validation and case studies (Project 1, AFD)

**Status:** proposal, nothing run yet (2026-11-25, branch `proj_imp` at `f8c331b`). See
[current_state.md](current_state.md) for what exists and [proposed_memory.md](proposed_memory.md)
for the per-side memory design this plan depends on.

**Sources:** [PROJECTS.md](../PROJECTS.md) and the course PDFs in `docs/given/`:

- `ECE201C_RAPID_LLM_Projects.pdf` is the same text as PROJECTS.md. It sets the required evaluation.
- `ECE201C_RAPID_LLM_Project_Options_v10.pdf` (Project 1 slides) gives the transfer and overlap
  formulas, the memory picture and an example case study.
- `RAPID-LLM_for_ECE201C.pdf` (kickoff deck) gives the regression rule on slide 29.

Every number marked *hand estimate* below is arithmetic from the configs. It is a prediction to
check against the runs, not a result.

## What the course expects

The Options slides ("Project Focus") list four things:

1. **Modify RAPID-LLM** to model the technique.
2. **Validate against published data** (papers, blogs, benchmarks).
3. **Case studies that show hardware design impact.** Changing the model, hardware, batch or
   sequence lengths is allowed "within reason". PROJECTS.md adds: find where the feature helps,
   where it stops helping, and why.
4. **Read the background papers** (DistServe, MegaScale-Infer) and decide which approach to model.

They also fix two details that the current code doesn't follow yet:

- **Overlap formula.** No overlap is `tA + tF + 2·tc`, and overlap is `max(tA, tF, tc)`, where `tc`
  is the **one-way** transfer time per layer. The code uses `max(tA, tF, 2·tc)`.
- **Neutral check.** "A and F on one GPU, zero transfer" must reproduce the original decode timing
  **exactly** (kickoff slide 29: "fails on any changed byte").

Kickoff slide 32 describes an earlier version of the project (an `afd:` config block,
`program/device_groups.py`, `course/regress.py`). None of those are in this repo, so this plan
doesn't use them.

## Notation

| Symbol | Meaning | Qwen3-30B-A3B value |
|---|---|---|
| L | layers | 48 |
| H | hidden size | 2048 |
| s | bytes per activation (bf16) | 2 |
| B | sequences in the batch (1 token each per decode step) | |
| S, O | prompt length, output length | |
| c | context at a decode step (S + tokens so far) | |
| tA(B, c) | attention side, per layer: QKV, attention, out proj, layernorm1 | from runs |
| tF(B) | FFN side, per layer: router, experts, dispatch, combine, layernorm2 | from runs |
| h(B) | output head, once per step, on the attention GPU | from runs |
| tc(B) | one-way transfer per layer = α + B·H·s / bw | 6.3 µs at B = 32 |

Time per decode step:

| System | Step time |
|---|---|
| Single GPU | L·(tA + tF) + h |
| AFD, no overlap | L·(tA + tF + 2·tc) + h |
| AFD, perfect overlap | L·max(tA, tF, tc) + h |

## Prerequisites (before any results)

| # | Item | Why |
|---|---|---|
| P1 | Overlap uses one-way `tc` | Matches the slides. The number barely moves at B = 32, but it matters in the link sweep (C3). |
| P2 | `no_overlap` = original layer time + 2·`tc` | With a free link it is then exactly `both`, which passes the neutral check. Today it adds the terms in a different order. |
| P3 | Per-side memory and largest batch | Designed in [proposed_memory.md](proposed_memory.md). Every case study needs B_max. |
| P4 | Print an AFD breakdown line | At each sampled decode step, print `tA`, `tF`, one-way `tc` and `h` (per layer). Then a script can compose C3 to C7 without re-running the simulator. |
| P5 | Sweep driver `afd_sweep.py` | Already proposed in proposed_memory.md. Extend it to take a grid (model, hardware, B, S, O, `afd_side`, switches) and append one CSV row per run under `results/`. A run takes about 1.6 s, so grids of hundreds of runs are fine. |
| P6 | Realism switches (if approved, see Decisions) | Two `project_param` switches, off by default: (a) charge the kernel launch once per grouped expert GEMM instead of once per expert; (b) price only the experts that receive tokens, E·(1 − (1 − k/E)^B). See V2. |
| P7 | Last-token prefill logits (if approved) | Prefill builds logits for every prompt token: 0.58 GiB per sequence at S = 2048 and 9.27 GiB at S = 32K. That limits the prefill batch and breaks C7. Add a `project_param` switch, off by default. |
| P8 | Tests | The neutral check (P2), attention + FFN = both, no-overlap − both = 2L·`tc`, and the memory checks listed in proposed_memory.md. |

## Comparison rules (used throughout)

- **Latency:** milliseconds per decode step. Every sequence in the batch gets one token per step, so
  this is also the time per output token for one user. Report the first step, the last step and
  the average.
- **Throughput:** B / (average step time) in tok/s, and **tok/s per GPU**. AFD uses two GPUs, so
  comparing it with one GPU at the same B is not fair.
- **Fair comparison:** use the same number of GPUs, run each system at its own largest batch that
  fits, and compare tok/s per GPU at the latency each one reaches. Example at the required point:
  one GPU at B = 32 gives 210 tok/s (210 per GPU). The AFD pair with overlap gives 224 tok/s, which
  is 112 per GPU. AFD only wins when it uses the larger batch it allows.
- **Stock vs realism:** the required evaluation uses the stock simulator. Case studies report stock
  numbers, plus realism-switch numbers wherever the conclusion changes.

## R: Required evaluation (exactly as PROJECTS.md specifies)

`qwen3_30b_a3b_inf.yaml` on `a100_80GB_1gpu.yaml`, B = 32, S = 2048, O = 128 and 256,
link 100 GB/s with α = 5 µs.

**Runs:** 2 output lengths × 5 `afd_side` modes, plus the memory runs and B_max search from
proposed_memory.md.

**Table R1: time per token**

| System | O | Step: first / last / avg (ms) | Total decode (s) | tok/s | tok/s per GPU | TTFT (s) |
|---|---|---|---|---|---|---|
| Single GPU | 128, 256 | | | | | |
| AFD, no overlap | 128, 256 | | | | | |
| AFD, overlap | 128, 256 | | | | | |

**Table R2: per-layer breakdown.** `tA`, `tF`, `tc` (latency part and data part) and `h`. Also how
much of `tF` is kernel launches (9 µs × 128 experts × 2 GEMMs).

**Table R3: memory.** Decode peak and largest batch for the attention GPU, the FFN GPU and a single
GPU, for O = 128 and 256. Footnote: the single-GPU prefill peak is 81.42 GiB, over the 80 GiB card
(see P7).

## Validation

**V1: reproduce the instructor's numbers.** Nearly done by hand already.

- Transfer: 2L·α = 480 µs of latency per step, and 2·L·B·H·s / 100 GB/s = 252 µs of data at B = 64.
- Memory: one GPU holds 61 GB of weights and leaves 19 GB for KV. The attention GPU holds 3 GB of
  weights and 77 GB of KV. The FFN GPU holds 58 GB of weights with 22 GB unused. These match the
  hand estimates in proposed_memory.md (56.87, 2.85 and 54.0 GiB).

**V2: check the single-GPU baseline against published measurements.**

- *Target:* A100 80 GB, bf16, Qwen3-30B-A3B decode numbers from vLLM, SGLang or similar benchmark
  posts, at a few batch sizes. If nothing good exists for this model, use a similar MoE model.
- *What we already know:* the simulator gives a 147.8 ms step at B = 1 (6.8 tok/s for one user) and
  152.4 ms at B = 32. There are two causes:
  - Kernel launches: 9 µs × 128 × 2 × 48 = 110.6 ms per step.
  - Routed tokens are padded so that every expert gets at least one token, so all 128 experts are
    priced at any B. At B = 1 that means reading 54 GiB of weights instead of the 3.4 GiB of the 8
    experts actually used. The expected number of experts in use is 8 at B = 1, about 52 at B = 8,
    about 112 at B = 32, and nearly all 128 from B = 64.
- *Method:* compare the stock simulator with the measurements and explain the gap with the
  breakdown. Then turn on the P6 switches and compare again. Report both.
- *Same parameters, fewer experts:* a copy of the model YAML with 16 experts, top-1 and expert
  width 6144 has the same total and active expert parameters and the same tokens per expert. It
  has 8× fewer launches. Comparing it with the original separates the launch cost from the weight
  reads without changing any code.

**V3: compare with the papers.** Compare our trends with MegaScale-Infer (SIGCOMM 2025, AFD for
MoE) on three things: per-GPU throughput gain, the attention-to-FFN GPU ratio they choose, and how
sensitive they are to the network. For the prefill/decode part of C7, compare with DistServe
(OSDI 2024). Quote their numbers from the papers, not from memory.

## Case studies

Defaults unless stated: Qwen3-30B-A3B, A100, S = 2048, O = 256, 100 GB/s, α = 5 µs.

### C1: When does an AFD pair beat two ordinary GPUs? (core)

- **Question:** at the same GPU count (2), which gives more tok/s per GPU at a given latency?
- **Sweep:** B = 1, 2, 4, 8, 16, 32, 64, 128, 256, then up to each system's B_max.
  - Two single GPUs serving separate batches: up to B ≈ 109 each (decode only, context 2304).
  - The AFD pair: up to B ≈ 365 (attention GPU limit).
  - Do this with the realism switches off and on.
- **Scope:** decode only, in steady state. Prefill is in C7.
- **Figures:**
  - Throughput against latency (tok/s per GPU vs ms per token), one curve per system.
  - Stacked bars of `tA`, `tF` and transfer against B.
- **Hypothesis (hand estimate, stock):** `tF` stays near 143 ms up to B = 365, because it is
  launch-bound and every expert is always priced.
  - Single GPU at B = 109: about 162 ms per step, or about 670 tok/s per GPU.
  - AFD pair at B = 365 with overlap: about 143 ms per step, or about 1,280 tok/s per GPU (≈ 1.9×).
  - AFD pair at B = 365 without overlap: about 193 ms per step, or about 950 tok/s per GPU (≈ 1.4×).
- **Hypothesis (realism on):** `tF` drops to about 30 ms. Attention, at about 6.7 ms + 0.114 ms per
  sequence, catches up with the FFN near B ≈ 200, well inside the attention GPU's limit. In the
  stock model that crossover would need B ≈ 1,200, which doesn't fit.

### C2: Context length (core)

- **Question:** attention and KV grow with context and the FFN doesn't. How does the AFD advantage
  change?
- **Sweep:** S = 1K, 2K, 8K, 32K (O = 256), each system at its B_max.
- **Hand estimate of B_max:**

  | S | KV per sequence | Single GPU, decode only | Attention GPU |
  |---|---|---|---|
  | 1K | 0.12 GiB | 197 | 658 |
  | 2K | 0.21 GiB | 109 | 365 |
  | 8K | 0.77 GiB | 29 | 99 |
  | 32K | 3.02 GiB | 7 | 25 |

  The ratio stays at about 3.3× for every context, because it is set by the free memory: 77.15 GiB
  vs 23.13 GiB. What changes is time. At 32K, attention at B = 25 takes about 47 ms per step. In the
  realism model that is more than the FFN, so the FFN GPU sits mostly idle. That leads into C4.
- **Figures:** B_max vs S, and tok/s per GPU vs S, for both systems.

### C3: The link: latency or bandwidth? (core)

- **Question:** where does the transfer stop being negligible? Does AFD need a fast link or a
  low-latency one?
- **Sweep:** α = 1, 2, 5, 10, 20, 50, 100 µs × bw = 10, 25, 50, 100, 200, 400 GB/s, at B = 32 and
  at B_max. This is arithmetic on the per-layer times from P4, with no new runs.
- **Figure:** a heatmap of the AFD throughput gain over single GPU (no overlap), with the contour
  where the gain reaches 1.0.
- **Hand estimate:**
  - The latency costs 2L·α = 96·α per step, whatever B is: 0.48 ms at 5 µs and 1.9 ms at 20 µs.
  - The data term equals the latency once B·H·s / bw = α, i.e. B ≈ 122 at 100 GB/s and 5 µs.
  - At B = 365 the data term is 5.7 ms per step at 25 GB/s and 0.36 ms at 400 GB/s.
  - With overlap, the link only limits anything when `tc` exceeds `tF` per layer: about 3 ms in the
    stock model, about 0.6 ms with realism. No realistic link gets there.
  - Expected takeaway: link latency matters more than bandwidth, and only without overlap.

### C4: How many attention GPUs can one FFN GPU serve? (next)

- **Question:** MegaScale-Infer's sizing question. The FFN cost hardly changes with batch, so one
  FFN GPU should be able to serve several attention GPUs.
- **Setup:** n attention GPUs, each at B_A = its B_max, feed one FFN GPU at batch n·B_A, with
  n = 1 … 8.
  - Run the FFN side at batch n·B_A. Its memory never runs out, since its limit is in the hundreds
    of thousands.
  - The FFN GPU's link port carries all n streams: `tc_F` = α + n·B_A·H·s / bw.
  - No routing between FFN GPUs is needed, because there is only one.
- **Metric:** tok/s per GPU = n·B_A / step / (n + 1). Find the best n.
- **Hypothesis:** by roofline, the FFN stays memory-bound until about 157 tokens per expert, i.e.
  B ≈ 2,500 in total. With B_A = 365, that is n ≈ 6 to 7 before the FFN becomes the limit. With
  realism on, the best n comes earlier, where `tF`(n·B_A) ≈ `tA`(B_A).

### C5: Can perfect overlap actually be reached? (next)

- **Question:** within one batch, attention and FFN depend on each other, so overlap needs m
  micro-batches taking turns. What does that cost?
- **Setup:** m = 1, 2, 4. Per layer, a step takes about m·max(`tA`(B/m), `tF`(B/m), `tc`(B/m)),
  plus one pipeline fill per step. Run at B/m.
- **Hypothesis:** each micro-batch re-reads every expert's weights. So `tF`(B/m) ≈ `tF`(B), and
  m = 2 roughly doubles the FFN time per step. That is worse than no overlap at all. Perfect overlap
  is then an optimistic bound. Micro-batching only pays off when attention dominates (long context,
  realism on) or the FFN becomes compute-bound. Kickoff slide 32 mentions micro-batches for exactly
  this reason.

### C6: Mixed hardware (next)

- **Question:** each side has a different bottleneck. The FFN needs HBM bandwidth (weights) and the
  attention GPU needs HBM capacity (KV). Does matching the hardware to the side pay off? The slides
  give the example of NVIDIA + Groq, where LPUs run the FFN.
- **Setup:** run the attention side on hardware X and the FFN side on hardware Y, then compose.
  - Use the existing configs: A100 (1,986 GB/s) and `H100_SXM5_80GB.yaml` (3,350 GB/s, tp 1).
    Combinations: A100 + H100 for FFN, H100 + A100 for FFN, and H100 + H100.
  - Optional what-ifs as **new** YAML files (the shared configs are never edited): an attention GPU
    with more HBM capacity, and an FFN GPU with 3× bandwidth.
- **Hypothesis (stock):** moving the FFN to an H100 only cuts the 29 ms weight-read part (to about
  17 ms). The 110 ms of launches stay the same, so the FFN side gets only about 10% faster. With
  realism on, the FFN side is mostly weight reads and gets close to 1.7× faster (the bandwidth
  ratio). Hardware conclusions therefore depend on P6, so report both.

### C7: Eight GPUs in pools (capstone; the instructor's example)

- **Question:** given 8 identical A100s, which is best?
  - **Unified:** 8 GPUs, each doing prefill, attention and FFN.
  - **Prefill/decode split:** 2 prefill GPUs + 6 GPUs that each do attention and FFN.
  - **Three pools:** 2 prefill GPUs + 3 attention GPUs + 3 FFN GPUs.
- **Grid (from the slide):** S = 1K / 32K, O = 32 / 128, bw = 25 / 400 GB/s, α = 1 / 20 µs. That is
  16 points × 3 setups. Optional: search every split of the 8 GPUs, using C4 for FFN GPUs that
  serve more than one attention GPU.
- **Report:** TTFT, ms per token (both overlap cases), max batch, tok/s. Find where the gain ends as
  transfer time or prompt length grows.
- **Model needed (new):**
  - Prefill pool: requests/s = n_P · B_P / T_prefill(B_P, S).
  - Decode pool: requests/s = (number of decode units) · B_D / (O · step time).
  - System: tok/s = O · min(prefill rate, decode rate).
  - Unified, with static batches: requests/s = 8 · B / (T_prefill + O · step time).
  - TTFT = T_prefill plus the KV handoff from prefill to attention: L·S·2 KiB per request, which is
    96 MiB at 1K and 3 GiB at 32K. At 25 GB/s that is 4 ms and 129 ms per request.
  - Assumptions to state: no queueing, no continuous batching, and the prefill GPUs hold the full
    model.
- **Depends on:** P3, P4, P7 (without P7, the 32K prefill batch is 1), and C4.

### C8: Other models (stretch)

- **Dense Llama-3.1-8B** (the kickoff deck mentions it):
  - FFN weights are 10.5 GiB; attention weights plus embeddings are 4.5 GiB.
  - KV is 128 KiB per token.
  - B_max at context 2304: one GPU about 231, attention GPU about 268, only about 16% more (hand
    estimate).
  - A dense FFN reads its weights once per step, whatever B is, and becomes compute-bound near
    B ≈ 157.
  - Expected: AFD helps a dense model much less than an MoE model.
  - Needs: a dense-layer memory split. proposed_memory.md raises `NotImplementedError` for this
    today.
- **Fewer, larger experts:** the V2 variant (16 experts, top-1, width 6144) also works as a case
  study on how the number of experts changes the AFD trade-off.

## Priority and order

| Tier | Items | Notes |
|---|---|---|
| Must | P1–P5, P8, R, V1 | Required evaluation, correct formulas, reproducible runs |
| Core | V2 (+ P6), C1, C2, C3 | Validation and the main story. C2 and C3 need no new simulator code. |
| Next | C4, C5, C6 | Reuse P4 and the composition script |
| Capstone | C7 (+ P7) | Needs the throughput model and the prefill/decode handoff |
| Stretch | C8 | Needs the dense memory split |

Suggested order:

1. P1, P2, P8 (small and independent).
2. P3, P4, P5.
3. R and V1.
4. V2. Its result decides whether P6 goes in.
5. C1 to C3.
6. Then work down the tiers.

## Decisions to confirm

1. **Realism switches (P6).** Add them and report stock and realism side by side? Without them, the
   only finding is that the FFN is launch-bound, and validation (V2) will fail on the baseline.
   *Proposed: yes, off by default.*
2. **Overlap formula.** Use the slides' one-way `tc` (P1)? *Proposed: yes.*
3. **Comparison basis.** Same GPU count, each system at its largest batch that fits, tok/s per GPU?
   *Proposed: yes.*
4. **Prefill logits (P7).** Add a last-token-only switch for case studies, and keep the stock
   behaviour (and the 81.42 GiB footnote) for R? *Proposed: yes.*
5. **Scope.** Is C7 in or out? Is C8 in or out?
6. **What-if hardware files (C6).** Is it fine to add new hardware YAMLs for what-ifs, as long as
   the shared configs are never edited? *Proposed: yes, in a separate folder.*
7. **Results in git.** Commit the CSVs under `results/` and a plotting script, so the report's
   figures can be rebuilt? (`output/` is gitignored.)

## Files this plan adds or touches

| File | Change | For |
|---|---|---|
| `inference_timing.py` | One-way `tc` in overlap, exact `no_overlap`, AFD breakdown print, P6/P7 switches | P1, P2, P4, P6, P7 |
| `train_timing.py` | Grouped-launch and active-expert switches in the MoE expert pricing | P6 |
| `memory_estimation.py` | Per-side memory (from proposed_memory.md); later, dense split | P3, C8 |
| `afd_sweep.py` (new) | Grid runner, writes CSV rows | P5 |
| `afd_compose.py` (new) | Link sweep, A:F ratio, micro-batches, pools, from the per-layer times | C3–C7 |
| `results/` (new) | CSVs and plots | all |
| `tests/` | Neutral check and the other checks | P8 |
| `docs/current_state.md` | Update as items land | |
