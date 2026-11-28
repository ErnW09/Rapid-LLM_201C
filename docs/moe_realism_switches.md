# Why the MoE realism switches exist

**Status:** implemented 2026-11-25 on branch `proj_imp`, both **off by default**. See
[current_state.md](current_state.md) for the rest of the project.

```yaml
project_param:
  moe_single_launch: false     # true: one launch per grouped expert GEMM (fused MoE), not one per expert
  moe_active_experts: false    # true: price only the experts the batch touches, E·(1 − (1 − k/E)^tokens)
```

## Short version

Stock RAPID-LLM prices Qwen3-30B-A3B decode as if the batch size barely mattered. A step takes
147.8 ms at B = 1 and 152.4 ms at B = 32. Two pricing simplifications in the MoE FFN cause this:

1. Every one of the 128 experts is launched as its own kernel. Launches alone cost 110.6 ms per
   step, about 75% of the FFN time.
2. Routed tokens are padded so that every expert gets at least one token, at any batch size. At
   B = 1 the simulator reads all 54 GiB of expert weights, but only 8 experts are really used.

Because of this, the AFD results show only one thing: the FFN is a fixed ~143 ms cost. Overlap
gains are small, attention never becomes the bottleneck, and the single-user speed (6.8 tok/s) is
far too slow to validate against published numbers. The two switches remove the simplifications.
They stay off by default, so the required evaluation runs on the unmodified simulator.

## The symptom

Decode step time (first step) on one A100, prompt 2048, stock simulator:

| Batch | 1 | 8 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| Step time (ms) | 147.8 | 148.8 | 152.4 | 157.3 | 167.7 |

From B = 1 to B = 128 the work grows 128×, but the step time grows only 13%. Almost all of the step
is a fixed cost that doesn't depend on the batch.

## Cause 1: one kernel launch per expert

**In the code:** `get_moe_ffn_f` (train_timing.py) prices the routed experts with
`_batched_gemm_forward_compute`. That function prices one expert's GEMM and multiplies by the
number of experts (`per_time * scale`). The per-expert time includes the 9 µs
`kernel_launch_overhead` from the hardware YAML (added in `get_gemm_time`, base_timing.py), so the
launch cost is multiplied by 128 too.

**The cost:** 9 µs × 128 experts × 2 GEMMs (gate+up, then down) × 48 layers = **110.6 ms per
step**, out of 142.6 ms of FFN time at B = 32.

**Why it's unrealistic:** MoE serving runs all experts of a layer in one grouped kernel per GEMM,
for example vLLM's `fused_moe` and TensorRT-LLM's grouped GEMM. RAPID-LLM already prices batched
attention GEMMs this way, with one launch for the whole batch (`single_gpu_gemm_forward`,
train_timing.py). The MoE path just doesn't.

**The switch, `moe_single_launch`:** keep one launch per grouped expert GEMM and remove the
other 127. The weight-read and compute time of each expert is unchanged.

## Cause 2: every expert is priced at every batch size

**In the code:** `_moe_balanced_routed_tokens_per_expert` (train_timing.py) assumes perfectly
balanced routing. It pads the routed tokens up to a multiple of the expert count, so each of the
128 experts gets at least one token. At B = 1, the 8 routed tokens become 128, one per expert.

**The cost:** every step reads all 54 GiB of expert weights and launches all 128 experts, even
when the batch touches only a few of them.

**Why it's unrealistic:** with top-8 routing, B tokens touch only a subset of the experts. With
uniform random routing, one token misses a given expert with probability 1 − k/E, so B tokens touch

> E_active = E · (1 − (1 − k/E)^B), with E = 128 and k = 8

| Batch | 1 | 8 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| Experts touched | 8 | 52 | 112 | 126 | 128 |

The padding is harmless from about B = 64 up, and in prefill, where 2048 tokens touch every expert.
It is badly wrong at small batch.

**The switch, `moe_active_experts`:** price round(E_active) experts. The B·k routed tokens are
split evenly across them, rounded up.

## What the switches change

Decode step time (first step, ms), prompt 2048:

| Batch | Stock | `moe_single_launch` | `moe_active_experts` | Both |
|---|---|---|---|---|
| 1 | 147.8 | 38.0 | 17.4 | 11.4 |
| 8 | 148.8 | 39.1 | 66.3 | 22.2 |
| 32 | 152.4 | 42.7 | 135.1 | 39.2 |
| 64 | 157.3 | 47.5 | 155.1 | 47.1 |
| 128 | 167.7 | 58.0 | 167.7 | 58.0 |

Aggregate decode throughput (B / step time):

| Batch | 1 | 8 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| Stock (tok/s) | 6.8 | 54 | 210 | 407 | 763 |
| Both switches (tok/s) | 88 | 360 | 816 | 1,359 | 2,207 |

**Checks:**

- With the switches off, every number matches the stock simulator exactly (B = 32: 152.4 ms per
  step, prefill 2.5209 s, decode 39.0769 s).
- `moe_single_launch` saves exactly 127 × 2 × 48 × 9 µs = 109.7 ms per decode step. It also saves
  the same 109.7 ms once per prefill (B = 32: 2.5209 → 2.4111 s), since prefill uses the same
  kernels.
- `moe_active_experts` changes nothing at B = 128 or in prefill, where every expert is busy.
- At B = 1, `moe_active_experts` gives 17.4 ms. That is the same step time as a model with only 8
  experts and top-8 routing, which is what B = 1 really uses.

### Effect on the AFD results (B = 32, prompt 2048, output 256)

Per decode step (first step, ms):

| `afd_side` | Stock | Both switches |
|---|---|---|
| `both` (one GPU) | 152.4 | 39.2 |
| `attention` | 9.9 | 9.9 |
| `ffn` | 142.6 | 29.3 |
| `no_overlap` | 153.0 | 39.8 |
| `overlap` | 142.9 | 29.6 |
| FFN share of the one-GPU step | 94% | 75% |
| Overlap saving vs one GPU | 6% | 25% |

The attention side doesn't change, because the switches touch only the MoE FFN.

## Why this matters for the project

- **AFD's tradeoffs only show up with the switches on.** The point of AFD is that attention cost
  grows with batch and context while the FFN's weight reads are shared across the batch. In the
  stock model the FFN is a fixed ~143 ms, so by hand estimate attention would only catch up at
  B ≈ 1,200, far beyond what the attention GPU can hold (about 365, from proposed_memory.md). With the switches the FFN is about 29 ms at B = 32
  and grows with batch, so the crossover should fall inside the feasible range. Case study C1 will
  measure it.
- **Overlap becomes reachable.** Perfect overlap assumes stage times shrink when the batch is split
  into micro-batches (see "Things to know" in current_state.md). In the stock model the step time
  barely shrinks with batch (148.8 ms at B = 8 vs 152.4 ms at B = 32), so micro-batching would
  nearly double the step. With the switches it does shrink (22.2 ms vs 39.2 ms).
- **Validation needs them.** The course slides ask us to validate against published data. At
  6.8 tok/s for one user, the stock model is far slower than any real A100 deployment of this
  model, so validation would fail on the baseline before AFD is involved.

## What the switches don't do

- **Attention-side launch overhead is unchanged.** Real decode often uses CUDA graphs, which hide
  most launch cost everywhere. Modeling that would mean changing `kernel_launch_overhead`, a
  hardware setting the course forbids editing. Both switches change only how the MoE kernel is
  priced, not the hardware.
- **Routing is assumed uniform.** Real routers are skewed, so somewhat fewer distinct experts are
  touched than E_active predicts. The existing `expert_imbalance_factor` has no effect on one GPU.
- **Tokens per expert are rounded up** (B = 32: 112 experts × 3 tokens for 256 routed tokens).
  This adds a little compute but no weight reads. Decode GEMMs this small are bound by memory
  bandwidth, so the effect is negligible.
- **One GPU only.** `moe_active_experts` raises `NotImplementedError` unless one GPU holds every
  expert. That always holds in this project, including on the AFD FFN GPU.
- **Forward pass only.** Training backward (`get_moe_ffn_b`) is unchanged.
- **Memory is unchanged.** All 128 experts are still stored. Only the time is affected.
- **Not yet validated.** The switched-on numbers (for example 88 tok/s for one user) still need to
  be checked against published A100 measurements. That is the next step (V2 in the plan).

## How the report should use them

- **Required evaluation:** stock simulator (switches off), exactly as specified.
- **Same tables with both switches on,** side by side, with this document's reasoning in short.
- **Validation and case studies:** switches on, with the stock result shown where the contrast
  helps.

## Code and tests

| File | What |
|---|---|
| `train_timing.py` | `_moe_active_experts` (new) and both switches in `get_moe_ffn_f` |
| `configs/model-config/course_project/qwen3_30b_a3b_inf.yaml` | the two keys, set to `false` |
| `tests/test_afd_moe_switches.py` | off by default; expert counts at B = 1, 8, 32 and 2048 tokens; single launch saves exactly 127 launches per GEMM; active experts cut B = 1 to 1/16 and leave 2048 tokens unchanged; one-GPU guard |
