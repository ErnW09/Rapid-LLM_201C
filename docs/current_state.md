# Project 1 (AFD): current state

As of 2026-11-25, branch `proj_imp`. The plan for results and case studies is in
[proposed_studies.md](proposed_studies.md).

The assignment is in [PROJECTS.md](../PROJECTS.md#project-1-attentionffn-disaggregation-afd), and
the code pointers are in [COURSE_README.md](../COURSE_README.md#project-1-attentionffn-disaggregation-qwen3-30b-a3b).
In short: estimate decode on two GPUs (attention plus KV cache on one, the experts on the other),
with a B × H hidden-state transfer each way at every layer. Report time per token without and
with overlap, and the largest batch each side can hold.

**Required evaluation:** `qwen3_30b_a3b_inf.yaml` on `a100_80GB_1gpu.yaml`, batch 32,
prompt 2048, output 128 and 256, link 100 GB/s with 5 µs latency.

## Done

| Commit | Who | What |
|---|---|---|
| `e910ccd` | ErnW09 | uv environment (`pyproject.toml`, `uv.lock`, Python ≥ 3.12). |
| `183b404` | chenr4731 | AFD decode timing switch (details below). Its commit message says "prefill and decode sides", but the split is between the **attention** and **FFN** sides of decode. Prefill is unchanged. |

### AFD timing switch (`183b404`)

**Config:** the `project_param` block in `configs/model-config/course_project/qwen3_30b_a3b_inf.yaml`.

```yaml
project_param:
  afd_side: "both"          # both | attention | ffn | no_overlap | overlap
  afd_link_bw_GBps: 100
  afd_link_latency_us: 5
```

**Code:** `inference_timing.py`, in `_build_decode_transformer_results`, right after
`transformer_forward` is computed. It sets the per-layer `transformer_time_f` like this:

| `afd_side` | Per-layer decode time | Output head (`linear_softmax_f`) |
|---|---|---|
| `both` (default) | Unchanged single-GPU time | kept |
| `attention` | MHA + layernorm1 (+ its reduction) | kept |
| `ffn` | MLP (incl. MoE router, dispatch, combine) + layernorm2 (+ its reduction) | **0**, because the head sits on the attention GPU |
| `no_overlap` | unsplit layer time + 2·tc (attention, send, FFN, send back) | kept, added after the layers |
| `overlap` | max(attention, FFN, tc) | kept, added after the layers |

`tc = latency + B × H × bytes_per_activation / bandwidth` is the **one-way** transfer of the hidden
state. No-overlap pays it twice per layer. Overlap counts it once, because each link direction is
its own stage (see Decisions). `no_overlap` adds 2·tc to the unsplit layer time rather than summing
the two sides, so that a free link (latency 0, infinite bandwidth) reproduces `both` exactly.

### Check run (2026-11-25)

This was a sanity check, not the final evaluation. It used batch 32, prompt 2048, output 256, and
scratch copies of the YAML with only `global_batch_size` and `afd_side` changed. All five modes
ran without errors.

| `afd_side` | Decode step, first (ms) | Decode step, last (ms) | Total decode (s) |
|---|---|---|---|
| `both` | 152.4 | 152.9 | 39.08 |
| `attention` | 9.9 | 10.3 | 2.58 |
| `ffn` | 142.6 | 142.6 | 36.49 |
| `no_overlap` | 153.0 | 153.5 | 39.23 |
| `overlap` | 142.9 | 142.9 | 36.59 |

**Consistency checks:**

- Prefill is 2.52 s in every mode.
- attention + ffn = both, to within rounding.
- no_overlap − both ≈ 0.6 ms per step. This matches the hand value of
  48 layers × 2 × (5 µs + 32·2048·2 B / 100 GB/s) ≈ 0.61 ms.

Peak memory on one GPU (baseline): prefill 81.42 GiB (**1.42 GiB over the 80 GiB A100**), decode
63.63 GiB. Every mode reports the same memory, because the switch changes only timing.

## Decisions

- **Overlap uses the one-way transfer: max(tA, tF, tc) (2026-11-25).** This follows the course
  slides (`docs/given/ECE201C_RAPID_LLM_Project_Options_v10.pdf`, "What exactly should the team
  implement?"). The link is taken as full duplex, with 100 GB/s in each direction, so the A→F and
  F→A transfers of different micro-batches run at the same time. A shared (half-duplex) link would
  make it max(tA, tF, 2·tc). Neither changes any result here: at B = 32, tc ≈ 6 µs per layer while
  tF ≈ 2,970 µs. Verified at B = 32 / output 256: `no_overlap` 153.0 ms and `overlap` 142.9 ms per
  step, the same as before the change. `no_overlap` with a free link matches `both` exactly.
- **MoE realism switches, both off by default (2026-11-25).** `project_param.moe_single_launch`
  charges one kernel launch per grouped expert GEMM instead of one per expert.
  `project_param.moe_active_experts` prices only the E·(1 − (1 − k/E)^tokens) experts the batch is
  expected to touch, instead of all 128. The required evaluation uses the stock simulator. The same
  tables, plus validation and case studies, also get the switched-on numbers. Why, and measured
  effects: [moe_realism_switches.md](moe_realism_switches.md). Tests:
  `tests/test_afd_moe_switches.py`. With both on at B = 32: one GPU 39.2 ms per step, FFN side
  29.3 ms, attention side 9.9 ms, overlap 29.6 ms.

## Things to know when interpreting results

- **Decode is FFN-bound (stock).** The FFN is about 93% of the step at B = 32. About 75% of the FFN
  time is kernel-launch overhead: 9 µs (`kernel_launch_overhead` in the hardware YAML) × 128
  experts × 2 GEMMs per layer. `_batched_gemm_forward_compute` (train_timing.py) multiplies the
  per-expert time, launch overhead included, by the expert count. As a result, `overlap` ≈ `ffn`,
  and AFD's speedup over single-GPU is small (~6%). This is mostly a modeling artifact. With both
  realism switches on, the FFN is 75% of the step and overlap saves 25%.
- **Overlap is an upper bound.** Overlap needs micro-batches. With m micro-batches of size
  b = B/m, a layer takes max(tA(b) + 2·tc(b) + tF(b), m · max(tA(b), tF(b), tc(b))). The overlap
  formula is the limit where stage times shrink in proportion to b. Our FFN time barely changes
  with batch (142.5 ms at B = 16 vs 142.6 ms at B = 32), so m = 2 would nearly double the step.
  The report should say this. Case study C5 will quantify it.
- **Output head placement.** In `overlap` and `no_overlap`, the output head runs serially after the
  layers, on the attention GPU.
- **Transfer size ignores `query_tokens`.** It uses `batch_size × hidden_dim`, which is fine for
  normal decode (1 token per step).
- **Per-token time.** Time per token is the time of one decode step, during which every sequence
  in the batch produces one token. Decide how the report will state it (per-step latency,
  aggregate tok/s, or both).
- **Use `transformer_time_f`, not the op table.** The op table lists softmax twice (`attention`
  already includes `attention_scale_softmax`), so sum the pieces of `transformer_time_f`.

## Left to do

### Required by PROJECTS.md

- [ ] **Largest batch per side (not started; design in [proposed_memory.md](proposed_memory.md)).**
  Split the memory estimate by side:
  - Attention GPU: attention weights, embedding, output head, KV cache.
  - FFN GPU: expert and router weights.

  Find the largest B that fits in 80 GiB on each. The starting point is
  `MemoryEstimator.build_memory_data` (memory_estimation.py). It returns per-layer static and
  KV-cache bytes, with `persistent_bytes_by_kind_*` for the breakdown. One option is to key the
  split off `afd_side` as well.

  *Hand estimates to check against (bf16):*
  - Expert weights: 48 × 128 × 3 × 2048 × 768 × 2 B ≈ **54.0 GiB**.
  - Attention weights: ≈ 1.7 GiB.
  - Embedding and output head: ≈ 1.2 GiB.
  - KV cache: 48 × 2 × 4 × 128 × 2 B = **96 KiB per token**, i.e. ≈ 6.75 GiB at B = 32 × 2304
    tokens.
  - Expected result: the attention GPU fits a few hundred sequences of 2304 tokens. The FFN GPU's
    limit depends only on activations.
- [ ] **Run the official evaluation.** Use B = 32, prompt 2048, output **128** (`seq_len: 2176`,
  `decode_len: 128`) and **256** (`seq_len: 2304`, `decode_len: 256`). Run all five modes for
  each output length.
  - The committed YAML still has `global_batch_size: 8`, so either set it to 32 or keep eval
    configs or a sweep script in the repo so the numbers can be reproduced.
- [ ] **Decide how to handle prefill going over memory at B = 32.** The baseline prefill peak is
  81.42 GiB. Options: report it as is with the warning, or discuss it as a limitation.

### Going further (PROJECTS.md asks for our own case studies)

- [ ] **Sweep batch size.** Find where attention catches up with the FFN, i.e. where overlap
  stops being FFN-bound.
- [ ] **Sweep context length.** Attention and KV grow with context, but the FFN doesn't.
- [ ] **Sweep link bandwidth and latency.** Find when the transfer becomes the bottleneck.
- [x] **Sensitivity to kernel-launch overhead.** Done with the MoE realism switches (see
  Decisions). Run every case study with them off and on.
- [ ] **Optional: another model or hardware**, for example a model with fewer, larger experts.

### Housekeeping

- [ ] **Tests (none yet).** At minimum, check that `both` equals the old result, that
  attention + ffn = both, and that no_overlap − both equals the hand transfer value.
- [ ] **Write the report:** setup, the modeling choices above, results tables, case studies, and
  limitations.

## How to run

```bash
uv sync    # once
uv run python run_perf.py \
  --hardware_config configs/hardware-config/course_project/a100_80GB_1gpu.yaml \
  --model_config configs/model-config/course_project/qwen3_30b_a3b_inf.yaml
```

Change `afd_side` (and `global_batch_size`, `seq_len`, `decode_len`) in the model YAML between
runs. Results go to `output/LLM/`, with peak memory in `output/LLM/memory-summary/`.

On screen, look for these lines:

- `[decode] sample step ...` gives the per-step time.
- `[prefill] time` is the prefill time.
- `[WARN] Peak memory` appears when a run goes over memory.
