# Project 1 (AFD): current state

As of 2026-11-25, branch `proj_imp` at `183b404`.

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
| `no_overlap` | attention + transfer + FFN | kept, added after the layers |
| `overlap` | max(attention, transfer, FFN) | kept, added after the layers |

The transfer term is `2 × (latency + B × H × bytes_per_activation / bandwidth)`, which is one round
trip of the hidden state per layer.

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

## Things to know when interpreting results

- **Decode is FFN-bound.** The FFN is about 93% of the step at B = 32. About 75% of the FFN time
  is kernel-launch overhead: 9 µs (`kernel_launch_overhead` in the hardware YAML) × 128 experts ×
  2 GEMMs per layer. `_batched_gemm_forward_compute` (train_timing.py) multiplies the per-expert
  time, launch overhead included, by the expert count. As a result, `overlap` ≈ `ffn`, and AFD's
  speedup over single-GPU is small (~6%). The report should flag that this is partly a modeling
  artifact.
- **Overlap mode counts the round trip as one stage.** The transfer inside the `max` is both
  directions together. PROJECTS.md says "the slowest of the three", so this reading is defensible,
  but the report should say so.
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
- [ ] **Sensitivity to kernel-launch overhead.** Use a model-YAML-side workaround, since the
  hardware configs must not be edited. This would separate the launch artifact from real
  weight-read cost.
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
