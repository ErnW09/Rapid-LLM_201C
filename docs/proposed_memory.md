# Proposal: per-side memory and largest batch (Project 1, AFD)

**Status:** proposal, not implemented (2026-11-25, branch `proj_imp` at `183b404`). See
[current_state.md](current_state.md) for the rest of the project.

## Goal

PROJECTS.md asks for "the largest batch each side can hold: the attention GPU keeps the attention
weights and the KV cache, the FFN GPU keeps the expert weights."

Today the `afd_side` switch changes only the decode **timing**. Every mode reports the same
single-GPU memory. This proposal makes the decode memory estimate side-aware and adds a search for
the largest batch per side.

## How memory is computed today

`MemoryEstimator.build_memory_data` (`memory_estimation.py:80`) returns a dict. The memory replay
(`simulate_train_graph.py:2058-2106`) turns that dict into a peak:

- **Static memory per GPU:**
  - 48 × `weight_mem_per_layer_moe`. This is attention, expert and router parameters together
    (`memory_estimation.py:270-275`).
  - 48 × `kv_cache_bytes_per_layer` (`memory_estimation.py:639-657`).
  - `extra_static_bytes_per_device`, which holds the embedding and output head
    (`memory_estimation.py:363-371`).
- **Temporary memory:** each operation's activations, looked up by `MemKind` (`QKV_PROJ`,
  `ATTENTION`, `MLP`, `SOFTMAX`, …). They are freed as the replay moves through the layer. The
  memory graph is flattened into one node per operation, each tagged with its own kind
  (`llm_execution.py:736`), so activations can be split by kind.
- **Peak** = static + the largest temporary at any moment.

`calc_time` computes prefill memory (`inference_timing.py:890`) and decode memory
(`inference_timing.py:949`) separately. The capacity warning uses the larger of the two.

## Proposed changes

### 1. Split by side in `build_memory_data`

Add an optional argument `afd_side: Optional[str] = None`. When it is `"attention"` or `"ffn"`,
keep only what that GPU holds. Any other value (`None`, `"both"`, `"no_overlap"`, `"overlap"`)
leaves memory as it is today.

The placement follows rchen's timing split in `inference_timing.py:548-564`, so time and memory
agree:

| | Attention GPU | FFN GPU |
|---|---|---|
| Weights per layer | `attention_params` (Q, K, V, O) | `expert_params_per_rank` + `router_params_per_rank` |
| Embedding + output head | kept | 0 |
| KV cache | kept | 0 |
| Activation kinds kept | LAYERNORM1, QKV_PROJ, ATTENTION, OUTPUT_PROJ, EMBEDDING, SOFTMAX | LAYERNORM2, MLP, ROUTER, MOE_DISPATCH, MOE_COMBINE |

The function already computes each parameter count separately before adding them together, so
most of the change is choosing which ones to keep:

```python
ATTN_KINDS = {MemKind.LAYERNORM1, MemKind.QKV_PROJ, MemKind.ATTENTION,
              MemKind.OUTPUT_PROJ, MemKind.EMBEDDING, MemKind.SOFTMAX}
FFN_KINDS = {MemKind.LAYERNORM2, MemKind.MLP, MemKind.ROUTER,
             MemKind.MOE_DISPATCH, MemKind.MOE_COMBINE}
afd_split = afd_side in ("attention", "ffn")

# (a) Embedding and output head: after they are computed (~line 190), before the
#     extra_static_bytes_per_device loop (~line 363).
if afd_side == "ffn":
    embedding_static_bytes = 0.0
    lm_head_static_bytes = 0.0

# (b) Weights: at the end of the `if use_moe:` block (~line 275).
if afd_split:
    side_params = (attention_params if afd_side == "attention"
                   else expert_params_per_rank + router_params_per_rank)
    weight_memory_layer_moe = side_params * bytes_per_param_weight
    transformer_static_layer_moe = weight_memory_layer_moe   # inference: no grads/optimizer

# (c) KV cache: after the KV block (~line 657).
if afd_side == "ffn":
    kv_cache_bytes_per_layer = 0.0

# (d) Activations: after the activation maps are built (~line 588). The dense and MoE dicts
#     can be the same object, so zero each distinct dict once.
if afd_split:
    drop = FFN_KINDS if afd_side == "attention" else ATTN_KINDS
    for d in {id(m): m for m in (persistent_bytes_by_kind_dense, persistent_bytes_by_kind_moe,
                                 transient_bytes_by_kind_dense, transient_bytes_by_kind_moe)}.values():
        for kind in drop:
            if kind in d:
                d[kind] = 0.0
```

**Dense layers** are out of scope. Qwen3-30B-A3B has none (`moe_layer_freq: 1`,
`first_k_dense_replace: 0`), so only the MoE weights need splitting. If the split runs on a model
with dense layers, raise `NotImplementedError` rather than give a wrong answer silently.

### 2. Call site in `calc_time` (`inference_timing.py`)

- **Decode call only.** Pass `afd_side=self.project_param.get("afd_side")` to the decode
  `build_memory_data` call (~line 949). The prefill call doesn't get it, because the spec says to
  run prefill as usual on one GPU.
- **Report the side's decode peak.** For `attention` and `ffn` runs, add a line to
  `memory_capacity_comparison.txt` and to the console, for example
  `AFD attention GPU decode peak: 9.61 GiB (headroom 70.39 GiB)`.
  - Base that line's headroom and warning on the **decode** peak. If it used the larger of prefill
    and decode as today, the single-GPU prefill (81.42 GiB at B = 32) would trigger the warning on
    both sides.
  - Leave the existing prefill line as it is.

### 3. Find the largest batch with a sweep script

Peak decode memory is linear in B: the weights are fixed, and the KV cache, logits and activations
all scale with B. The only exception is a small rounding step from MoE token padding. So the search
needs only a few runs:

1. For each side and output length, run at B = 32 and B = 64, and read the side's decode peak.
2. Fit `M(B) = a + b·B` and solve `B_max = floor((80 GiB − a) / b)`.
3. **Attention side:** confirm with runs at `B_max` (fits) and `B_max + 1` (doesn't).
4. **FFN side:** report the fitted value and skip the confirmation run. B_max is in the hundreds
   of thousands (see below), and running prefill at that batch size means nothing.

**Proposed script:** `afd_sweep.py` at the repo root, next to `run_perf.py`. It should also run
the official evaluation, which needs the same plumbing.

- **YAML copies:** it writes copies of `qwen3_30b_a3b_inf.yaml` to a temp directory with
  `global_batch_size`, `seq_len` (= 2048 + output), `decode_len` and `afd_side` set. The committed
  config is never edited.
- **Runs:** it calls `run_perf.py` once per configuration, using the hardware config
  `a100_80GB_1gpu.yaml`.
- **Output:** it reads `output/LLM/memory-summary/memory_capacity_comparison.txt` and the
  `[decode] sample step` lines after each run.
  - `run_perf.py` has no output-directory flag, so runs must be sequential.

The KV cache is largest at the end of decode, so the attention-side limit depends on the final
context (prompt + output). It needs one answer per output length. The FFN side doesn't depend on
context.

**Alternative:** binary-search B inside `calc_time`, re-running `simulate_peak` on the same decode
graph with a new memory estimate for each B. One run would give the answer. It relies on the
graph's structure not depending on B, which hasn't been verified, so start with the script.

## Expected results (hand estimates, bf16)

These are the numbers the implementation should reproduce.

**Inputs:**

| Quantity | Calculation | Value |
|---|---|---|
| KV cache per token | 48 layers × 2 (K, V) × 4 KV heads × 128 × 2 B | 96 KiB |
| Attention weights | 48 × (2048·4096 + 2·2048·512 + 4096·2048) × 2 B | 1.69 GiB |
| Embedding + output head | 2 × 151,936 × 2048 × 2 B | 1.16 GiB |
| Expert weights | 48 × 128 × 3 × 2048 × 768 × 2 B | 54.0 GiB |
| Router weights | 48 × 2048 × 128 × 2 B | 24 MiB |
| Decode logits per sequence | 151,936 × 2 B | 0.3 MiB |
| Prefill logits per sequence | 2048 × 151,936 × 2 B | 0.58 GiB |

**These inputs match the simulator's own peaks at B = 32:**

- **Decode:** 56.87 GiB of weights + 32 × 216 MiB (KV at 2304 tokens) = **63.63 GiB**. The
  simulator reports 63.63 GiB.
- **Prefill:** 56.87 GiB + 32 × (192 MiB KV at 2048 tokens + 0.58 GiB logits) = **81.42 GiB**. The
  simulator reports 81.42 GiB.

**Largest batch:**

| Setup | Fixed memory | Per sequence | Largest B |
|---|---|---|---|
| Attention GPU, output 256 (context 2304) | 2.85 GiB | 216 MiB KV + 0.3 MiB logits | **≈ 365** |
| Attention GPU, output 128 (context 2176) | 2.85 GiB | 204 MiB KV + 0.3 MiB logits | **≈ 386** |
| FFN GPU | 54.0 GiB | activations only (tens of KiB) | **hundreds of thousands** |
| Single GPU, decode only (baseline) | 56.87 GiB | 216 MiB | ≈ 109 |
| Single GPU, including prefill (baseline) | 56.87 GiB | 0.77 GiB | ≈ 30 |

**At B = 32, a correct implementation should report:**

- Attention side: about **9.6 GiB** (2.85 + 32 × 0.211).
- FFN side: about **54.0 GiB** plus small activations.
- `both`: unchanged at **63.63 GiB**.

## What these numbers mean for the report

- **AFD raises the decode batch limit from about 109 to about 365** sequences of 2304 tokens. The
  attention GPU no longer has to hold 54 GiB of experts.
- **Memory never limits the FFN side.** Its limit is time. The FFN step costs the same at B = 16
  and B = 32 (142.5 vs 142.6 ms), so one FFN GPU could serve several attention GPUs. This
  motivates the case study on the ratio of attention GPUs to FFN GPUs.
- **The 81.42 GiB prefill warning at B = 32 is a modeling artifact.** Prefill builds logits for
  all 2048 prompt tokens (32 × 0.58 GiB ≈ 18.5 GiB), while real serving computes logits only for
  the last token. Without that tensor the prefill peak would be about 63 GiB. Note this in the
  report. Changing it is out of scope for this proposal.

## Tests

Add these to `tests/`:

1. **`both` is unchanged.** At B = 32, output 256, the decode peak is 63.63 GiB.
2. **The two sides add up.** Static memory of the attention side + the FFN side = single-GPU
   static memory.
3. **Sides match the hand values.** At B = 32, the attention decode peak is about 9.6 GiB and the
   FFN decode peak is about 54.0 GiB (within 0.1 GiB).
4. **Prefill is unaffected.** The prefill peak is identical for every `afd_side`.

## Decisions to confirm

- **Router on the FFN side.** This matches the timing split. Moving it to the attention side would
  change memory by only 24 MiB.
- **Transfer buffers ignored.** B × H × 2 B is 128 KiB at B = 32.
- **Sides are separate runs.** Both GPUs' memory are reported only from `afd_side: attention` and
  `afd_side: ffn` runs. The `overlap` and `no_overlap` modes still report single-GPU memory.
  The alternative is to have those modes print both sides' peaks.

## Files touched

| File | Change |
|---|---|
| `memory_estimation.py` | `afd_side` argument and the four zeroing steps in `build_memory_data` |
| `inference_timing.py` | Pass `afd_side` at the decode call; side-peak line in the memory summary |
| `afd_sweep.py` (new) | Largest-batch search and official evaluation runs |
| `tests/` | The four checks above |
| `docs/current_state.md` | Mark the item done once implemented |
