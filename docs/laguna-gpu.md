# laguna on the GPU — design, measurements, and the assumptions that will break

Status: XS 2.1 works end to end on an AMD MI50 (gfx906, 32 GB) via HIP. This
records what the engine does, why, and — more usefully — which parts are sized
for XS and will not survive Laguna S or M.

    CPU only                    4.5 tok/s
    + dense weights in VRAM    10.5
    + all experts in VRAM      20.1
    + sampler fix              20.1 decode, and ~2x in the chat path

Numbers throughout are a 5950X + MI50, Laguna XS 2.1 int4, unless stated.

## What runs where

Two GPU backends are linked into `laguna`, which is unusual and deliberate:

- **`backend_cuda_ink.cu`** — a single bf16 matmul. Holds the *dense* weights:
  attention q/k/v/g/o, `lm_head`, the shared-expert MLP. 3.6 GB for XS.
- **`backend_cuda.cu`** — colibri's full backend. Holds the *routed experts* as
  resident int4 tensors and runs them through `coli_cuda_expert_group`.

Everything else — routing, RoPE, norms, KV, sampling — is CPU. Attention math is
CPU; only its projections are on the GPU.

The split exists because the two backends want different things. The ink backend
is tiny and takes f32 activations against bf16 weights. The colibri backend
already had int4 expert kernels written for GLM, and the laguna container turned
out to be byte-compatible with them (see below), so there was nothing to port.

## Residency, and the assumption that breaks

`experts_upload()` is **all-or-nothing**: either every routed expert goes to VRAM
or none does. It is attempted only when the container is int4 and the whole set
plus a 2 GB margin fits in free VRAM; otherwise the engine keeps its host-side
LRU cache and nothing changes.

That is correct for XS and wrong for everything larger:

| model | D | I | experts | per expert | all experts | fits in ~30.6 GB? |
|-------|---|---|---------|-----------|-------------|-------------------|
| XS 2.1 | 2048 | 512 | 39x256 | 1.51 MiB | **15.8 GB** | yes, with ~13 GB spare |
| S 2.1 | 3072 | 1024 | 47x256 | 4.52 MiB | **57.0 GB** | no — about half |
| M.1 | — | — | — | — | ~2x S again | no — about a quarter |

So S and M need **partial residency**, which does not exist yet. The shape it
should take, because the pieces are already here:

- The engine already keeps a per-expert usage histogram (`m->eusage`, persisted
  to `.coli_usage`) and already uses it to pin hot experts in the RAM cache
  (`pins_load`). The same ranking picks VRAM residents.
- The dashboard protocol already models three tiers — `TIERS <vram> <ram>
  <disk>` — and currently reports either 0 or all experts in VRAM. Partial
  residency is what that field was shaped for.
- `slot_find` is the natural dispatch point: resident in VRAM -> grouped GPU
  call; resident in RAM -> CPU chain; neither -> fill then CPU.

The hard part is not the mechanism, it is that a token's K routed experts will
be split across tiers, so one launch per token becomes one launch plus a CPU
remainder, and the two want different batch shapes. Measure before assuming a
split-tier token is cheaper than doing all K on the CPU.

## Things that were measured and are not guessable

**The int4 container is `fmt=2` with no repack.** Same packing (one byte per two
weights along the contraction dim, low nibble first, offset-binary) and one f32
scale per output row. `gate` and `up` are the contiguous halves of `gate_up`, so
one read yields two tensors. Do not take this on faith — `make expert-compat-test
SNAP=<model>` proves it against a double-precision decode of a *real* expert,
with a nibble-swapped null to show the check discriminates.

**`coli_cuda_expert_group` sizes its grid `max_rows * count`, not `sum(rows)`.**
Consequence: batching prefill tokens by expert does work proportional to the
*busiest* expert times the expert count. With real (skewed) routing that measured
**45% slower** at S=513 despite cutting launch count 40x. Rows-all-equal-1 is
what that kernel wants. The dense shared expert batches fine and should
(5.0s -> 0.9s on a 1025-token prefill); the routed experts must not. It also
caps at **64 experts per call** (`GroupDesc host[64]`).

**The GPU path is more accurate than the CPU path, not less.** `matmul_q4`
quantizes activations to int8 per 32-block; the GPU int4 kernel does not.
Measured 0.0378% aggregate rel-RMS over 8,112 expert chains, and the CPU is the
side contributing it. Relatedly, the ink kernel used to round activations to
bf16 to imitate `_mm512_dpbf16_ps`; on a host without AVX512-BF16 that made the
GPU the *less* accurate side, so it was dropped.

**OMP tuning matters more with a GPU, not less.** `laguna.c`'s self-tune block is
deliberately NOT gated on `COLI_CUDA` (colibri.c's is). Every routed expert is on
the CPU in the fallback path, so an untuned GPU build runs at **0.24 tok/s**
against 20 — libgomp defaults to one thread per logical CPU and expert-mm goes
0.3s -> 130s.

**HIP's `__shfl_*_sync` requires a 64-bit mask equal to the active lanes**, so
the CUDA idiom `__shfl_down_sync(0xffffffffu, ...)` is a hard compile error, not
a silent miscompute. `coli_shfl_down32` in `backend_gpu_compat.h` uses the
width-taking `__shfl_down`, which is the honest spelling anyway: 32 is a property
of the kernel, not of the wavefront.

## Chat throughput is not decode throughput

`serve_one` starts its timer before prefill and calls `state_reset` — it
**re-prefills the entire conversation every turn**. There is no KV prefix reuse.
So the tok/s the dashboard reports is `completion / (prefill + decode)` and
swings with answer length: a short reply after a long history reads far worse
than the engine is running.

Prefill is ~29 ms/token, roughly the same as decode, which is backwards. On a
1025-token prompt: **attention 16.5s, expert-mm 9.5s, shared 0.9s** of 29.1s.

Two levers, in order of value:

1. **KV prefix reuse across turns.** Eliminates the cost instead of shrinking it,
   and it is the dominant term in real chat use. `kv_persist.h` and the
   `cache_slot` plumbing exist; the engine does not use them for this.
2. **Attention.** Now ~57% of prefill. The ink matmul is one output row per
   32-lane group, no tiling, no reuse across rows.

## Failure modes that look like success

**Residency silently declining.** `coli` spawns the engine without waiting for a
previous one to release its 15.8 GB; 19 GB still in use was observed at the moment
a back-to-back run checked capacity. The capacity check now retries 5x at 2s and
both failure paths are loud, but if it does decline you get the CPU path at half
speed with no error. Tells: RSS ~11 GB instead of 1.6, a nonzero expert cache hit
rate, `fill` above 0.0s, and the dashboard VRAM tier reading 0 experts.

**A verifier that checks nothing.** `GPU_VERIFY=1` recomputes the CPU chain on the
live path and reports parity. It needed three corrections before it was worth
reading, all of them in the checker rather than the engine: a cache too small to
hold the batch's distinct experts (so K experts aliased one slot), a single
counter covering both the dense and expert paths, and an *absolute* difference
metric in a pipeline whose expert activations span ~1 to ~1e6. It reports
per-chain and magnitude-weighted relative RMS now.

**A benchmark fixture that is fiction.** The sampler's candidate pruning measured
**1.0x — no gain at all** against synthesized logits, because the invented
distribution put the bulk only ~16 below the max where nothing prunes. Real
logits span +22.5 to -14.4 and leave a median of 86 candidates out of 100,352;
the true speedup is 136x at temp 0.7. `LOGIT_DUMP=<path>` in `generate_stream`
exists so nobody has to guess the shape of that distribution again.

## Tools

    make laguna HIP=1 HIP_ARCH="gfx906 gfx1030"   # fat binary, both cards
    make ink-test HIP=1 HIP_ARCH=gfx906           # bf16 matmul vs double reference
    make expert-compat-test HIP=1 SNAP=<model>    # int4 container vs fmt=2
    GPU_VERIFY=1 ./laguna ...                     # live CPU-parity gate
    LOGIT_DUMP=<path> ./laguna ...                # capture real logit vectors
    EXPERTS_VRAM=0 ./laguna ...                   # force the CPU expert path
    GPU_DEV=<n> / COLI_GPU / COLI_GPUS / NOGPU    # device selection

`GPU_VERIFY` costs the host RAM the whole change exists to save: it needs a cache
slot per distinct expert routed across the batch, so it re-enables the full
expert cache. That is intended — a checker that cannot hold its references is
worse than none, as the three corrections above demonstrate.
