#!/usr/bin/env python3
"""Convert Poolside Laguna BF16 checkpoints into a colibri snapshot readable by laguna.c.

Weight mapping follows Laguna's architecture:
  - routed experts -> int4 (per-row symmetric scales) + `.qs` f32 row scales.
  - fused gate/up: per-expert gate_proj and up_proj are concatenated [gate; up]
    then stacked across experts into a single 3D [E, 2*I, D] tensor.
  - norms, router bias, and dense MLP weights -> f32 (or bf16 passthrough).
  - Attention projections and others -> bf16 passthrough.

The HF checkpoint stores experts as individual per-expert tensors:
  model.layers.N.mlp.experts.{e}.gate_proj.weight  [I, D]
  model.layers.N.mlp.experts.{e}.up_proj.weight    [I, D]
  model.layers.N.mlp.experts.{e}.down_proj.weight  [D, I]
This converter fuses them into the 3D format the C engine's container path expects:
  model.layers.N.mlp.experts.gate_up_proj  [E, 2*I, D]  (U8 when quantized)
  model.layers.N.mlp.experts.down_proj     [E, D, I]     (U8 when quantized)

Expert tensors for a single layer may span multiple input shards. This converter
uses the safetensors index to locate every tensor, then fuses across shards.

Modes:
  --indir DIR --outdir DIR [--watch]   convert local shards.
  --selftest                           numpy-only unit tests (int4).
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ---------- quantization (identical to Inkling) ----------

def quant_int8_rows(w):
    """w: f32 [O,I] -> (int8 bytes [O,I] viewed u8, f32 scales [O])"""
    s = np.abs(w).max(axis=1, keepdims=True) / 127.0
    s[s < 1e-8] = 1e-8
    q = np.clip(np.rint(w / s), -128, 127).astype(np.int8)
    return q.view(np.uint8), s[:, 0].astype(np.float32)


def quant_int4_rows(w):
    """w: f32 [O,I], I even -> (packed u8 [O,I/2], f32 scales [O]).
    Low nibble = even column, high nibble = odd column, offset +8."""
    O, I = w.shape
    assert I % 2 == 0
    s = np.abs(w).max(axis=1, keepdims=True) / 7.0
    s[s < 1e-8] = 1e-8
    q = np.clip(np.rint(w / s), -8, 7).astype(np.int32)
    lo = (q[:, 0::2] + 8).astype(np.uint8)
    hi = (q[:, 1::2] + 8).astype(np.uint8)
    return (lo | (hi << 4)), s[:, 0].astype(np.float32)


# ---------- name mapping (Laguna HF -> colibri) ----------

SKIP_PREFIXES = ("model.audio.", "model.visual.", "model.mtp.")

# f32 targets (small / numerically sensitive)
F32_RE = re.compile(
    r"(norm\.weight$|layernorm\.weight$|e_score_correction_bias$)"
)

# Regex to detect per-expert tensors in the HF checkpoint
EXPERT_TENSOR_RE = re.compile(
    r"^(model\.layers\.\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def map_name(name):
    if name.startswith(SKIP_PREFIXES):
        return None
    # Laguna HF names already match what the C engine expects.
    return name


# ---------- 3D expert quantization ----------

def quantize_expert_3d(w3d, xbits, out, out_name):
    """Quantize a 3D expert tensor [E, R, C] into packed rows + per-row scales.

    Works for both gate_up_proj [E, 2*I, D] and down_proj [E, D, I].
    Output: out[out_name] = flat U8 packed bytes, out[out_name + ".qs"] = f32 scales [E*R].
    """
    from concurrent.futures import ThreadPoolExecutor
    E, R, C = w3d.shape
    if xbits == 4:
        qs = np.empty((E * R, C // 2), np.uint8)
    else:
        qs = np.empty((E * R, C), np.uint8)
    sc = np.empty(E * R, np.float32)

    def work(e):
        w = w3d[e].float().numpy()
        q, s = quant_int4_rows(w) if xbits == 4 else quant_int8_rows(w)
        qs[e * R:(e + 1) * R] = q
        sc[e * R:(e + 1) * R] = s

    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(work, e) for e in range(E)]
        for fu in futs:
            fu.result()
    out[out_name] = torch.from_numpy(qs.reshape(-1))
    out[out_name + ".qs"] = torch.from_numpy(sc)


# ---------- cross-shard expert fusion ----------

def fuse_and_quantize_experts(indir, index_map, xbits, outdir):
    """Load per-expert tensors across shards, fuse into 3D, quantize, and write
    one output safetensors file per batch of layers.

    index_map: {tensor_name: shard_filename}
    """
    # Group expert tensors by layer
    layer_experts = {}  # {layer_prefix: {expert_idx: {proj: tensor_name}}}
    for name, shard in index_map.items():
        m = EXPERT_TENSOR_RE.match(name)
        if m:
            lp, eid, proj = m.group(1), int(m.group(2)), m.group(3)
            layer_experts.setdefault(lp, {}).setdefault(eid, {})[proj] = name

    if not layer_experts:
        return

    # Cache open shard file handles
    shard_cache = {}  # {shard_path: safe_open handle}

    def get_shard(shard_name):
        path = os.path.join(indir, shard_name)
        if path not in shard_cache:
            shard_cache[path] = safe_open(path, framework="pt")
        return shard_cache[path]

    # Process each layer
    out = {}
    out_idx = 0
    n_layers_done = 0
    total_layers = len(layer_experts)

    for lp in sorted(layer_experts.keys()):
        experts = layer_experts[lp]
        E = max(experts.keys()) + 1
        t0 = time.time()

        gate_up_list = []
        down_list = []
        for e in range(E):
            ed = experts[e]
            g_shard = index_map[ed["gate_proj"]]
            u_shard = index_map[ed["up_proj"]]
            d_shard = index_map[ed["down_proj"]]
            g = get_shard(g_shard).get_tensor(ed["gate_proj"])
            u = get_shard(u_shard).get_tensor(ed["up_proj"])
            d = get_shard(d_shard).get_tensor(ed["down_proj"])
            gate_up_list.append(torch.cat([g, u], dim=0))  # [2*I, D]
            down_list.append(d)                            # [D, I]

        gate_up_3d = torch.stack(gate_up_list, dim=0)  # [E, 2*I, D]
        down_3d = torch.stack(down_list, dim=0)        # [E, D, I]

        gu_name = f"{lp}.mlp.experts.gate_up_proj"
        dn_name = f"{lp}.mlp.experts.down_proj"
        if xbits:
            quantize_expert_3d(gate_up_3d, xbits, out, gu_name)
            quantize_expert_3d(down_3d, xbits, out, dn_name)
        else:
            out[gu_name] = gate_up_3d.contiguous()
            out[dn_name] = down_3d.contiguous()

        n_layers_done += 1
        el = time.time() - t0
        print(f"  {lp}: {E} experts fused{' + quantized' if xbits else ''} "
              f"({el:.0f}s, {n_layers_done}/{total_layers})", flush=True)

        # Write in batches to keep memory bounded.
        # Each MoE layer's experts are ~256 * (2*512*2048 + 2048*512) * 2 bytes = ~3GB bf16.
        # After quantization (int4): ~0.5GB + scales. After passthrough: ~3GB.
        # Write every 4 layers to keep output files reasonable.
        if n_layers_done % 4 == 0 or n_layers_done == total_layers:
            dst = os.path.join(outdir, f"out-experts-{out_idx:05d}.safetensors")
            save_file(out, dst + ".tmp")
            os.replace(dst + ".tmp", dst)
            gb = os.path.getsize(dst) / 1e9
            print(f"  wrote {os.path.basename(dst)} ({gb:.1f}G)", flush=True)
            out = {}
            out_idx += 1

    # Close all shard handles
    for f in shard_cache.values():
        del f
    shard_cache.clear()


# ---------- per-shard conversion (non-expert tensors) ----------

def convert_non_expert_shard(path, out, xbits):
    """Process all non-expert tensors in a single shard."""
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            if name.startswith(SKIP_PREFIXES):
                continue
            # Skip per-expert tensors (handled by cross-shard fusion)
            if EXPERT_TENSOR_RE.match(name):
                continue

            base = map_name(name)
            if base is None:
                continue

            t = f.get_tensor(name)
            if F32_RE.search(base):
                out[base] = t.float()
            else:
                out[base] = t  # bf16 passthrough


AUX_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "chat_template.jinja"]


def shard_out_name(src):
    m = re.search(r"model-(\d+)-of-\d+\.safetensors$", os.path.basename(src))
    return f"out-{m.group(1)}.safetensors" if m else "out-" + os.path.basename(src)


def convert_dir(indir, outdir, xbits, watch=False, delete_src=False):
    os.makedirs(outdir, exist_ok=True)
    index_path = os.path.join(indir, "model.safetensors.index.json")

    # Load the index to get the global tensor->shard map.
    # If there's no index (single-file checkpoint), build one from the file itself.
    index_map = None
    if os.path.exists(index_path):
        index_map = json.load(open(index_path))["weight_map"]
    else:
        single = os.path.join(indir, "model.safetensors")
        if os.path.exists(single):
            with safe_open(single, framework="pt") as f:
                index_map = {k: "model.safetensors" for k in f.keys()}

    # Phase 1: convert non-expert tensors shard by shard
    print("Phase 1: converting non-expert tensors (attn, norms, router, dense MLP, embed)...",
          flush=True)
    done_src = set()
    while True:
        shards = sorted(glob.glob(os.path.join(indir, "model-*.safetensors")))
        if not shards:
            single = os.path.join(indir, "model.safetensors")
            if os.path.exists(single):
                shards = [single]
        for sp in shards:
            if sp in done_src:
                continue
            dst = os.path.join(outdir, shard_out_name(sp))
            if os.path.exists(dst):
                done_src.add(sp)
                continue
            t0 = time.time()
            out = {}
            convert_non_expert_shard(sp, out, xbits)
            if out:
                save_file(out, dst + ".tmp")
                os.replace(dst + ".tmp", dst)
            else:
                open(dst, "wb").close()
            done_src.add(sp)
            src_gb = os.path.getsize(sp) / 1e9
            dst_gb = os.path.getsize(dst) / 1e9
            print(f"  {os.path.basename(sp)} ({src_gb:.1f}G) -> "
                  f"{os.path.basename(dst)} ({dst_gb:.1f}G) in {time.time()-t0:.0f}s",
                  flush=True)
            # Remove 0-byte output files (shards that held only expert tensors)
            if os.path.getsize(dst) == 0:
                os.remove(dst)
            if delete_src:
                os.remove(sp)
        if not watch:
            break
        # Check completion for non-expert shards
        if index_map:
            total = sorted(s for s in set(index_map.values())
                           if re.match(r"model-\d+-of-\d+\.safetensors$", s))
            if total and all(os.path.exists(os.path.join(outdir, shard_out_name(s))) for s in total):
                break
        time.sleep(60)

    # Phase 2: fuse + quantize expert tensors across shards
    if index_map:
        # Filter to only tensors that actually exist (some may be in incomplete downloads)
        existing = {}
        for k, v in index_map.items():
            if os.path.exists(os.path.join(indir, v)):
                existing[k] = v
        # Check if there are expert tensors to process
        has_experts = any(EXPERT_TENSOR_RE.match(k) for k in existing)
        if has_experts:
            print(f"\nPhase 2: fusing + {'quantizing' if xbits else 'converting'} "
                  f"expert tensors across shards...", flush=True)
            fuse_and_quantize_experts(indir, existing, xbits, outdir)

    # Copy aux files
    for fn in AUX_FILES:
        src = os.path.join(indir, fn)
        if os.path.exists(src):
            shutil.copy(src, outdir)
    print(f"\nconversion complete -> {outdir}")


def selftest():
    # 1) int4 pack/unpack round-trip against the C engine's convention
    w = np.random.randn(16, 32).astype(np.float32)
    q, s = quant_int4_rows(w)
    lo = (q & 0x0F).astype(np.int32) - 8
    hi = ((q >> 4) & 0x0F).astype(np.int32) - 8
    deq = np.empty_like(w)
    deq[:, 0::2] = lo
    deq[:, 1::2] = hi
    deq *= s[:, None]
    ref = np.clip(np.rint(w / s[:, None]), -8, 7) * s[:, None]
    assert np.array_equal(deq, ref), "int4 round-trip mismatch"
    # 2) name mapping spot checks
    cases = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
        "model.layers.0.input_layernorm.weight": "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight": "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.g_proj.weight": "model.layers.0.self_attn.g_proj.weight",
        "model.layers.0.mlp.gate_proj.weight": "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.down_proj.weight": "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.mlp.gate.weight": "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.gate.e_score_correction_bias": "model.layers.1.mlp.gate.e_score_correction_bias",
        "model.mtp.layers.0.input_proj.weight": None,
        "model.audio.encoder.weight": None,
    }
    for src, want in cases.items():
        got = map_name(src)
        assert got == want, f"map_name({src}) = {got}, want {want}"
    # 3) expert tensor regex
    m = EXPERT_TENSOR_RE.match("model.layers.3.mlp.experts.5.gate_proj.weight")
    assert m and m.group(1) == "model.layers.3" and int(m.group(2)) == 5 and m.group(3) == "gate_proj"
    m = EXPERT_TENSOR_RE.match("model.layers.3.mlp.experts.5.down_proj.weight")
    assert m and m.group(3) == "down_proj"
    assert not EXPERT_TENSOR_RE.match("model.layers.3.mlp.experts.e_score_correction_bias")
    assert not EXPERT_TENSOR_RE.match("model.layers.3.mlp.experts.gate_up_proj")
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir")
    ap.add_argument("--outdir")
    ap.add_argument("--xbits", type=int, default=4, choices=[0, 4, 8],
                    help="routed-expert bits: 4 (default), 8, or 0 = bf16 passthrough")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--delete-src", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not a.indir or not a.outdir:
        ap.error("--indir and --outdir required")
    convert_dir(a.indir, a.outdir, a.xbits, watch=a.watch, delete_src=a.delete_src)


if __name__ == "__main__":
    main()