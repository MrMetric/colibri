#!/usr/bin/env python3
"""Build a tiny random-weight Laguna model + oracle fixture for laguna.c.

Stage-A validation, mirroring the Inkling/OLMoE ref.json flow: saves a small
LagunaForCausalLM snapshot (safetensors + config.json) and a ref_laguna.json
with {prompt_ids, full_ids, tf_pred} from HF transformers, which the C engine
must reproduce token-for-token (run with bits=0 for f32-exact experts).

The tiny config exercises every architectural branch of the big model:
sliding + global attention layers, dense + sparse MLP layers, 1 shared expert,
per-head softplus gating, YaRN RoPE on full-attention layers, partial rotary.

Requires poolside/Laguna-XS-2.1 config code (trust_remote_code):
  pip install transformers
  The configuration_laguna.py / modeling_laguna.py must be importable — either
  from a local clone or via trust_remote_code on a downloaded checkpoint.

Usage: python3 make_tiny_laguna.py <outdir>
"""
import json
import os
import sys

import torch

# Add the ref dir if it's present (downloaded alongside this script)
here = os.path.dirname(os.path.abspath(__file__))
for cand in (here, "/tmp/laguna_ref"):
    if os.path.exists(os.path.join(cand, "configuration_laguna.py")):
        # modeling_laguna.py uses `from .configuration_laguna import ...` (relative),
        # so we need the *parent* of laguna_ref/ on sys.path, then import as a package.
        parent = os.path.dirname(cand)
        pkg = os.path.basename(cand)
        # Create an __init__.py if missing so it's importable as a package
        init = os.path.join(cand, "__init__.py")
        if not os.path.exists(init):
            open(init, "w").close()
        sys.path.insert(0, parent)
        mod_cfg = f"{pkg}.configuration_laguna"
        mod_model = f"{pkg}.modeling_laguna"
        break
else:
    mod_cfg = "configuration_laguna"
    mod_model = "modeling_laguna"

try:
    LagunaConfig = __import__(mod_cfg, fromlist=["LagunaConfig"]).LagunaConfig
    LagunaForCausalLM = __import__(mod_model, fromlist=["LagunaForCausalLM"]).LagunaForCausalLM
except ImportError as e:
    sys.exit(f"Cannot import Laguna model code: {e}\n"
             f"  hf download poolside/Laguna-XS-2.1 configuration_laguna.py modeling_laguna.py --local-dir /tmp/laguna_ref")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "tiny_laguna"
    torch.manual_seed(0)

    cfg = LagunaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=96,           # dense MLP
        num_hidden_layers=8,
        num_attention_heads=4,          # base head count (global layers)
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=8,               # SWA window
        layer_types=[
            "full_attention",           # 0
            "sliding_attention",         # 1
            "sliding_attention",         # 2
            "sliding_attention",         # 3
            "full_attention",            # 4
            "sliding_attention",         # 5
            "sliding_attention",         # 6
            "sliding_attention",         # 7
        ],
        mlp_only_layers=[0],           # layer 0 = dense MLP
        num_attention_heads_per_layer=[4, 6, 6, 6, 4, 6, 6, 6],  # per-layer heads
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        norm_topk_prob=True,
        moe_routed_scaling_factor=2.5,
        gating="per-head",
        rope_parameters={
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 10000.0,
                "factor": 4.0,
                "original_max_position_embeddings": 64,
                "beta_slow": 1.0,
                "beta_fast": 32.0,
                "attention_factor": 1.2,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
        },
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        bos_token_id=2,
        eos_token_id=[2, 24],
        pad_token_id=9,
        tie_word_embeddings=False,
    )

    model = LagunaForCausalLM(cfg).eval().float()

    prompt = [7, 42, 199, 3, 88, 154, 21, 60, 9, 133, 77, 245]
    ids = torch.tensor([prompt], dtype=torch.long)
    n_new = 24

    with torch.no_grad():
        # Generate WITHOUT cache (the C engine processes all tokens per step,
        # so we match that path — use_cache=True can diverge due to internal
        # state changes between the generate and tf forward pass)
        full = list(prompt)
        for _ in range(n_new):
            logits = model(torch.tensor([full], dtype=torch.long)).logits[0, -1]
            full.append(logits.argmax().item())
        tf = model(torch.tensor([full], dtype=torch.long)).logits[0].argmax(-1).tolist()

    model.save_pretrained(out, safe_serialization=True)
    ref = {"prompt_ids": prompt, "full_ids": full, "tf_pred": tf}
    with open(f"{out}/ref_laguna.json", "w") as f:
        json.dump(ref, f)
    print(f"saved tiny model + ref_laguna.json to {out}/")
    print("continuation:", full[len(prompt):])


if __name__ == "__main__":
    main()