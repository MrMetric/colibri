#!/usr/bin/env python3
"""Extract one routed expert from a laguna int4 container into a flat file.

Feeds tests/test_expert_int4_compat.c, which needs a REAL expert rather than a
synthesized one: the question that test answers is whether the bytes the
converter actually wrote are laid out the way backend_cuda.cu's fmt=2 expects,
and a fixture built from our own understanding of the format could not answer
it (it would share the assumption under test).

Layout written:
    int32 D, I, O13, O2
    uint8 gate_up weights  [O13 rows x D/2 bytes]
    f32   gate_up scales   [O13]
    uint8 down weights     [O2 rows x I/2 bytes]
    f32   down scales      [O2]

Usage: dump_expert.py <snapshot-dir> [layer] [expert] [out.bin]
"""
import glob
import json
import os
import struct
import sys


def load_index(snap):
    """name -> (file, absolute byte offset of the data, nbytes)."""
    index = {}
    for path in sorted(glob.glob(os.path.join(snap, "out-experts-*.safetensors"))):
        with open(path, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(hlen))
        base = 8 + hlen
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            index[name] = (path, base + start, end - start)
    if not index:
        sys.exit(f"no out-experts-*.safetensors under {snap}")
    return index


def read_at(entry, offset, nbytes):
    path, base, total = entry
    if offset + nbytes > total:
        sys.exit(f"slice {offset}+{nbytes} runs past the {total}-byte tensor in {path}")
    with open(path, "rb") as fh:
        fh.seek(base + offset)
        data = fh.read(nbytes)
    if len(data) != nbytes:
        sys.exit(f"short read from {path}: wanted {nbytes}, got {len(data)}")
    return data


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    snap = sys.argv[1]
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    eid = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    out = sys.argv[4] if len(sys.argv) > 4 else "expert.bin"

    cfg = json.load(open(os.path.join(snap, "config.json")))
    D = cfg["hidden_size"]
    I = cfg["moe_intermediate_size"]
    rb13, rb2 = D // 2, I // 2          # int4: two weights per byte along the contraction dim

    index = load_index(snap)
    gu = f"model.layers.{layer}.mlp.experts.gate_up_proj"
    dn = f"model.layers.{layer}.mlp.experts.down_proj"
    for name in (gu, gu + ".qs", dn, dn + ".qs"):
        if name not in index:
            sys.exit(f"{name} not in the container — is layer {layer} dense rather than sparse?")

    w13 = read_at(index[gu], eid * 2 * I * rb13, 2 * I * rb13)
    s13 = read_at(index[gu + ".qs"], eid * 2 * I * 4, 2 * I * 4)
    w2 = read_at(index[dn], eid * D * rb2, D * rb2)
    s2 = read_at(index[dn + ".qs"], eid * D * 4, D * 4)

    with open(out, "wb") as fh:
        fh.write(struct.pack("<4i", D, I, 2 * I, D))
        fh.write(w13)
        fh.write(s13)
        fh.write(w2)
        fh.write(s2)
    print(f"layer {layer} expert {eid}: gate_up[{2*I},{D}] down[{D},{I}] -> {out} "
          f"({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
