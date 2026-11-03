#!/usr/bin/env python3
"""tools/moe_surgery.py — offline surgery on MoE GGUFs for the Iluvatar pod: cut layer
prefixes, zero or drop experts, renumber blocks, without dequantizing anything.

The point of this tool (architecture experiment, 2026-11-03): JEV-35B is a qwen35moe
hybrid (SSM blocks + a full-attention block every 4th, 256 routed experts + 1 shared per
block, an MTP "nextn" tail layer). A prefix cut keeps blocks 0..K-1 verbatim — the hybrid
rhythm survives (block 3, 7, 11... stay attention), the merged decision-head rows live in
output.weight which is copied untouched, so the carved model stays a drop-in `--model-dir`
for jev_decide.py with the ORIGINAL decision_head.json / calibration.json. Whether it
still decides correctly is exactly what the experiment measures.

    python3 tools/moe_surgery.py prefix-cut \
        --src ../../models/JEV-35B-Q4_K_M.gguf --out ../../models/JEV-mini-L12.gguf --keep 12

Quantized tensors (Q4_K/Q6_K/...) are copied byte-for-byte; only the GGUF container
(header KV pairs with their exact original types, tensor infos, offsets) is rebuilt.
Everything runs at mmap speed — a 12-layer cut of the 21 GiB JEV takes ~a minute.
"""
from __future__ import annotations

import argparse
import mmap
import struct
import sys
from pathlib import Path

GGUF_MAGIC = 0x46554747
DEFAULT_ALIGN = 32

# GGUF v3 metadata value types -> (name, struct fmt or None)
META = {0: ("u8", "<B"), 1: ("i8", "<b"), 2: ("u16", "<H"), 3: ("i16", "<h"),
        4: ("u32", "<I"), 5: ("i32", "<i"), 6: ("f32", "<f"), 7: ("bool", "<?"),
        8: ("string", None), 9: ("array", None), 10: ("u64", "<Q"), 11: ("i64", "<q"),
        12: ("f64", "<d")}


class KV:
    __slots__ = ("key", "tid", "value")

    def __init__(self, key, tid, value):
        self.key, self.tid, self.value = key, tid, value


def read_exact(fh, n):
    b = fh.read(n)
    if len(b) != n:
        raise EOFError("truncated GGUF")
    return b


def parse(path: Path):
    """-> (version, [KV], [tensor infos], data_start, alignment). Tensor info is
    (name, dims, ggml_type_id, offset)."""
    kvs, tensors = [], []
    with path.open("rb") as fh:
        if struct.unpack("<I", read_exact(fh, 4))[0] != GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", read_exact(fh, 20))

        def rstr():
            (n,) = struct.unpack("<Q", read_exact(fh, 8))
            return read_exact(fh, n).decode("utf-8")

        def rval(tid):
            name, fmt = META[tid]
            if name == "string":
                return rstr()
            if name == "array":
                (etid,) = struct.unpack("<I", read_exact(fh, 4))
                (n,) = struct.unpack("<Q", read_exact(fh, 8))
                return (etid, [rval(etid) for _ in range(n)])
            (v,) = struct.unpack(fmt, read_exact(fh, struct.calcsize(fmt)))
            return v

        for _ in range(n_kv):
            key = rstr()
            (tid,) = struct.unpack("<I", read_exact(fh, 4))
            kvs.append(KV(key, tid, rval(tid)))
        for _ in range(n_tensors):
            name = rstr()
            (nd,) = struct.unpack("<I", read_exact(fh, 4))
            dims = list(struct.unpack(f"<{nd}Q", read_exact(fh, 8 * nd)))
            tid, off = struct.unpack("<IQ", read_exact(fh, 12))
            tensors.append((name, dims, tid, off))
        align = DEFAULT_ALIGN
        for kv in kvs:
            if kv.key == "general.alignment" and isinstance(kv.value, int) and kv.value:
                align = kv.value
        data_start = (fh.tell() + align - 1) // align * align
    return version, kvs, tensors, data_start, align


def wval(out, tid, value):
    name, fmt = META[tid]
    if name == "string":
        b = value.encode("utf-8")
        out.write(struct.pack("<Q", len(b)) + b)
    elif name == "array":
        etid, items = value
        out.write(struct.pack("<IQ", etid, len(items)))
        for it in items:
            wval(out, etid, it)
    else:
        out.write(struct.pack(fmt, value))


# ggml block geometry for the types these files actually contain (id -> (elems/block, bytes/block));
# 0 elems/block means a plain per-element type with the given bytes-per-element.
TYPE_SIZE = {0: (0, 4), 1: (0, 2), 8: (32, 34), 12: (256, 144), 13: (256, 176),
             14: (256, 210), 26: (0, 4), 30: (0, 2)}


def nbytes(dims, tid):
    n = 1
    for d in dims:
        n *= d
    eb, bb = TYPE_SIZE[tid]
    return n // eb * bb if eb else n * bb


def prefix_cut(src: Path, out: Path, keep: int, arch: str):
    version, kvs, tensors, data_start, align = parse(src)
    n_blk = next(int(kv.value) for kv in kvs if kv.key == f"{arch}.block_count")
    if not 0 < keep < n_blk:
        raise SystemExit(f"--keep must be in 1..{n_blk - 1} (block {n_blk - 1} is the MTP tail)")

    kept = [t for t in tensors if not t[0].startswith("blk.")
            or int(t[0].split(".")[1]) < keep]
    for kv in kvs:                                   # patch the structural keys in place
        if kv.key == f"{arch}.block_count":
            kv.value = keep
        if kv.key == f"{arch}.nextn_predict_layers":
            kv.value = 0
        # hybrid archs carry per-layer flag arrays (e.g. attention.recurrent_layers,
        # True=SSM False=full-attention) sized block_count(+nextn); slice to the cut
        if (kv.tid == 9 and isinstance(kv.value, tuple)
                and len(kv.value[1]) in (n_blk, n_blk + 1)
                and kv.key.endswith(("recurrent_layers", "attention_layers"))):
            kv.value = (kv.value[0], kv.value[1][:keep])
    out.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        with out.open("wb") as o:
            o.write(struct.pack("<IIQQ", GGUF_MAGIC, version, len(kept), len(kvs)))
            for kv in kvs:                           # KVs keep their exact original types
                b = kv.key.encode("utf-8")
                o.write(struct.pack("<Q", len(b)) + b + struct.pack("<I", kv.tid))
                wval(o, kv.tid, kv.value)
            off = 0
            for name, dims, tid, _ in kept:          # infos first, offsets relative to data
                b = name.encode("utf-8")
                o.write(struct.pack("<Q", len(b)) + b + struct.pack("<I", len(dims))
                        + struct.pack(f"<{len(dims)}Q", *dims)
                        + struct.pack("<IQ", tid, off))
                off = (off + nbytes(dims, tid) + align - 1) // align * align
            o.write(b"\0" * ((o.tell() + align - 1) // align * align - o.tell()))
            for name, dims, tid, soff in kept:       # then the payload, verbatim bytes
                sz = nbytes(dims, tid)
                o.write(mm[data_start + soff:data_start + soff + sz])
                pad = (align - sz % align) % align
                if pad:
                    o.write(b"\0" * pad)
        mm.close()
    print(f"[ok] {out}  ({out.stat().st_size / 2**30:.2f} GiB, {keep}/{n_blk} blocks, "
          f"{len(kept)} tensors kept of {len(tensors)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("prefix-cut", help="keep blocks 0..K-1 verbatim (hybrid rhythm + "
                                           "merged decision head survive); MTP tail dropped")
    pc.add_argument("--src", required=True, type=Path)
    pc.add_argument("--out", required=True, type=Path)
    pc.add_argument("--keep", required=True, type=int, help="number of leading blocks")
    pc.add_argument("--arch", default="qwen35moe")
    args = ap.parse_args()
    if args.cmd == "prefix-cut":
        prefix_cut(args.src, args.out, args.keep, args.arch)


if __name__ == "__main__":
    main()
