#!/usr/bin/env python3
"""Build the single-file CodinGame submission.

  pack.py [-n NET.bin -H hidden] [-D NAME=VALUE ...] [-o out/submission.cpp] [--keep-local] [SRC]

Strips `#ifdef LOCAL ... #endif` blocks (unless --keep-local), comments, blank lines and
indentation, bakes -D overrides in as #defines, and embeds the net as a UTF-16 string literal:
a bit stream at 15 bits per character, first-layer weights and biases as int8, the output layer
as int16 (CodinGame's 100k limit counts characters, not bytes).
"""
import argparse, os, re
import numpy as np

OFF = 0x2100  # 15-bit value + OFF stays in [0x2100, 0xA100): no controls, quotes, line separators or surrogates
NIN, NB = 189, 10
ap = argparse.ArgumentParser()
ap.add_argument("src", nargs="?", default="bot.cpp")
ap.add_argument("-n", "--net")
ap.add_argument("-H", "--hidden", type=int, default=256)
ap.add_argument("-D", action="append", default=[])
ap.add_argument("-o", "--out", default="out/submission.cpp")
ap.add_argument("--keep-comments", action="store_true")
ap.add_argument("--keep-local", action="store_true", help="keep LOCAL blocks (test the embedded net locally)")
a = ap.parse_args()

s = open(a.src, encoding="utf-8").read()
out, skip = [], 0
for line in s.split("\n"):
    t = line.strip()
    if skip:
        if re.match(r"#\s*if", t): skip += 1
        elif re.match(r"#\s*endif", t): skip -= 1
        continue
    if t == "#ifdef LOCAL" and not a.keep_local:
        skip = 1
        continue
    out.append(line)
if not a.keep_comments:
    out = [re.sub(r"\s*//(?!@NET_DATA@).*$", "", l) for l in out]
    out = [l.strip() for l in out if l.strip()]
s = "\n".join(out)
defs = "".join(f"#define {d.replace('=', ' ', 1)}\n" for d in a.D)
net = ""
if a.net:
    h = a.hidden
    raw = open(a.net, "rb").read()
    n16 = NIN * h + h + NB * 2 * h
    assert len(raw) == n16 * 2 + 4 * NB, f"net size mismatch for hidden={h}"
    v = np.frombuffer(raw[:n16 * 2], dtype="<i2").astype(np.int64)
    ob = np.frombuffer(raw[n16 * 2:], dtype="<i4").astype(np.int64)
    w8 = v[:NIN * h + h]                        # feature weights + biases -> int8
    rest = np.concatenate([v[NIN * h + h:], np.stack([ob >> 16, ob & 0xFFFF], axis=1).reshape(-1)])
    clipped = int((np.abs(w8) > 127).sum())
    if clipped:
        print(f"warning: {clipped} first-layer weights outside int8, clipped (train with --wclip)")
    data = np.clip(w8, -127, 127).astype(np.int8).tobytes() + (rest & 0xFFFF).astype("<u2").tobytes()
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
    bits = np.concatenate([bits, np.zeros((-len(bits)) % 15, dtype=np.uint8)]).reshape(-1, 15)
    codes = (bits * (1 << np.arange(15))).sum(1) + OFF
    text = "".join(map(chr, codes.tolist()))
    net = f'#define NN_EMBED 1\n#define NN_OFF {OFF}\nstatic const char16_t NN_DATA[] = u"{text}";\n'
    defs = f"#define NN_H {h}\n" + defs
assert s.count("//@NET_DATA@") == 1
s = s.replace("//@NET_DATA@", net)
if defs:
    s = defs + s
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
open(a.out, "w", encoding="utf-8").write(s)
print(f"{a.out}: {len(s)} chars" + ("  ** over the 100k limit **" if len(s) > 100000 else ""))
