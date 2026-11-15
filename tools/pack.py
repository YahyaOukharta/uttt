#!/usr/bin/env python3
"""Build the single-file CodinGame submission.

  pack.py [-n NET.bin -H hidden] [-D NAME=VALUE ...] [-o out/submission.cpp] [--keep-local] [SRC]

Strips `#ifdef LOCAL ... #endif` blocks (unless --keep-local), comments, blank lines and
indentation, bakes -D overrides in as #defines, and embeds the net as a UTF-16 string literal:
a bit stream at 15 bits per character, all weights as int8 (the output layer is clipped to
[-127, 127] in training), output biases as int32 (CodinGame's 100k limit counts characters, not bytes).
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
ap.add_argument("--book", help="opening book: lines 'HASH_HEX ROW COL' (tools/book.py)")
ap.add_argument("--huff", action="store_true", help="Huffman-code the int8 weights (canonical code, ~6.7 bits each)")
a = ap.parse_args()

def huff_stream(w, words):
    """Bits (LSB-first per field): output-bias words (16 bits each), 256 code lengths (5 bits each,
    symbol = int8 value as a byte), then each weight's canonical Huffman code, most significant bit first."""
    import heapq
    sym = w.view(np.uint8).astype(np.int64)
    freq = np.bincount(sym, minlength=256)
    heap = [(int(f), i, (i,)) for i, f in enumerate(freq) if f]
    heapq.heapify(heap)
    length = np.zeros(256, dtype=np.int64)
    uid = 256
    while len(heap) > 1:
        f1, _, s1 = heapq.heappop(heap)
        f2, _, s2 = heapq.heappop(heap)
        for x in s1 + s2:
            length[x] += 1
        heapq.heappush(heap, (f1 + f2, uid, s1 + s2))
        uid += 1
    if len(heap) == 1 and length.max() == 0:
        length[heap[0][2][0]] = 1
    assert length.max() <= 31, "code too long"
    code = np.zeros(256, dtype=np.int64)          # canonical: by (length, symbol)
    c, prev = 0, 0
    for x in sorted((int(length[i]), i) for i in range(256) if length[i]):
        c <<= x[0] - prev
        prev = x[0]
        code[x[1]] = c
        c += 1
    def lsb(v, n):
        return [(int(v) >> i) & 1 for i in range(n)]
    head = []
    for x in words:
        head += lsb(x, 16)
    for i in range(256):
        head += lsb(length[i], 5)
    table = [np.array([(int(code[i]) >> (int(length[i]) - 1 - k)) & 1 for k in range(int(length[i]))], dtype=np.uint8)
             for i in range(256)]
    body = np.concatenate([table[x] for x in sym.tolist()])
    print(f"huffman: {len(sym)} weights, {len(body) / len(sym):.2f} bits each")
    return np.concatenate([np.array(head, dtype=np.uint8), body])


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
    out = [re.sub(r"\s*//(?!@NET_DATA@|@BOOK_DATA@).*$", "", l) for l in out]
    out = [l.strip() for l in out if l.strip()]
s = "\n".join(out)
defs = "".join(f"#define {d.replace('=', ' ', 1)}\n" for d in a.D)
net = ""
if a.net:
    h = a.hidden
    raw = open(a.net, "rb").read()
    dsz = NB * 2 * h * 2 + 4 * NB
    for nin in (NIN, NIN + 18):                 # 207 inputs = with small-board threat features
        n16 = nin * h + h + NB * 2 * h
        base = n16 * 2 + 4 * NB
        if len(raw) in (base, base + dsz):
            break
    assert len(raw) in (base, base + dsz), f"net size mismatch for hidden={h}"
    thr = nin > NIN
    draw = len(raw) == base + dsz               # second head: P(draw)
    v = np.frombuffer(raw[:n16 * 2], dtype="<i2").astype(np.int64)
    ob = np.frombuffer(raw[n16 * 2:base], dtype="<i4").astype(np.int64)
    if draw:
        assert a.huff, "the draw head needs --huff"
        v = np.concatenate([v, np.frombuffer(raw[base:base + NB * 2 * h * 2], dtype="<i2").astype(np.int64)])
        ob = np.concatenate([ob, np.frombuffer(raw[base + NB * 2 * h * 2:], dtype="<i4").astype(np.int64)])
    w8 = v                                      # feature weights, biases and output weights -> int8
    rest = np.stack([ob >> 16, ob & 0xFFFF], axis=1).reshape(-1)
    clipped = int((np.abs(w8) > 127).sum())
    if clipped:
        print(f"warning: {clipped} weights outside int8, clipped (train with --wclip)")
    if a.huff:
        bits = huff_stream(np.clip(w8, -127, 127).astype(np.int8), rest & 0xFFFF)
    else:
        data = np.clip(w8, -127, 127).astype(np.int8).tobytes() + (rest & 0xFFFF).astype("<u2").tobytes()
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
    bits = np.concatenate([bits, np.zeros((-len(bits)) % 15, dtype=np.uint8)]).reshape(-1, 15)
    codes = (bits * (1 << np.arange(15))).sum(1) + OFF
    text = "".join(map(chr, codes.tolist()))
    net = (f'#define NN_EMBED 1\n' + ('#define NN_HUFF 1\n' if a.huff else '') + ('#define NN_DRAW 1\n' if draw else '') +
           f'#define NN_OFF {OFF}\nstatic const char16_t NN_DATA[] = u"{text}";\n')
    defs = f"#define NN_H {h}\n" + ("#define NN_THR 1\n" if thr else "") + defs
if a.book:   # sorted 30-bit hashes, Rice-coded gaps + 7-bit move (move = board * 9 + cell), 15 bits per character
    ent = {}
    for l in open(a.book):
        f = l.split()
        if len(f) < 3 or f[0].startswith("#"):
            continue
        r, c = int(f[1]), int(f[2])
        ent.setdefault(int(f[0], 16) & 0x3FFFFFFF, (r // 3 * 3 + c // 3) * 9 + (r % 3) * 3 + c % 3)
    hs = sorted(ent)
    def rice(k):
        out, prev = [], 0
        for hh in hs:
            d = hh - prev
            prev = hh
            out += [1] * (d >> k) + [0] + [d >> b & 1 for b in range(k)] + [ent[hh] >> b & 1 for b in range(7)]
        return out
    k = min(range(8, 28), key=lambda k: len(rice(k)))
    bits = rice(k)
    bits += [0] * (-len(bits) % 15)
    chars = [chr(OFF + sum(bits[i + b] << b for b in range(15))) for i in range(0, len(bits), 15)]
    assert s.count("//@BOOK_DATA@") == 1
    s = s.replace("//@BOOK_DATA@", f'#define BOOK_OFF {OFF}\n#define BOOK_N {len(hs)}\n#define BOOK_K {k}\n'
                                   f'static const char16_t BOOK[] = u"{"".join(chars)}";')
    print(f"book: {len(hs)} entries, {len(chars)} characters (Rice k = {k})")
assert s.count("//@NET_DATA@") == 1
s = s.replace("//@NET_DATA@", net)
if defs:
    s = defs + s
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
open(a.out, "w", encoding="utf-8").write(s)
print(f"{a.out}: {len(s)} chars" + ("  ** over the 100k limit **" if len(s) > 100000 else ""))
