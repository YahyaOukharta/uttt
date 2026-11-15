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
ap.add_argument("--deep", help="float arrays of a deep head (tools/deep_export.py), test builds")
ap.add_argument("--no-minify", action="store_true", help="keep line breaks, spaces and identifiers as written")
ap.add_argument("--book", help="opening book: lines 'HASH_HEX ROW COL' (tools/book.py), most important first")
ap.add_argument("--max-chars", type=int, default=99400, help="drop book entries from the end of the file to fit")
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


def minify(src):
    """Smaller code, same token stream after preprocessing: whitespace between tokens is dropped where the
    tokens stay apart, code lines are joined, and long identifiers that occur often are replaced by two-character
    macros (#define Qx name, placed after the #includes; preprocessor lines themselves are left alone)."""
    lines, pp, code = src.split("\n"), [], []
    blocks = []   # ("pp", text) or ("code", text)
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("#") or l.startswith("//@"):
            t = l
            while t.endswith("\\") and i + 1 < len(lines):
                i += 1
                t += "\n" + lines[i]
            blocks.append(("pp", t))
        else:
            blocks.append(("code", l))
        i += 1
    tok = re.compile(r'(?:u8|u|U|L)?"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'|[A-Za-z_]\w*|\d[\w.]*|\s+|.', re.S)
    WORD = re.compile(r"\w")
    JOIN = {"++", "--", "+=", "-=", "//", "/*", "*/", "<<", ">>", "&&", "||", "::", "->", "==", "!=", "<=", ">=",
            "+-", "-+", "<:", "%:", "..", ".."}
    def tokens(t):
        return [m.group(0) for m in tok.finditer(t)]
    # identifiers worth a macro: saving (len - 2) per use minus the #define line
    import collections
    cnt = collections.Counter()
    for k, t in blocks:
        if k == "code":
            for x in tokens(t):
                if re.fullmatch(r"[A-Za-z_]\w*", x):
                    cnt[x] += 1
    used = set(cnt)
    for k, t in blocks:
        if k == "pp":
            used |= set(re.findall(r"[A-Za-z_]\w*", t))
    names = (f"Q{c}" for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    names = [n for n in (f"{a}{b}" for a in "QZ" for b in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") if n not in used]
    gain = sorted(((len(x) - 2) * n - (len(x) + 12), x) for x, n in cnt.items() if len(x) > 2 and x not in ("u",))
    ren = {}
    for g, x in reversed(gain):
        if g <= 0 or not names:
            break
        ren[x] = names.pop(0)
    out, cur = [], []
    def flush():
        if cur:
            out.append("".join(cur))
            cur.clear()
    prev = ""
    for k, t in blocks:
        if k == "pp":
            flush()
            out.append(t)
            prev = ""
            continue
        pend_space = bool(cur)   # a line break counts as whitespace
        for x in tokens(t):
            if x.isspace():
                pend_space = True
                continue
            x = ren.get(x, x)
            if cur and pend_space:
                a, b = prev[-1], x[0]
                if (WORD.match(a) and WORD.match(b)) or (a + b) in JOIN or (a in "'\"" and WORD.match(b)) or (WORD.match(a) and b in "'\""):
                    cur.append(" ")
            cur.append(x)
            prev = x
            pend_space = False
    flush()
    defs = "".join(f"#define {v} {x}\n" for x, v in ren.items())
    res = "\n".join(out)
    last = max(m.end() for m in re.finditer(r"^#include.*$", res, re.M))
    return res[:last] + "\n" + defs.rstrip("\n") + res[last:]


s = open(a.src, encoding="utf-8").read()
out, skip = [], 0
for line in s.split("\n"):
    t = line.strip()
    if skip:
        if re.match(r"#\s*if", t): skip += 1
        elif re.match(r"#\s*endif", t): skip -= 1
        continue
    if (t == "#ifdef LOCAL" and not a.keep_local) or (t == "#ifdef NN_DEEP" and not a.deep):
        skip = 1
        continue
    out.append(line)
if not a.keep_comments:
    out = [re.sub(r"\s*//(?!@NET_DATA@|@BOOK_DATA@|@DEEP_DATA@).*$", "", l) for l in out]
    out = [l.strip() for l in out if l.strip()]
s = "\n".join(out)
if not a.no_minify and not a.keep_comments:
    s = minify(s)
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
    lines = []
    for l in open(a.book):
        f = l.split()
        if len(f) < 3 or f[0].startswith("#"):
            continue
        r, c = int(f[1]), int(f[2])
        lines.append((int(f[0], 16) & 0x3FFFFFFF, (r // 3 * 3 + c // 3) * 9 + (r % 3) * 3 + c % 3))
    # room for the book: the limit minus the rest of the file (about 1.9 characters per entry)
    room = a.max_chars - len(s) - len(net) - len(defs) - 200
    keep = min(len(lines), max(0, int(room / 1.95)))
    if keep < len(lines):
        print(f"book: {len(lines) - keep} entries dropped to fit {a.max_chars} characters")
    ent = {}
    for hh, mv in lines[:keep]:
        ent.setdefault(hh, mv)
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
if a.deep:
    import re as _re
    dh = open(a.deep).read()
    n = _re.search(r"DEEP_B2\[\] = \{([^}]*)\}", dh).group(1).count(",") + 1
    assert s.count("//@DEEP_DATA@") == 1
    s = s.replace("//@DEEP_DATA@", dh)
    defs = f"#define NN_DEEP {n}\n" + defs
assert s.count("//@NET_DATA@") == 1
s = s.replace("//@NET_DATA@", net)
if defs:
    s = defs + s
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
open(a.out, "w", encoding="utf-8").write(s)
print(f"{a.out}: {len(s)} chars" + ("  ** over the 100k limit **" if len(s) > 100000 else ""))
