#!/usr/bin/env python3
"""Refuse a book that changes our move in a frequent position (exit 1).

  book_guard.py [--min 30] [--book tools/book.txt] [--base BOOK]
For every position of ours reached in at least MIN games of arena/book_games.json (the games behind the
last book.py positions), the book's move must be the move we played most there, unless that move's games
scored below 0.3 (a losing trunk may change). Catches book entries that reshuffle every game of a seat.
"""
import collections, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mn = int(sys.argv[sys.argv.index("--min") + 1]) if "--min" in sys.argv else 30
games = json.load(open(os.path.join(ROOT, "arena", "book_games.json")))
sc = {"W": 1.0, "D": 0.5, "L": 0.0}
st = collections.defaultdict(lambda: collections.defaultdict(list))
for g in games:
    for i in range(0 if g["first"] else 1, len(g["moves"]), 2):
        st[" ".join(g["moves"][:i])][g["moves"][i]].append(sc[g["res"]])
book = {}
BOOKF = sys.argv[sys.argv.index("--book") + 1] if "--book" in sys.argv else os.path.join(ROOT, "tools", "book.txt")
for l in open(BOOKF):
    f = l.split()
    if len(f) >= 3 and not f[0].startswith("#"):
        book.setdefault(f[0], f"{f[1]} {f[2]}")
base = {}   # --base BOOK: changes the base book already makes are accepted (only new ones are checked)
if "--base" in sys.argv:
    for l in open(sys.argv[sys.argv.index("--base") + 1]):
        f = l.split()
        if len(f) >= 3 and not f[0].startswith("#"):
            base.setdefault(f[0], f"{f[1]} {f[2]}")
hb = "/tmp/ut_hashbin"
if not os.path.exists(hb):
    subprocess.check_call(["g++", "-std=c++17", "-O2", "-DLOCAL", "-o", hb, os.path.join(ROOT, "out", "c24_n8t.cpp")])
bad = 0
for pre, mv in st.items():
    n = sum(len(v) for v in mv.values())
    if n < mn:
        continue
    top = max(mv, key=lambda m: len(mv[m]))
    score = sum(mv[top]) / len(mv[top])
    env = {**os.environ, "UT_ANALYZE": pre if pre else " ", "UT_FIRST_ITERS": "10", "UT_POOL": "65536", "UT_PARAMS": "PONDER=0"}
    h = subprocess.run([hb], env=env, capture_output=True, text=True).stdout.split("hash ")[1].split()[0]
    if h in book and book[h] != top and score >= 0.3 and base.get(h) != book[h]:
        bad += 1
        print(f"CHANGES TRUNK: ply {len(pre.split()) // 2} '{pre}' played {top} x{len(mv[top])} (score {score:.2f}) -> book {book[h]}")
print(f"book guard: {bad} trunk changes")
sys.exit(1 if bad else 0)
