#!/usr/bin/env python3
"""Refresh the best version's book against the opponents' current code, safely.

  refresh.py SUB [--base tools/book_v115.txt] [--out /tmp/refresh_book.txt]
Takes the entries of a fresh book.py build (tools/book.txt) only at positions of ours where every game of
submission SUB (the best version in the arena) was lost, and puts them before the base book. Positions
where SUB scored anything keep the base book's (or the search's) move.
"""
import collections, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
a = sys.argv[1:]
sub = int(a[0])
base_f = a[a.index("--base") + 1] if "--base" in a else os.path.join(ROOT, "tools", "book_v115.txt")
out_f = a[a.index("--out") + 1] if "--out" in a else "/tmp/refresh_book.txt"
st = collections.defaultdict(collections.Counter)
for l in open(os.path.join(ROOT, "arena", "games.jsonl")):
    g = json.loads(l)
    if g["submission"] != sub:
        continue
    r = "W" if g["won"] else "D" if g["draw"] else "L"
    for i in range(0 if g["me_first"] else 1, len(g["moves"]), 2):
        st[" ".join(g["moves"][:i])][r] += 1
lost = {p for p, c in st.items() if c["L"] and not c["W"] and not c["D"]}
h2p = {}
for f in ("book_an.jsonl", "book_an_deep.jsonl"):
    p = os.path.join(ROOT, "arena", f)
    if os.path.exists(p):
        for l in open(p):
            x = json.loads(l)
            h2p[x["hash"]] = x["prefix"]
base = {}
for l in open(base_f):
    f = l.split()
    if len(f) >= 3 and not f[0].startswith("#"):
        base.setdefault(f[0], f"{f[1]} {f[2]}")
delta = []
for l in open(os.path.join(ROOT, "tools", "book.txt")):
    f = l.split()
    if len(f) < 3 or f[0].startswith("#"):
        continue
    pre = h2p.get(f[0])
    if pre in lost and base.get(f[0]) != f"{f[1]} {f[2]}":
        delta.append(l.rstrip("\n"))
open(out_f, "w").write("\n".join(delta) + "\n" + open(base_f).read())
print(f"{len(lost)} lost-only positions in {sub}'s games; {len(delta)} new entries + {len(base)} base -> {out_f}")
