#!/usr/bin/env python3
"""Opening book from arena games: many opponents (and our bot) play deterministically, so the
same lost games repeat. Search every position of our non-won games much longer than a turn
allows and answer the ones where the long search prefers another move.

  book.py positions SUB [SUB ...]          our positions in lost/drawn games -> arena/book_pos.json
  modal run tools/modal_book.py --src out/x.cpp [--iters 4000000]   long searches -> arena/book_an.jsonl
  book.py build [--gain 0.02] [--draw-gain 0.05]                    -> tools/book.txt (HASH ROW COL)

Positions that occur in a won game of ours are left alone.
Logs of test games (arena/logs/*.log) count as games too (`positions --logs`).
"""
import collections, glob, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POS = os.path.join(ROOT, "arena", "book_pos.json")
AN = os.path.join(ROOT, "arena", "book_an.jsonl")
BOOK = os.path.join(ROOT, "tools", "book.txt")


def games(subs, logs):
    out = []
    for l in open(os.path.join(ROOT, "arena", "games.jsonl")):
        g = json.loads(l)
        if g["submission"] in subs:
            out.append({"moves": g["moves"], "first": g["me_first"],
                        "res": "W" if g["won"] else "D" if g["draw"] else "L", "opp": g["opponent"]})
    if logs:
        for f in glob.glob(os.path.join(ROOT, "arena", "logs", "test-*.log")):
            t = open(f).read()
            me = re.search(r"^\s+(\d) \S+ \(me", t, re.M)
            opp = re.search(r"^\s+\d (\S+) \(agent_id (\d+)\)", t, re.M)
            sc = re.search(r"scores: 0=(-?\d+) 1=(-?\d+)", t)
            if not me or not sc or not opp or opp.group(1) == "Default":
                continue
            seat = int(me.group(1))
            a, b = int(sc.group(1)), int(sc.group(2))
            mine, theirs = (a, b) if seat == 0 else (b, a)
            out.append({"moves": re.findall(r"stdout: (\d \d)", t), "first": seat == 0,
                        "res": "W" if mine > theirs else "D" if mine == theirs else "L", "opp": opp.group(1)})
    return out


def positions(subs, logs):
    gs = games(subs, logs)
    won = set()                      # positions of ours in won games: left to the search
    for g in gs:
        if g["res"] == "W":
            for i in range(0 if g["first"] else 1, len(g["moves"]), 2):
                won.add(" ".join(g["moves"][:i]))
    pos = {}
    for g in gs:
        if g["res"] == "W":
            continue
        for i in range(0 if g["first"] else 1, len(g["moves"]), 2):
            pre = " ".join(g["moves"][:i])
            if pre in won:
                continue
            e = pos.setdefault(pre, {"prefix": pre, "played": collections.Counter(), "L": 0, "D": 0, "opps": []})
            e["played"][g["moves"][i]] += 1
            e[g["res"]] += 1
            if g["opp"] not in e["opps"]:
                e["opps"].append(g["opp"])
    rows = []
    for e in pos.values():
        e["played"] = dict(e["played"])
        rows.append(e)
    rows.sort(key=lambda e: (len(e["prefix"]), e["prefix"]))
    json.dump(rows, open(POS, "w"))
    print(f"{len(gs)} games, {sum(g['res'] != 'W' for g in gs)} not won: {len(rows)} positions -> {POS}")


def build(gain, dgain):
    pos = {e["prefix"]: e for e in json.load(open(POS))}
    out, n = [], collections.Counter()
    for l in open(AN):
        a = json.loads(l)
        e = pos.get(a["prefix"])
        if not e or a.get("proven") or not a["children"]:
            continue
        ch = {c["move"]: c for c in a["children"]}
        best = a["best"]
        played = max(e["played"], key=e["played"].get)
        if best == played or played not in ch:
            n["same"] += 1
            continue
        g = ch[best]["q"] - ch[played]["q"]
        need = gain if e["L"] >= e["D"] else dgain   # drawn lines: only clear improvements
        if g < need:
            n["small"] += 1
            continue
        n["book"] += 1
        r, c = best.split()
        out.append(f"{a['hash']} {r} {c}  # ply {a['ply']} played {played} gain {g:+.3f} L{e['L']} D{e['D']} {','.join(e['opps'][:3])}")
    open(BOOK, "w").write("\n".join(out) + "\n")
    print(f"{n['book']} entries -> {BOOK} (same move {n['same']}, gain too small {n['small']})")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "positions":
        positions([int(x) for x in sys.argv[2:] if x.isdigit()], "--logs" in sys.argv)
    elif c == "build":
        args = sys.argv[2:]
        gain = float(args[args.index("--gain") + 1]) if "--gain" in args else 0.02
        dgain = float(args[args.index("--draw-gain") + 1]) if "--draw-gain" in args else 0.05
        build(gain, dgain)
