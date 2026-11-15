#!/usr/bin/env python3
"""Where our non-won games against the top players are decided.

  losses.py SUB [SUB ...] [--top 25] [--seat first|second] [--games]

For each of our turns with a long-search result (arena/book_an*.jsonl) the value v_i of the position, the
value of our move and of the best move are known. Between two of our turns the value changes by
  (best_i - v_i)          search noise (ignored)
  -(best_i - played_i)    our error: the move we played is worse than the long search's best
  v_{i+2} - played_i      surprise: the opponent's reply was better than the long search expected
The report sums both kinds per game, finds the decisive moment (largest single drop) and groups by ply.
"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import book as B

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_an():
    an = {}
    for f in (B.AN, B.AN_DEEP):
        if os.path.exists(f):
            for l in open(f):
                a = json.loads(l)
                an[a["prefix"]] = a
    return an


def value(a):
    """Value of a position for the side to move: proven results exactly, else the root q."""
    if a.get("proven") == 1:
        return 0.0
    if a.get("proven") == 3:
        return 1.0
    if a.get("proven") == 2:
        return 0.5
    return a["q"]


def child_q(a, m):
    for c in a["children"]:
        if c["move"] == m:
            return {1: 1.0, 3: 0.0, 2: 0.5}.get(c["proven"], c["q"])
    return None


def main():
    args = sys.argv[1:]
    subs = [int(x) for x in args if x.isdigit()]
    ntop = int(args[args.index("--top") + 1]) if "--top" in args else 25
    seat = args[args.index("--seat") + 1] if "--seat" in args else None
    top = [l.split()[1] for l in os.popen(f"python3 {ROOT}/tools/cg.py top {ntop}").read().splitlines() if l.split() and l.split()[0].isdigit()]
    an = load_an()
    gs = [json.loads(l) for l in open(os.path.join(ROOT, "arena", "games.jsonl"))]
    seen, games = set(), []
    for g in gs:
        if g["game_id"] in seen or g["submission"] not in subs or g["opponent"] not in top or g["won"]:
            continue
        seen.add(g["game_id"])
        if seat and g["me_first"] != (seat == "first"):
            continue
        games.append(g)
    tot = collections.Counter()
    byply = collections.defaultdict(lambda: [0.0, 0.0])
    decisive = collections.Counter()
    rows = []
    for g in games:
        ms = g["moves"]
        err = sur = 0.0
        worst = (0.0, None, None)
        start = None
        idx = list(range(0 if g["me_first"] else 1, len(ms), 2))
        for i in idx:
            a = an.get(" ".join(ms[:i]))
            if not a or not a["children"]:
                continue
            v = value(a)
            if start is None:
                start = (i, v)
            if a.get("proven"):
                break
            pq = child_q(a, ms[i])
            bq = max(child_q(a, c["move"]) for c in a["children"])
            nxt = an.get(" ".join(ms[:i + 2]))
            if pq is None:
                continue
            e = bq - pq
            err += e
            byply[i // 10 * 10][0] += e
            if e > worst[0]:
                worst = (e, i, f"our {ms[i]} (best {a['best']}) {pq:.2f} vs {bq:.2f}")
            if nxt and nxt["children"]:
                s = pq - value(nxt)
                if s > 0:
                    sur += s
                    byply[i // 10 * 10][1] += s
                    if s > worst[0]:
                        worst = (s, i, f"reply {ms[i + 1]} to our {ms[i]}: {pq:.2f} -> {value(nxt):.2f}")
        res = "D" if g["draw"] else "L"
        tot["games"] += 1
        tot["err"] += err
        tot["sur"] += sur
        kind = "none" if worst[1] is None else ("error" if worst[2].startswith("our") else "surprise")
        decisive[kind] += 1
        rows.append((g["opponent"], "first" if g["me_first"] else "second", res, start, err, sur, worst))
    print(f"{tot['games']} non-won games vs the top {ntop}: our errors sum {tot['err']:.1f}, surprises sum {tot['sur']:.1f}")
    print("decisive moment:", dict(decisive))
    print("by ply (errors / surprises):", {k: (round(v[0], 2), round(v[1], 2)) for k, v in sorted(byply.items())})
    if "--games" in args:
        for r in sorted(rows, key=lambda r: (r[1], r[0])):
            st = f"start ply {r[3][0]} v {r[3][1]:.2f}" if r[3] else "no analysis"
            print(f"  {r[1]:<6} {r[0]:<13} {r[2]}  {st}  err {r[4]:.2f} sur {r[5]:.2f}  worst ply {r[6][1]}: {r[6][2]}")


if __name__ == "__main__":
    main()
