#!/usr/bin/env python3
"""Evaluate a bot against the current top players with CodinGame test games.

Each test game replays a past arena battle: our code takes our seat of that battle and the
opponent plays its current arena code. The suite holds, for each top-N opponent, one battle where
we moved first and one where we moved second.

  cg_suite.py build [N]                 pick the battles (top N opponents) -> arena/suite.json
  cg_suite.py lost N SUB                one non-won battle of submission SUB per (top-N opponent, seat)
  cg_suite.py run SRC LABEL [ROUNDS]    play every battle of the suite ROUNDS times (UT_SKIP=K: start at battle K)
  cg_suite.py stats [LABEL...]          W/D/L by seat and by opponent
Results go to arena/suite_results.jsonl. Test games are limited (10 per minute, 80 per hour,
600 per day): one round of a top-10 suite is 20 games.
"""
import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE = os.path.join(ROOT, "arena", "suite.json")
RES = os.path.join(ROOT, "arena", "suite_results.jsonl")


def build(n):
    opps = cg.call(f"/puzzles/{cg.PUZ}/opponents")
    opps = opps.get("opponents", opps)
    top = [o["nickname"] for o in opps if not o.get("me")][:n]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "arena", "battles.jsonl"))]
    rows.sort(key=lambda r: r.get("ended_at") or 0, reverse=True)
    suite = []
    for o in top:
        for seat in (0, 1):
            for r in rows:
                pl = r["raw"]["players"]
                me = [i for i, p in enumerate(pl) if p.get("me")][0]
                if r["opponent"] == o and me == seat:
                    suite.append({"opponent": o, "seat": seat, "game_id": r["game_id"]})
                    break
    json.dump(suite, open(SUITE, "w"), indent=1)
    print(f"{len(suite)} battles for {len(top)} opponents -> {SUITE}")


def build_lost(n, sub):
    """One non-won game per (top-n opponent, seat) of submission SUB, seats read from the logs."""
    opps = cg.call(f"/puzzles/{cg.PUZ}/opponents")
    opps = opps.get("opponents", opps)
    top = [o["nickname"] for o in opps if not o.get("me")][:n]
    gs = [json.loads(l) for l in open(os.path.join(ROOT, "arena", "games.jsonl"))]
    gs = [g for g in gs if g["submission"] == sub and not g["won"]]
    suite = []
    for o in top:
        for first in (True, False):
            x = [g for g in gs if g["opponent"] == o and g["me_first"] == first]
            if x:
                g = max(x, key=lambda g: g["game_id"])
                suite.append({"opponent": o, "seat": 0 if first else 1, "game_id": g["game_id"],
                              "was": "D" if g["draw"] else "L"})
    json.dump(suite, open(SUITE, "w"), indent=1)
    print(f"{len(suite)} non-won battles vs the top {n} -> {SUITE}")


def result(path):
    """(our seat, W/D/L) from a test-game log; agent 0 moves first."""
    log = open(path).read()
    me = re.search(r"^\s+(\d) \S+ \(me", log, re.M)
    sc = re.search(r"scores: 0=(-?\d+) 1=(-?\d+)", log)
    if not me or not sc:
        return None, None
    a, b = int(sc.group(1)), int(sc.group(2))
    mine, theirs = (a, b) if me.group(1) == "0" else (b, a)
    return int(me.group(1)), "W" if mine > theirs else ("D" if mine == theirs else "L")


def run(src, label, rounds):
    suite = json.load(open(SUITE))
    for k in range(rounds):
        for e in suite[int(os.environ.get("UT_SKIP", "0")) if k == 0 else 0:]:
            t = time.time()
            try:
                path = cg.test(src, replay=e["game_id"])
            except (Exception, SystemExit) as ex:   # e.g. the opponent resubmitted: its old agent is no longer testable
                print(f"round {k + 1}: {e['opponent']:<14} skipped: {str(ex)[:120]}", flush=True)
                continue
            seat, r = result(path)
            with open(RES, "a") as f:
                f.write(json.dumps({"label": label, "opponent": e["opponent"], "seat": seat, "result": r,
                                    "log": os.path.basename(path), "game_id": e["game_id"], "t": int(time.time())}) + "\n")
            print(f"round {k + 1}: {e['opponent']:<14} seat {seat}: {r}", flush=True)
            time.sleep(max(0.0, 40.0 - (time.time() - t)))   # test games: 30 per 15 min, 80 per hour


def stats(labels):
    rows = [json.loads(l) for l in open(RES)] if os.path.exists(RES) else []
    labels = labels or sorted(set(r["label"] for r in rows))
    for lb in labels:
        rs = [r for r in rows if r["label"] == lb and r["result"]]
        for seat in (0, 1):
            x = [r["result"] for r in rs if r["seat"] == seat]
            if x:
                w, d, l = x.count("W"), x.count("D"), x.count("L")
                print(f"{lb:<10} {'first ' if seat == 0 else 'second'}  W {w:>3}  D {d:>3}  L {l:>3}  "
                      f"score {(w + d / 2) / len(x):.0%}  ({len(x)} games)")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif c == "lost":
        build_lost(int(sys.argv[2]), int(sys.argv[3]))
    elif c == "run":
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    elif c == "stats":
        stats(sys.argv[2:])
