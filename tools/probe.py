#!/usr/bin/env python3
"""Probe the top bots' deterministic play for second-player lines that do not lose.

Arena bots repeat whole games: with our moves fixed, a game against a given opponent replays move for
move. For each lost second-player game against a top opponent, this tries other moves of ours near the
decisive moment (alternatives the long search rates close to its best, arena/book_an*.jsonl) in
CodinGame test games that replay the battle against the opponent's current code. The probe source is the
current bot with our moves forced up to the deviation (book entries placed first). A probe that draws or
wins is kept: every move of ours in that test game goes to tools/probe_book.txt, which book.py build
puts first, so the arena replays the line.

  probe.py plan SUB [SUB ...] [--top 6] [--window 8] [--delta 0.06]   -> arena/probe_plan.json
  probe.py run [--max 80] [--net tools/nets/n7t640.bin]               play the plan (40 s per test game)
  probe.py stats
Results: arena/probe_results.jsonl.
"""
import json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cg, cg_suite

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = os.path.join(ROOT, "arena", "probe_plan.json")
RES = os.path.join(ROOT, "arena", "probe_results.jsonl")
PBOOK = os.path.join(ROOT, "tools", "probe_book.txt")
HASHBIN = "/tmp/ut_hashbin"


def opt(args, name, default):
    return type(default)(args[args.index(name) + 1]) if name in args else default


def load_an():
    an = {}
    for f in ("book_an.jsonl", "book_an_deep.jsonl"):
        p = os.path.join(ROOT, "arena", f)
        if os.path.exists(p):
            for l in open(p):
                a = json.loads(l)
                an[a["prefix"]] = a
    return an


def q_of(c):
    return {1: 1.0, 3: 0.0, 2: 0.5}.get(c["proven"], c["q"])


_hcache = {}


def phash(prefix):
    """Book hash of the position after the moves of `prefix` (engine analysis output, 10 iterations)."""
    if prefix not in _hcache:
        if not os.path.exists(HASHBIN):
            subprocess.check_call(["g++", "-std=c++17", "-O2", "-DLOCAL", "-o", HASHBIN, os.path.join(ROOT, "out", "c24_n8t.cpp")])
        env = {**os.environ, "UT_ANALYZE": prefix if prefix else " ", "UT_FIRST_ITERS": "10", "UT_POOL": "65536",
               "UT_PARAMS": "PONDER=0"}
        out = subprocess.run([HASHBIN], env=env, capture_output=True, text=True).stdout
        _hcache[prefix] = out.split("hash ")[1].split()[0]
    return _hcache[prefix]


def plan(subs, ntop, window, delta):
    top = [l.split()[1] for l in os.popen(f"python3 {ROOT}/tools/cg.py top {ntop + 1}").read().splitlines()
           if l.split() and l.split()[0].isdigit() and l.split()[1] != "yhyoxx"][:ntop]
    an = load_an()
    done = {(r["line"], r["dev"]) for r in map(json.loads, open(RES))} if os.path.exists(RES) else set()
    lines = {}
    for l in open(os.path.join(ROOT, "arena", "games.jsonl")):
        g = json.loads(l)
        if g["submission"] in subs and g["opponent"] in top and not g["me_first"] and not g["won"] and not g["draw"]:
            lines.setdefault((g["opponent"], tuple(g["moves"])), g["game_id"])
    per_line = []
    seen = set()
    reach = {}   # position (our turn) -> opponents that reached it in these submissions' games
    for l in open(os.path.join(ROOT, "arena", "games.jsonl")):
        g = json.loads(l)
        if g["submission"] in subs:
            for i in range(0 if g["me_first"] else 1, len(g["moves"]), 2):
                # positions where some game scored (W/D) are protected: any opponent's lost games only are fair game
                reach.setdefault(" ".join(g["moves"][:i]), set()).add(g["opponent"] if not (g["won"] or g["draw"]) else "*scored*")
    json.dump({k: sorted(v) for k, v in reach.items()}, open(os.path.join(ROOT, "arena", "probe_reach.json"), "w"))
    book = {l.split()[0]: f"{l.split()[1]} {l.split()[2]}" for l in open(os.path.join(ROOT, "tools", "book.txt")) if l.strip() and not l.startswith("#")}
    for (opp, mt), gid in lines.items():
        ms, mv = list(mt), " ".join(mt)
        # decisive moment: the largest drop between two of our turns (our error or the opponent's surprise)
        vals = []
        for i in range(1, len(ms), 2):
            a = an.get(" ".join(ms[:i]))
            if a and a["children"]:
                vals.append((i, a))
        worst, dply = 0.0, None
        for (i, a), (j, b) in zip(vals, vals[1:]):
            ch = {c["move"]: q_of(c) for c in a["children"]}
            if ms[i] in ch:
                bv = {1: 0.0, 3: 1.0, 2: 0.5}.get(b.get("proven"), b["q"])
                drop = max(ch.values()) - bv
                if drop > worst:
                    worst, dply = drop, i
        if dply is None:
            continue
        cands = []
        for i, a in vals:
            if i < dply - window or i > dply:
                continue
            ch = sorted(a["children"], key=lambda c: -q_of(c))
            best = q_of(ch[0])
            for c in ch[:4]:
                if c["move"] != ms[i] and q_of(c) >= best - delta and (mv, f"{i}:{c['move']}") not in done:
                    cands.append({"opp": opp, "game_id": gid, "line": mv, "moves": ms, "ply": i, "dev": f"{i}:{c['move']}",
                                  "move": c["move"], "q": round(q_of(c), 3), "best_q": round(best, 3)})
        cands = [c for c in cands if "*scored*" not in reach.get(" ".join(ms[:c["ply"]]), set())]   # only lost games get here
        cands = [c for c in cands if book.get(an[" ".join(ms[:c["ply"]])]["hash"]) != c["move"]]   # the book plays it already
        cands = [c for c in cands if (opp, " ".join(ms[:c["ply"]]), c["move"]) not in seen and not seen.add((opp, " ".join(ms[:c["ply"]]), c["move"]))]
        cands.sort(key=lambda c: (-c["ply"], -c["q"]))   # late deviations first: the known line stays longest
        per_line.append(cands)
    out = []   # round-robin over lines so every opponent is probed early
    while any(per_line):
        for c in per_line:
            if c:
                out.append(c.pop(0))
    json.dump(out, open(PLAN, "w"), indent=0)
    print(f"{len(lines)} lost second-player lines vs {', '.join(top)}: {len(out)} probes -> {PLAN}")


def forced_entries(moves, upto, first):
    """Book lines forcing our moves of `moves` at our plies < upto (and the deviation at upto)."""
    out = []
    for i in range(0 if first else 1, upto + 1, 2):
        r, c = moves[i].split()
        out.append(f"{phash(' '.join(moves[:i]))} {r} {c}")
    return out


def run(maxn, net):
    pl = json.load(open(PLAN))
    done = {(r["line"], r["dev"]) for r in map(json.loads, open(RES))} if os.path.exists(RES) else set()
    pl = [p for p in pl if (p["line"], p["dev"]) not in done][:maxn]
    print(f"{len(pl)} probes", flush=True)
    src = os.path.join(ROOT, "out", "probe.cpp")
    bk = "/tmp/ut_probe_book.txt"
    dead = set()   # opponents whose battles can no longer be replayed (they resubmitted)
    for p in pl:
        if p["opp"] in dead:
            continue
        t = time.time()
        ms = p["moves"][:p["ply"]] + [p["move"]]
        ent = forced_entries(ms, p["ply"], False)
        base = open(os.path.join(ROOT, "tools", "book.txt")).read()
        extra = open(PBOOK).read() if os.path.exists(PBOOK) else ""
        open(bk, "w").write("\n".join(ent) + "\n" + extra + base)
        subprocess.run(["python3", os.path.join(ROOT, "tools", "pack.py"), "-n", net, "-H", "640", "--huff",
                        "--book", bk, "-o", src, os.path.join(ROOT, "bot.cpp")], capture_output=True, check=True)
        try:
            path = cg.test(src, replay=p["game_id"])
        except (Exception, SystemExit) as ex:
            print(f"{p['opp']:<12} {p['dev']:<8} skipped: {str(ex)[:100]}", flush=True)
            if "404" in str(ex):
                dead.add(p["opp"])
            time.sleep(40)
            continue
        seat, r = cg_suite.result(path)
        log = open(path).read()
        import re
        played = re.findall(r"stdout: (\d \d)", log)
        follow = played[:p["ply"] + 1] == ms
        rec = {**p, "seat": seat, "result": r, "followed": follow, "log": os.path.basename(path), "t": int(time.time())}
        with open(RES, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{p['opp']:<12} ply {p['ply']:>2} {p['move']} (q {p['q']:.2f}/{p['best_q']:.2f}): {r}"
              + ("" if follow else "  (line not followed)"), flush=True)
        if r in ("W", "D") and seat == 1:   # keep: force every move of ours in this game (even off the planned line)
            keep_game(played, f"{p['opp']} {r} {p['dev']} {os.path.basename(path)}")
        time.sleep(max(0.0, 40.0 - (time.time() - t)))


def keep_game(played, note):
    """Force our (second-player) moves of a drawn / won test game, except at positions where some arena
    game of ours scored (those stay as they are for every opponent)."""
    reach = json.load(open(os.path.join(ROOT, "arena", "probe_reach.json")))
    last = len(played) - 1 if (len(played) - 1) % 2 == 1 else len(played) - 2
    keep = [e for i, e in zip(range(1, last + 1, 2), forced_entries(played, last, False))
            if "*scored*" not in reach.get(" ".join(played[:i]), [])]
    with open(PBOOK, "a") as f:
        f.write(f"# {note}\n" + "\n".join(keep) + "\n")
    print(f"  KEPT {len(keep)} entries -> {PBOOK}", flush=True)


def stats():
    rows = [json.loads(l) for l in open(RES)] if os.path.exists(RES) else []
    import collections
    st = collections.defaultdict(collections.Counter)
    for r in rows:
        st[r["opp"]][r["result"] or "?"] += 1
    for o, c in st.items():
        print(f"{o:<14} {dict(c)}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "plan":
        plan([int(x) for x in a[1:] if x.isdigit() and len(x) > 6], opt(a, "--top", 6), opt(a, "--window", 8), opt(a, "--delta", 0.06))
    elif a[0] == "keep":   # keep LOG NOTE: force the moves of a drawn / won second-player test game
        import re
        keep_game(re.findall(r"stdout: (\d \d)", open(os.path.join(ROOT, "arena", "logs", a[1])).read()), " ".join(a[2:]) + " " + a[1])
    elif a[0] == "run":
        run(opt(a, "--max", 80), opt(a, "--net", os.path.join(ROOT, "tools", "nets", "n7t640.bin")))
    else:
        stats()
