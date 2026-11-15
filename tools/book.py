#!/usr/bin/env python3
"""Opening book from arena games: many opponents (and our bot) play deterministically, so the
same lost games repeat. Search every position of our non-won games much longer than a turn
allows and answer the ones where the long search prefers another move.

  book.py positions SUB [SUB ...]          our positions in lost/drawn games -> arena/book_pos.json
  modal run tools/modal_book.py --src out/x.cpp [--iters 4000000]   long searches -> arena/book_an.jsonl
  book.py build [--gain 0.02] [--draw-gain 0.05] [--record-gain 1]  -> tools/book.txt (HASH ROW COL)

Positions that occur in a won game of ours, or in a drawn one as the second player, are changed by the long
search only for a gain of at least 0.03 (and left alone if they occur in no other game); but where our own move varied and one move has the better record (gain >= RECORD_GAIN games;
at least 3 games unless the other moves score <= 25%), the book replays that move unless the long search rates it more than 0.05 below its best.
Logs of test games (arena/logs/*.log) count as games too (`positions --logs`).
"""
import collections, glob, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POS = os.path.join(ROOT, "arena", "book_pos.json")
AN = os.path.join(ROOT, "arena", "book_an.jsonl")
AN_DEEP = os.path.join(ROOT, "arena", "book_an_deep.jsonl")
PRE = os.path.join(ROOT, "arena", "book_pre.jsonl")   # likely replies to book moves (tools/modal_pre.py)   # longer searches of frequent positions, preferred
BOOK = os.path.join(ROOT, "tools", "book.txt")
GAMES = os.path.join(ROOT, "arena", "book_games.json")   # the games behind book_pos.json (moves, seat, result)


def games(subs, logs):
    out, seen = [], set()
    for l in open(os.path.join(ROOT, "arena", "games.jsonl")):
        g = json.loads(l)
        if g["game_id"] in seen:   # two downloads may have appended the same game
            continue
        seen.add(g["game_id"])
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
    json.dump([{"moves": g["moves"], "first": g["first"], "res": g["res"]} for g in gs], open(GAMES, "w"))
    # Positions of ours in won games, and in drawn games as the second player (a good result against
    # the top), are left to the search. A position that also occurs in a non-won game is "mixed": the
    # book may change it, but only for a clearly better move (build: MIXED_GAIN).
    def kept(g):
        return g["res"] == "W" or (g["res"] == "D" and not g["first"])
    won = collections.Counter()
    for g in gs:
        if kept(g):
            for i in range(0 if g["first"] else 1, len(g["moves"]), 2):
                won[" ".join(g["moves"][:i])] += 1
    pos = {}
    for g in gs:
        if kept(g):
            continue
        for i in range(0 if g["first"] else 1, len(g["moves"]), 2):
            pre = " ".join(g["moves"][:i])
            e = pos.setdefault(pre, {"prefix": pre, "played": collections.Counter(), "L": 0, "D": 0, "K": won[pre], "opps": []})
            e["played"][g["moves"][i]] += 1
            e[g["res"]] += 1
            if g["opp"] not in e["opps"]:
                e["opps"].append(g["opp"])
    # Positions where our own move varied (the search is not deterministic) and one move has the better
    # record: the book can replay it against deterministic opponents, even in won / drawn lines.
    sc = {"W": 1.0, "D": 0.5, "L": 0.0}
    st = collections.defaultdict(lambda: collections.defaultdict(list))
    for g in gs:
        for i in range(0 if g["first"] else 1, len(g["moves"]), 2):
            st[" ".join(g["moves"][:i])][g["moves"][i]].append(sc[g["res"]])
    for pre, mv in st.items():
        if len(mv) < 2:
            continue
        avg = {m: sum(v) / len(v) for m, v in mv.items()}
        best = max(avg, key=lambda m: (avg[m], len(mv[m])))
        gain = sum(avg[best] - x for m, v in mv.items() if m != best for x in v)
        if gain >= 0.5:
            e = pos.setdefault(pre, {"prefix": pre, "played": collections.Counter(), "L": 0, "D": 0, "opps": []})
            e["emp"] = {m: [len(v), round(avg[m], 3)] for m, v in mv.items()}
    rows = []
    for e in pos.values():
        e["played"] = dict(e["played"])
        rows.append(e)
    rows.sort(key=lambda e: (len(e["prefix"]), e["prefix"]))
    json.dump(rows, open(POS, "w"))
    print(f"{len(gs)} games, {sum(g['res'] != 'W' for g in gs)} not won: {len(rows)} positions -> {POS}")


LOST_Q = 0.1   # below this long-search value a position counts as lost
MIXED_GAIN = 0.03   # positions that also occur in won games / second-player draws
PRE_GAIN = 0.03     # pre-emptive entries (book_pre.jsonl)


def losing_moves(an, lost_q):
    """{prefix: moves of ours refuted in practice}: every observed opponent reply after the move led to a
    position that is proven lost, or whose moves are all refuted or valued below lost_q. Computed from the
    deepest positions up, so a refutation found late in a line moves the deviation as far back as needed."""
    if not os.path.exists(GAMES):
        return {}
    follow = collections.defaultdict(lambda: collections.defaultdict(set))   # P -> our move -> next P
    for g in json.load(open(GAMES)):
        ms = g["moves"]
        for i in range(0 if g["first"] else 1, len(ms) - 2, 2):
            follow[" ".join(ms[:i])][ms[i]].add(" ".join(ms[:i + 2]))
    lost, bad = {}, {}
    for pre in sorted(set(follow) | set(an), key=lambda p: -len(p)):
        a = an.get(pre)
        if a and a.get("proven") == 1:        # proven win for the side that moved into it
            lost[pre] = True
            continue
        b = {m for m, nxt in follow.get(pre, {}).items() if all(lost.get(q, False) for q in nxt)}
        if b:
            bad[pre] = b
        if a and a["children"]:
            rest = [c["q"] for c in a["children"] if c["move"] not in b]
            lost[pre] = bool(b) and (not rest or max(rest) < lost_q)
        else:
            lost[pre] = False
    return bad


TRUNK_N, TRUNK_SCORE = 30, 0.3   # positions of ours with this many games whose usual move scores this well


def trunk():
    """{prefix: usual move} for frequent positions where the usual move does well: the book leaves them alone
    (an entry there reshuffles every game through it, e.g. v1.18's first reply as second player)."""
    if not os.path.exists(GAMES):
        return {}
    sc = {"W": 1.0, "D": 0.5, "L": 0.0}
    st = collections.defaultdict(lambda: collections.defaultdict(list))
    for g in json.load(open(GAMES)):
        for i in range(0 if g["first"] else 1, len(g["moves"]), 2):
            st[" ".join(g["moves"][:i])][g["moves"][i]].append(sc[g["res"]])
    out = {}
    for pre, mv in st.items():
        if sum(len(v) for v in mv.values()) >= TRUNK_N:
            top = max(mv, key=lambda m: len(mv[m]))
            if sum(mv[top]) / len(mv[top]) >= TRUNK_SCORE:
                out[pre] = top
    return out


def build(gain, dgain, emp_gain=1.0, emp_tol=0.05):
    pos = {e["prefix"]: e for e in json.load(open(POS))}
    out, n = [], collections.Counter()
    tr = trunk()
    an = {}
    for f in (AN, AN_DEEP):
        if os.path.exists(f):
            for l in open(f):
                a = json.loads(l)
                an[a["prefix"]] = a
    bad = losing_moves(an, LOST_Q)
    for a in an.values():
        if a["prefix"] in tr:
            n["trunk"] += 1
            continue
        e = pos.get(a["prefix"])
        if not a.get("proven") and a["children"] and a["prefix"] in bad:
            # A move of ours that the opponents' actual replies refuted: play the best other move.
            ch = {c["move"]: c for c in a["children"]}
            alt = max((c for c in a["children"] if c["move"] not in bad[a["prefix"]]), key=lambda c: c["q"], default=None)
            if alt and alt["q"] >= LOST_Q:
                n["refuted"] += 1
                r, c = alt["move"].split()
                out.append(f"{a['hash']} {r} {c}  # ply {a['ply']} refuted {','.join(sorted(bad[a['prefix']]))} alt q {alt['q']:.3f}")
                continue
        if not e or a.get("proven") or not a["children"]:
            continue
        ch = {c["move"]: c for c in a["children"]}
        best = a["best"]
        if "emp" in e:   # replay the move with the better record unless the long search finds it clearly worse
            emp = e["emp"]
            eb = max(emp, key=lambda m: (emp[m][1], emp[m][0]))
            eg = sum(n * (emp[eb][1] - q) for m, (n, q) in emp.items() if m != eb)
            no = sum(n for m, (n, q) in emp.items() if m != eb)
            others = sum(n * q for m, (n, q) in emp.items() if m != eb) / no
            sure = emp[eb][0] >= 3 or others <= 0.25   # few games only against moves that (nearly) always lose
            # and the difference must be clear given the numbers of games: a few lucky games of a rare move
            # must not replace a move with hundreds of games (v1.18: 4 games of 4 3 vs 1447 of 4 5 at ply 1)
            if no >= 20:
                sure = sure and emp[eb][1] - 1 / emp[eb][0] ** 0.5 > others + 1 / no ** 0.5
            if sure and eb not in bad.get(a["prefix"], ()) and eg >= emp_gain and eb in ch and best in ch and ch[eb]["q"] >= ch[best]["q"] - emp_tol:
                n["record"] += 1
                r, c = eb.split()
                rec = " ".join(f"{m.replace(' ', ',')}:{k}x{q}" for m, (k, q) in emp.items())
                out.append(f"{a['hash']} {r} {c}  # ply {a['ply']} record {rec} gain {eg:.1f}")
                continue
        if not e["played"]:
            continue
        played = max(e["played"], key=e["played"].get)
        if best == played or played not in ch:
            n["same"] += 1
            continue
        g = ch[best]["q"] - ch[played]["q"]
        need = gain if e["L"] >= e["D"] else dgain   # drawn lines: only clear improvements
        if e.get("K"):                               # also on a won line: change it only for a clear gain
            need = max(need, MIXED_GAIN)
        if g < need:
            n["small"] += 1
            continue
        n["book"] += 1
        r, c = best.split()
        out.append(f"{a['hash']} {r} {c}  # ply {a['ply']} played {played} gain {g:+.3f} L{e['L']} D{e['D']} {','.join(e['opps'][:3])}")
    # Pre-emptive entries: positions after the opponent's likely replies to our book moves, where a short
    # search (about one of our turns) picks a move clearly worse than the long search's best.
    have = {l.split()[0] for l in out}
    pre = []
    if os.path.exists(PRE):
        for l in open(PRE):
            a = json.loads(l)
            if a["hash"] in have or a.get("proven") or not a["children"] or not a.get("fast_best") or a["prefix"] in tr:
                continue
            ch = {c["move"]: c["q"] for c in a["children"]}
            fb, b = a["fast_best"], a["best"]
            if fb == b or fb not in ch or b not in ch or ch[b] - ch[fb] < PRE_GAIN:
                continue
            have.add(a["hash"])
            n["pre"] += 1
            r, c = b.split()
            pre.append((ch[b] - ch[fb], f"{a['hash']} {r} {c}  # ply {a['ply']} pre: short search {fb} {ch[fb]:.3f} vs {ch[b]:.3f}"))
    out += [l for g, l in sorted(pre, reverse=True)]   # largest gains first: pack.py drops from the end if too long
    # Probed lines (tools/probe.py): forced moves of second-player games that drew or won against a top
    # bot's deterministic play. First, so they take precedence and are never trimmed.
    pb = os.path.join(ROOT, "tools", "probe_book.txt")
    if os.path.exists(pb):
        probe = [l.strip() for l in open(pb) if l.strip() and not l.startswith("#")]
        n["probe"] = len(probe)
        out = probe + out
    open(BOOK, "w").write("\n".join(out) + "\n")
    print(f"{n['book'] + n['record'] + n['refuted'] + n['pre']} entries ({n['record']} by record, {n['refuted']} refuted, "
          f"{n['pre']} pre-emptive, {n['probe']} probed) -> {BOOK} "
          f"(same move {n['same']}, gain too small {n['small']}, trunk kept {n['trunk']})")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "positions":
        positions([int(x) for x in sys.argv[2:] if x.isdigit()], "--logs" in sys.argv)
    elif c == "build":
        args = sys.argv[2:]
        gain = float(args[args.index("--gain") + 1]) if "--gain" in args else 0.02
        dgain = float(args[args.index("--draw-gain") + 1]) if "--draw-gain" in args else 0.05
        eg = float(args[args.index("--record-gain") + 1]) if "--record-gain" in args else 1.0
        build(gain, dgain, eg)
