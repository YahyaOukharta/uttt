#!/usr/bin/env python3
"""Archive our CodinGame Ultimate Tic-Tac-Toe arena battles (the API only exposes games we played in).

  arena.py fetch [--logs [SUB_ID]]
                            list battles of every submission -> arena/battles.jsonl;
                            with --logs also download each game's moves (of one submission) -> arena/games.jsonl
  arena.py stats [SUB_ID]   win rate per opponent (and per colour when logs are present)
Respects the API rate limit (about 100 requests/minute).
"""
import json, os, re, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://www.codingame.com/api/v1"
PUZ = "tic-tac-toe"
BATTLES = os.path.join(ROOT, "arena", "battles.jsonl")
GAMES = os.path.join(ROOT, "arena", "games.jsonl")
TOKEN = open(os.path.expanduser("~/.config/codingame/token")).read().strip()


def get(path, raw=False):
    for _ in range(5):
        req = urllib.request.Request(API + path, headers={"Authorization": "Bearer " + TOKEN})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read().decode("utf-8")
            time.sleep(0.65)
            return body if raw else json.loads(body)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(int(e.headers.get("Retry-After", "30")) + 1)
                continue
            raise
    raise RuntimeError("rate limited: " + path)


def load(path, key):
    d = {}
    if os.path.exists(path):
        for l in open(path):
            if l.strip():
                r = json.loads(l)
                d[r[key]] = r
    return d


def save(path, d):
    with open(path, "w") as f:
        for k in sorted(d):
            f.write(json.dumps(d[k]) + "\n")


def fetch(logs, only=None):
    battles = load(BATTLES, "game_id")
    subs = get(f"/puzzles/{PUZ}/submissions")
    subs = subs.get("submissions", subs)
    for s in subs:
        sid, before = s["submission_id"], None
        while True:
            q = f"/puzzles/{PUZ}/battles?submission={sid}&limit=100" + (f"&before={before}" if before else "")
            page = get(q)
            for b in page.get("battles", []):
                if b.get("done"):
                    opp = [p for p in b["players"] if not p.get("me")][0]
                    pos = [p.get("position") for p in b["players"]]
                    draw = len(pos) == 2 and pos[0] is not None and pos[0] == pos[1]
                    battles[b["game_id"]] = {"game_id": b["game_id"], "submission": sid, "won": b["my_position"] == 0 and not draw,
                                             "draw": draw, "opponent": opp["nickname"], "ended_at": b.get("ended_at"), "raw": b}
            before = page.get("next_before")
            if not before:
                break
        print(f"submission {sid}: {sum(1 for b in battles.values() if b['submission'] == sid)} battles", flush=True)
    save(BATTLES, battles)
    if not logs:
        return
    games = load(GAMES, "game_id")
    todo = [g for g in sorted(battles, reverse=True) if g not in games and (not only or battles[g]["submission"] == only)]
    print(f"{len(todo)} game logs to download", flush=True)
    with open(GAMES, "a") as f:
        for i, gid in enumerate(todo):
            try:
                body = get(f"/games/{gid}", raw=True)
                log = json.loads(body).get("log", "") if body.lstrip().startswith("{") else body
            except Exception as e:
                print("skip", gid, e, flush=True)
                continue
            agents = re.findall(r"^\s+(\d) (.+?) \((?:me, )?agent_id", log, re.M)
            me_seat = next((int(a) for a, line in zip([x[0] for x in agents], re.findall(r"^\s+\d .*$", log, re.M)) if "(me" in line), None)
            moves = re.findall(r"stdout: (\d \d)", log)
            rec = {"game_id": gid, "submission": battles[gid]["submission"], "opponent": battles[gid]["opponent"],
                   "won": battles[gid]["won"], "draw": battles[gid].get("draw", False), "me_first": me_seat == 0, "moves": moves}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(todo)}", flush=True)


def stats(sub):
    battles = load(BATTLES, "game_id")
    games = load(GAMES, "game_id")
    if not sub:
        sub = max(b["submission"] for b in battles.values())
    rows, col = {}, {True: [0, 0, 0], False: [0, 0, 0]}
    for b in battles.values():
        if b["submission"] != int(sub):
            continue
        k = 2 if b.get("draw") else (0 if b["won"] else 1)
        r = rows.setdefault(b["opponent"], [0, 0, 0])
        r[k] += 1
        g = games.get(b["game_id"])
        if g:
            col[g["me_first"]][k] += 1
    w, l, d = (sum(r[i] for r in rows.values()) for i in range(3))
    print(f"submission {sub}: +{w} ={d} -{l} ({100 * (w + d / 2) / max(w + l + d, 1):.1f}%)")
    for c, name in ((True, "first"), (False, "second")):
        if sum(col[c]):
            print(f"  {name}: +{col[c][0]} ={col[c][2]} -{col[c][1]}")
    for o, r in sorted(rows.items(), key=lambda x: ((x[1][0] + x[1][2] / 2) / sum(x[1]), -x[1][1])):
        print(f"  {o:<22}+{r[0]} ={r[2]} -{r[1]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        i = sys.argv.index("--logs") if "--logs" in sys.argv else -1
        fetch(i >= 0, int(sys.argv[i + 1]) if 0 <= i < len(sys.argv) - 1 else None)
    else:
        stats(sys.argv[2] if len(sys.argv) > 2 else None)
