#!/usr/bin/env python3
"""CodinGame API helper for Ultimate Tic-Tac-Toe.

  cg.py test SRC [OPPONENT] [--replay GAME_ID]   play a test game, save the log to arena/logs/, print the result
  cg.py submit SRC                               submit SRC to the arena and mirror it into the IDE
  cg.py progress                                 rank / score / league of the current submission
  cg.py opponents                                players that can be used as test opponents
  cg.py top [N]                                  leaderboard
OPPONENT: default | boss | me | <agent_id>
"""
import json, os, sys, time, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://www.codingame.com/api/v1"
PUZ = "tic-tac-toe"
LOGS = os.path.join(ROOT, "arena", "logs")
TOKEN = open(os.path.expanduser("~/.config/codingame/token")).read().strip()


def call(path, body=None, raw=False):
    for _ in range(6):
        req = urllib.request.Request(API + path, headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"},
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                text = r.read().decode("utf-8")
            return text if raw else json.loads(text)
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")
            if e.code == 429:
                wait = int(e.headers.get("Retry-After", "30")) + 1
                print(f"rate limited, waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            if e.code == 409 and "test-game-running" in text:
                time.sleep(10)
                continue
            sys.exit(f"HTTP {e.code}: {text[:600]}")
    sys.exit("gave up: " + path)


def test(src, opp="default", replay=None, ide=False):
    body = {"language": "C++", "code": open(src, encoding="utf-8").read()}
    if ide:   # also make this the code of the CodinGame IDE (overwrites what is there)
        body["overwrite_ide_code"] = True
    if replay:
        body["replay_game_id"] = int(replay)
    else:
        body["opponents"] = [int(opp) if opp.isdigit() else opp]
    r = call(f"/puzzles/{PUZ}/test-games?wait=30", body)
    t0 = time.time()
    while r.get("status") == "running":
        if time.time() - t0 > 300:   # a stuck test game: give up after 5 minutes
            sys.exit(f"test game {r.get('test_game_id')} still running after 5 min")
        r = call(f"/test-games/{r['test_game_id']}?wait=30&format=json")
    if r.get("status") != "done":
        sys.exit(json.dumps(r)[:600])
    log = r["log"] if isinstance(r["log"], str) else call(f"/test-games/{r['test_game_id']}", raw=True)
    if not isinstance(log, str) or "frame" not in log:
        log = call(f"/test-games/{r['test_game_id']}", raw=True)
    os.makedirs(LOGS, exist_ok=True)
    path = os.path.join(LOGS, f"test-{r['test_game_id']}.log")
    open(path, "w").write(log)
    tail = log.strip().splitlines()
    agents = [l.strip() for l in tail if "agent_id" in l][:2]
    print(path, "|", " / ".join(agents), "|", " ".join(l.strip() for l in tail[-2:]))
    return path


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "test":
        rep = a[a.index("--replay") + 1] if "--replay" in a else None
        rest = [x for x in a[1:] if x != "--replay" and x != rep]
        test(rest[0], rest[1] if len(rest) > 1 else "default", rep)
    elif a[0] == "submit":
        r = call(f"/puzzles/{PUZ}/submissions", {"language": "C++", "code": open(a[1], encoding="utf-8").read()})
        print(json.dumps({k: v for k, v in r.items() if k != "limits"}), r.get("limits", {}).get("submits"))
        test(a[1], "default", ide=True)   # mirror the submitted code into the IDE
    elif a[0] == "progress":
        r = call(f"/puzzles/{PUZ}/progress")
        print(json.dumps({k: v for k, v in r.items() if k not in ("limits", "league_history")}))
    elif a[0] == "opponents":
        for o in call(f"/puzzles/{PUZ}/opponents")["opponents"]:
            print(o)
    elif a[0] == "top":
        for p in call(f"/puzzles/{PUZ}/leaderboard?limit={a[1] if len(a) > 1 else 20}")["rows"]:
            print(f"{p['rank']:>4} {p['nickname']:<22}{p['league']:<8}{p['score']:>7} {p['language']}")


if __name__ == "__main__":
    main()
