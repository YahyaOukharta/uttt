"""Pre-emptive book on Modal: answer the opponent's likely replies to our book moves before the arena does.

  modal run tools/modal_pre.py --src out/c24_n8t.cpp [--replies 2] [--iters 4000000] [--fast 35000] [--chunk 16]

For every book entry (tools/book.txt: position P, our move m) whose prefix is known (arena/book_an*.jsonl),
the position after m is searched (1M iterations, opponent to move) for its most visited replies r; each
position P+m+r is then searched long (ITERS) and short (FAST, about one turn of ours on CodinGame). Results go
to arena/book_pre.jsonl; book.py build adds an entry where the short search's move is worse than the long
search's best by >= 0.03. Each run also expands the long search's move in the positions found so far
(replies with at least 20% of the top reply's visits), so repeated runs go one turn deeper.
"""
import json, os
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = modal.App("ut-pre")
image = modal.Image.debian_slim().apt_install("g++")
RX_ROOT = r"ply (\d+) side (\d) iters \d+ root q ([\d.]+) proven (\d) best (\d) (\d) hash ([0-9a-f]+)"
RX_CH = r"^\s+(\d) (\d)\s+N (\d+)\s+q ([\d.]+)\s+prior [\d.]+\s+proven (\d)"


@app.function(image=image, cpu=8, memory=24576, timeout=3600)
def run(src: str, jobs: list, replies: int, iters: int, fast: int):
    import re, subprocess
    from concurrent.futures import ThreadPoolExecutor
    open("/tmp/a.cpp", "w").write(src)
    subprocess.check_call(["g++", "-O3", "-std=c++17", "-march=native", "-DLOCAL", "-o", "/tmp/a", "/tmp/a.cpp"])

    def an(pre, n):
        env = {**os.environ, "UT_ANALYZE": pre, "UT_FIRST_ITERS": str(n), "UT_ALL": "1",
               "UT_POOL": str(1 << 26), "UT_PARAMS": "PONDER=0"}
        out = subprocess.run(["/tmp/a"], env=env, capture_output=True, text=True).stdout
        h = re.search(RX_ROOT, out)
        if not h:
            return None
        ch = [{"move": f"{m[0]} {m[1]}", "N": int(m[2]), "q": float(m[3]), "proven": int(m[4])}
              for m in re.findall(RX_CH, out, re.M)]
        return {"prefix": pre, "ply": int(h.group(1)), "q": float(h.group(3)), "proven": int(h.group(4)),
                "best": f"{h.group(5)} {h.group(6)}", "hash": h.group(7), "children": ch}

    def one(job):
        pre_m = job
        opp = an(pre_m, 1000000)
        if not opp or opp["proven"]:
            return []
        out = []
        for c in sorted(opp["children"], key=lambda c: -c["N"])[:replies]:
            if c["proven"]:
                continue
            p = pre_m + " " + c["move"]
            deep = an(p, iters)
            if not deep or not deep["children"]:
                continue
            short = an(p, fast)
            deep["fast_best"] = short["best"] if short else None
            deep["reply_share"] = c["N"] / max(1, opp["children"][0]["N"] if opp["children"] else 1)
            out.append(deep)
        return out

    with ThreadPoolExecutor(8) as ex:
        return [r for rs in ex.map(one, jobs) for r in rs]


@app.local_entrypoint()
def main(src: str, replies: int = 2, iters: int = 4000000, fast: int = 35000, chunk: int = 48):
    an = {}
    for f in ("book_an.jsonl", "book_an_deep.jsonl", "book_pre.jsonl"):
        p = os.path.join(ROOT, "arena", f)
        if os.path.exists(p):
            for l in open(p):
                a = json.loads(l)
                an.setdefault(a["hash"], a["prefix"])
    path = os.path.join(ROOT, "arena", "book_pre.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            p = json.loads(l)["prefix"]
            done.add(p.rsplit(" ", 2)[0])   # P+m already expanded
    jobs = []
    for l in open(os.path.join(ROOT, "tools", "book.txt")):
        f = l.split()
        pre = an.get(f[0])
        if pre is None:
            continue
        pm = (pre + " " if pre else "") + f"{f[1]} {f[2]}"
        if pm not in done:
            jobs.append(pm)
    if os.path.exists(path):   # one turn deeper: the move we will play in each covered position
        for l in open(path):
            a = json.loads(l)
            if a.get("proven") or not a["children"] or a.get("reply_share", 1) < 0.2:
                continue
            pm = a["prefix"] + " " + a["best"]
            if pm not in done:
                jobs.append(pm)
    jobs = list(dict.fromkeys(jobs))
    print(f"{len(jobs)} moves to expand ({len(done)} done)", flush=True)
    if not jobs:
        return
    s = open(src).read()
    chunks = [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
    n = 0
    with open(path, "a") as f:
        for res in run.map([s] * len(chunks), chunks, [replies] * len(chunks), [iters] * len(chunks), [fast] * len(chunks)):
            for r in res:
                f.write(json.dumps(r) + "\n")
                n += 1
            f.flush()
    print(f"{n} positions -> {path}")
