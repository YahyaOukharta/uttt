"""Long searches of book positions on Modal (see tools/book.py).

  modal run tools/modal_book.py --src out/x.cpp [--iters 4000000] [--chunk 64]
Reads arena/book_pos.json, skips prefixes already in arena/book_an.jsonl, appends results there.
The source must embed its net and keep LOCAL blocks (pack.py --keep-local).
"""
import json, os
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = modal.App("ut-book")
image = modal.Image.debian_slim().apt_install("g++")


@app.function(image=image, cpu=8, memory=16384, timeout=3600)
def analyze(src: str, prefixes: list, iters: int):
    import re, subprocess
    from concurrent.futures import ThreadPoolExecutor
    open("/tmp/a.cpp", "w").write(src)
    subprocess.check_call(["g++", "-O3", "-std=c++17", "-march=native", "-DLOCAL", "-o", "/tmp/a", "/tmp/a.cpp"])

    def one(pre):
        env = {**os.environ, "UT_ANALYZE": pre if pre else " ", "UT_FIRST_ITERS": str(iters), "UT_ALL": "1",
               "UT_POOL": str(1 << 26), "UT_PARAMS": "PONDER=0"}
        out = subprocess.run(["/tmp/a"], env=env, capture_output=True, text=True).stdout
        h = re.search(r"ply (\d+) side (\d) iters \d+ root q ([\d.]+) proven (\d) best (\d) (\d) hash ([0-9a-f]+)", out)
        if not h:
            return {"prefix": pre, "error": out[-300:]}
        ch = [{"move": f"{m[0]} {m[1]}", "N": int(m[2]), "q": float(m[3]), "proven": int(m[4])}
              for m in re.findall(r"^\s+(\d) (\d)\s+N (\d+)\s+q ([\d.]+)\s+prior [\d.]+\s+proven (\d)", out, re.M)]
        return {"prefix": pre, "ply": int(h.group(1)), "q": float(h.group(3)), "proven": int(h.group(4)),
                "best": f"{h.group(5)} {h.group(6)}", "hash": h.group(7), "children": ch}

    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, prefixes))


@app.local_entrypoint()
def main(src: str, iters: int = 4000000, chunk: int = 64):
    pos = json.load(open(os.path.join(ROOT, "arena", "book_pos.json")))
    path = os.path.join(ROOT, "arena", "book_an.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            done.add(json.loads(l)["prefix"])
    todo = [e["prefix"] for e in pos if e["prefix"] not in done]
    print(f"{len(todo)} positions to analyze ({len(done)} done)")
    s = open(src).read()
    chunks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
    n = 0
    with open(path, "a") as f:
        for res in analyze.map([s] * len(chunks), chunks, [iters] * len(chunks)):
            for r in res:
                if "error" in r:
                    print("error", r["prefix"], r["error"][-120:])
                    continue
                f.write(json.dumps(r) + "\n")
                n += 1
            f.flush()
            print(f"{n}/{len(todo)}", flush=True)
