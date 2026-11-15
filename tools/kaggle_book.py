#!/usr/bin/env python3
"""Long searches of book positions on Kaggle CPU kernels (free; see tools/book.py, tools/modal_book.py).

  kaggle_book.py push TAG N --src out/c24_n8t.cpp [--iters 4000000] [--min-games 0] [--out book_an.jsonl]
  kaggle_book.py status TAG N
  kaggle_book.py fetch TAG N [--out book_an.jsonl]
push splits the positions of arena/book_pos.json not yet in arena/OUT over N kernels ut-TAG-1..N (the source,
with its net and LOCAL blocks, travels inside the kernel script); fetch appends the results to arena/OUT.
UT_KP=2 selects the second account.
"""
import base64, json, os, re, sys, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kaggle_gen as K

ROOT = K.ROOT
KERNEL = r'''import base64, json, os, re, subprocess, zlib
from concurrent.futures import ThreadPoolExecutor
SRC = %r
PRE = %r
ITERS = %d
open("/kaggle/working/a.cpp", "wb").write(zlib.decompress(base64.b64decode(SRC)))
subprocess.check_call(["g++", "-O3", "-std=c++17", "-march=native", "-DLOCAL", "-o", "/kaggle/working/a", "/kaggle/working/a.cpp"])
os.remove("/kaggle/working/a.cpp")
pool = 1 << 26 if ITERS <= 6000000 else 1 << 27

def one(pre):
    env = {**os.environ, "UT_ANALYZE": pre if pre else " ", "UT_FIRST_ITERS": str(ITERS), "UT_ALL": "1",
           "UT_POOL": str(pool), "UT_PARAMS": "PONDER=0"}
    out = subprocess.run(["/kaggle/working/a"], env=env, capture_output=True, text=True).stdout
    h = re.search(r"ply (\d+) side (\d) iters \d+ root q ([\d.]+) proven (\d) best (\d) (\d) hash ([0-9a-f]+)", out)
    if not h:
        return None
    ch = [{"move": f"{m[0]} {m[1]}", "N": int(m[2]), "q": float(m[3]), "proven": int(m[4])}
          for m in re.findall(r"^\s+(\d) (\d)\s+N (\d+)\s+q ([\d.]+)\s+prior [\d.]+\s+proven (\d)", out, re.M)]
    return {"prefix": pre, "ply": int(h.group(1)), "q": float(h.group(3)), "proven": int(h.group(4)),
            "best": f"{h.group(5)} {h.group(6)}", "hash": h.group(7), "children": ch}

with open("/kaggle/working/an.jsonl", "w") as f, ThreadPoolExecutor(os.cpu_count()) as ex:
    for r in ex.map(one, json.loads(PRE)):
        if r:
            f.write(json.dumps(r) + "\n")
            f.flush()
os.remove("/kaggle/working/a")
'''


def opt(args, name, default):
    return type(default)(args[args.index(name) + 1]) if name in args else default


def push(tag, n, args):
    out = os.path.join(ROOT, "arena", opt(args, "--out", "book_an.jsonl"))
    done = {json.loads(l)["prefix"] for l in open(out)} if os.path.exists(out) else set()
    mg = opt(args, "--min-games", 0)
    todo = [e["prefix"] for e in json.load(open(os.path.join(ROOT, "arena", "book_pos.json")))
            if e["prefix"] not in done and e["L"] + e["D"] >= mg]
    print(f"{len(todo)} positions over {n} kernels")
    if not todo:
        return
    src = base64.b64encode(zlib.compress(open(opt(args, "--src", "out/c24_n8t.cpp"), "rb").read(), 9)).decode()
    for i in range(1, n + 1):
        part = todo[i - 1::n]
        d = os.path.join(ROOT, "kaggle", f"b{K.SFX}_{tag}_{i}")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "an.py"), "w").write(KERNEL % (src, json.dumps(part), opt(args, "--iters", 4000000)))
        json.dump({"id": f"{K.USER}/ut-{tag}-{i}", "title": f"ut-{tag}-{i}", "code_file": "an.py", "language": "python",
                   "kernel_type": "script", "is_private": True, "enable_gpu": os.environ.get("UT_KGPU") == "1",
                   "enable_internet": False, "dataset_sources": []}, open(os.path.join(d, "kernel-metadata.json"), "w"))
        r = K.kg("kernels", "push", "-p", d)
        print(i, len(part), (r.stdout + r.stderr).strip().splitlines()[-1][:100], flush=True)


def fetch(tag, n, args):
    out = os.path.join(ROOT, "arena", opt(args, "--out", "book_an.jsonl"))
    st = K.status(tag, n)
    got = 0
    with open(out, "a") as f:
        for i in range(1, n + 1):
            if st[i - 1] != "COMPLETE":
                continue
            d = os.path.join(ROOT, "kaggle", "out", f"b{K.SFX}_{tag}_{i}")
            if os.path.isdir(d):
                continue
            os.makedirs(d)
            K.kg("kernels", "output", f"{K.USER}/ut-{tag}-{i}", "-p", d)
            p = os.path.join(d, "an.jsonl")
            if os.path.exists(p):
                for l in open(p):
                    f.write(l)
                    got += 1
    print(f"{got} results -> {out}")


if __name__ == "__main__":
    c, tag = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3])
    {"push": push, "fetch": fetch}.get(c, lambda t, k, a: K.status(t, k))(tag, n, sys.argv[4:])
