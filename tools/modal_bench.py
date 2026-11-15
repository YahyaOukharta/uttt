"""Speed benchmark on Modal x86: time a fixed number of first-turn iterations for one source
under several parameter sets, alternating runs.

  modal run tools/modal_bench.py --src out/a.cpp,out/b.cpp --params "CT=0|CT=16" [--iters 300000] [--reps 5]
      [--opening "4 4 4 0 0 4 4 1"]
"""
import os
import modal

app = modal.App("ut-bench")
image = modal.Image.debian_slim().apt_install("g++")


@app.function(image=image, cpu=2, memory=4096, timeout=1800)
def bench(srcs: list, params: str, iters: int, reps: int, opening: str):
    import subprocess, time
    bins = []
    for k, src in enumerate(srcs):
        open(f"/tmp/a{k}.cpp", "w").write(src)
        subprocess.check_call(["g++", "-O3", "-std=c++17", "-march=native", "-DLOCAL", "-o", f"/tmp/a{k}", f"/tmp/a{k}.cpp"])
        bins.append(f"/tmp/a{k}")
    inp = "-1 -1\n81\n" + "".join(f"{r} {c}\n" for r in range(9) for c in range(9))
    res = {(k, p): [] for k in range(len(bins)) for p in params.split("|")}
    out = {}
    for _ in range(reps):
        for (k, p) in res:
            env = {**os.environ, "UT_PARAMS": p, "UT_FIRST_ITERS": str(iters), "UT_ITERS": "1"}
            if opening:
                env["UT_OPENING"] = opening
            t = time.time()
            r = subprocess.run([bins[k]], input=inp if not opening else None, env=env, capture_output=True, text=True)
            res[(k, p)].append(round(time.time() - t, 3))
            out[(k, p)] = (r.stdout.strip().split("\n")[0], r.stderr.strip().split("\n")[-1][:90])
    return [(k, p, v, out[(k, p)]) for (k, p), v in res.items()]


@app.local_entrypoint()
def main(src: str, params: str = "", iters: int = 300000, reps: int = 5, opening: str = ""):
    files = src.split(",")
    r = bench.remote([open(f).read() for f in files], params, iters, reps, opening)
    for k, p, ts, (mv, err) in r:
        ts = sorted(ts)
        print(f"{os.path.basename(files[k]):<16} {p:<14} median {ts[len(ts) // 2]:.3f}s  all {ts}  move {mv}  [{err}]")
