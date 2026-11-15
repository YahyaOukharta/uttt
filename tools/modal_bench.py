"""Speed benchmark on Modal x86: time a fixed number of first-turn iterations for one source
under several parameter sets, alternating runs.

  modal run tools/modal_bench.py --src out/x_local.cpp --params "CT=0|CT=16" [--iters 300000] [--reps 5]
      [--opening "4 4 4 0 0 4 4 1"]
"""
import os
import modal

app = modal.App("ut-bench")
image = modal.Image.debian_slim().apt_install("g++")


@app.function(image=image, cpu=2, memory=4096, timeout=1800)
def bench(src: str, params: str, iters: int, reps: int, opening: str):
    import subprocess, time
    open("/tmp/a.cpp", "w").write(src)
    subprocess.check_call(["g++", "-O3", "-std=c++17", "-march=native", "-DLOCAL", "-o", "/tmp/a", "/tmp/a.cpp"])
    inp = "-1 -1\n81\n" + "".join(f"{r} {c}\n" for r in range(9) for c in range(9))
    res = {p: [] for p in params.split("|")}
    for _ in range(reps):
        for p in res:
            env = {**os.environ, "UT_PARAMS": p, "UT_FIRST_ITERS": str(iters), "UT_ITERS": "1"}
            if opening:
                env["UT_OPENING"] = opening
            t = time.time()
            subprocess.run(["/tmp/a"], input=inp if not opening else None, env=env, capture_output=True, text=True)
            res[p].append(round(time.time() - t, 3))
    return res


@app.local_entrypoint()
def main(src: str, params: str = "", iters: int = 300000, reps: int = 5, opening: str = ""):
    r = bench.remote(open(src).read(), params, iters, reps, opening)
    for p, ts in r.items():
        ts = sorted(ts)
        print(f"{p:<20} median {ts[len(ts) // 2]:.3f}s  all {ts}")
