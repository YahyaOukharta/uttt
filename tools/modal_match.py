"""Gate matches on Modal: one container per pairing, many games in parallel.

  modal run tools/modal_match.py --a out/new.cpp --b out/v0.3.cpp [--games 300] [--ms 100] [--iters 0]
      [--fa "-DNN_H=256"] [--fb ""] [--pa "CN=0.4"] [--pb ""] [--open 2] [--jobs 8] [--reps 1]
A/B are single-file sources (packed nets embedded, LOCAL blocks kept). --reps N runs N containers
with different seeds and sums the results.
"""
import json, math, os
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = modal.App("ut-match")
image = modal.Image.debian_slim().apt_install("g++").add_local_file(os.path.join(ROOT, "tools", "referee.cpp"), "/src/referee.cpp")


@app.function(image=image, cpu=8, memory=8192, timeout=3 * 3600)
def match(a_src: str, b_src: str, fa: str, fb: str, pa: str, pb: str, games: int, ms: int, iters: int, opn: int, jobs: int, seed: int):
    import subprocess
    open("/tmp/a.cpp", "w").write(a_src)
    open("/tmp/b.cpp", "w").write(b_src)
    cc = ["g++", "-O3", "-std=c++17", "-march=native", "-DLOCAL"]
    subprocess.check_call(cc + ["-pthread", "-o", "/tmp/referee", "/src/referee.cpp"])
    subprocess.check_call(cc + fa.split() + ["-o", "/tmp/a", "/tmp/a.cpp"])
    subprocess.check_call(cc + fb.split() + ["-o", "/tmp/b", "/tmp/b.cpp"])
    cmd = ["/tmp/referee", "/tmp/a", "/tmp/b", "-g", str(games), "-s", str(seed), "-o", str(opn), "-j", str(jobs)]
    cmd += ["-n", str(iters)] if iters else ["-n", "0", "-t", str(ms), "-x", "20"]
    if pa: cmd += ["-pa", pa]
    if pb: cmd += ["-pb", pb]
    out = subprocess.check_output(cmd, text=True)
    cpu = subprocess.run("grep -m1 'model name' /proc/cpuinfo", shell=True, capture_output=True, text=True).stdout.strip()
    return json.loads(out.strip().splitlines()[-1]) | {"cpu": cpu, "threads": len(os.sched_getaffinity(0))}


@app.local_entrypoint()
def main(a: str, b: str, games: int = 300, ms: int = 100, iters: int = 0, fa: str = "", fb: str = "", pa: str = "", pb: str = "",
         open_plies: int = 2, jobs: int = 8, reps: int = 1, seed: int = 4242):
    """--pa may hold several A configs separated by '|': each runs in its own container, one app."""
    sa, sb = open(a).read(), open(b).read()
    cfgs = pa.split("|")
    args, keys = [], []
    for c in cfgs:
        for r in range(reps):
            args.append((sa, sb, fa, fb, c, pb, games, ms, iters, open_plies, jobs, seed + 7919 * r))
            keys.append(c)
    tot = {c: {"wins_a": 0, "wins_b": 0, "draws": 0, "faults_a": 0, "faults_b": 0, "mx": [0, 0]} for c in cfgs}
    for c, r in zip(keys, match.starmap(args)):
        t = tot[c]
        for k in ("wins_a", "wins_b", "draws", "faults_a", "faults_b"):
            t[k] += r[k]
        t["mx"] = [max(t["mx"][0], r["max_ms_a"]), max(t["mx"][1], r["max_ms_b"])]
    for c in cfgs:
        t = tot[c]
        w, l, d = t["wins_a"], t["wins_b"], t["draws"]
        n = w + l + d
        p = (w + d / 2 + 0.5) / (n + 1)
        var = (w * (1 - p) ** 2 + l * p ** 2 + d * (0.5 - p) ** 2) / max(n, 1)
        se = math.sqrt(var / max(n, 1))
        elo = -400 * math.log10(1 / p - 1)
        err = (-400 * math.log10(1 / min(p + 1.96 * se, .999) - 1) + 400 * math.log10(1 / max(p - 1.96 * se, .001) - 1)) / 2
        mode = f"{iters} it" if iters else f"{ms} ms, max {t['mx'][0]:.0f}/{t['mx'][1]:.0f}"
        print(f"RESULT {os.path.basename(a)} [{fa} {c}] vs {os.path.basename(b)} [{fb} {pb}]: +{w} ={d} -{l} ({100 * p:.1f}%)  "
              f"elo {elo:+.0f} +/- {err:.0f}  faults {t['faults_a']}/{t['faults_b']}  [{mode}]", flush=True)
