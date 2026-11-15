#!/usr/bin/env python3
"""Play gate matches on Kaggle CPU kernels (x86, idle machines: fair real-time games).

  kaggle_match.py push NAME [--games 300] [--iters 20000] [--ms 0] [--open 2] A B [A2 B2 ...]
  kaggle_match.py status NAME
  kaggle_match.py fetch NAME
A and B are bot specs: a source file, optionally followed by '@' and compiler flags and by '#'
and runtime parameters (UT_PARAMS), e.g.
  bot.cpp@-DROLL=2  league/src/legacy.cpp  out/x.cpp#CP=0.8,PONDER=0
--ms > 0 plays real time (2 games at once on the 4 vCPUs), otherwise fixed iterations.
Each pairing is one private kernel ut-match-NAME-i. UT_KP=2 selects the second account
(~/.kaggle/user_2, ~/.kaggle/access_token_2).
"""
import glob, json, math, os, re, shutil, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROF = os.environ.get("UT_KP", "1")
_SUF = "" if PROF == "1" else f"_{PROF}"
USER = open(os.path.expanduser(f"~/.kaggle/user{_SUF}")).read().strip()
SFX = "" if PROF == "1" else f"_p{PROF}"
ENV = {**os.environ, "KAGGLE_API_TOKEN": open(os.path.expanduser(f"~/.kaggle/access_token{_SUF}")).read().strip(),
       "PATH": os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"]}
SCRIPT = '''import glob, os, subprocess
A, FA, B, FB, GAMES, ITERS, MS, SEED, OPEN, PA, PB = %r, %r, %r, %r, %d, %d, %d, %d, %d, %r, %r
d = os.path.dirname(glob.glob("/kaggle/input/**/referee.cpp", recursive=True)[0])
print(subprocess.run("lscpu | grep -E 'Model name|^CPU\\\\(s\\\\)'", shell=True, capture_output=True, text=True).stdout, flush=True)
cc = ["g++", "-O3", "-std=c++17", "-march=native", "-DLOCAL"]
subprocess.check_call(cc + ["-pthread", "-o", "/tmp/referee", d + "/referee.cpp"])
subprocess.check_call(cc + FA.split() + ["-o", "/tmp/a", d + "/" + A])
subprocess.check_call(cc + FB.split() + ["-o", "/tmp/b", d + "/" + B])
mode = ["-j", "2", "-n", "0", "-t", str(MS), "-x", "20"] if MS else ["-j", "4", "-n", str(ITERS)]
pp = (["-pa", PA] if PA else []) + (["-pb", PB] if PB else [])
out = subprocess.check_output(["/tmp/referee", "/tmp/a", "/tmp/b", "-g", str(GAMES), "-s", str(SEED), "-o", str(OPEN)] + mode + pp, text=True)
print("RESULT", A + "@" + FA + "#" + PA, B + "@" + FB + "#" + PB, out.strip().splitlines()[-1], flush=True)
'''


def kg(*a):
    return subprocess.run(["kaggle", *a], env=ENV, capture_output=True, text=True)


def push(name, specs, games, iters, ms, opn):
    d = os.path.join(ROOT, "kaggle", "matchsrc" + SFX)
    os.makedirs(d, exist_ok=True)
    new = not os.path.exists(os.path.join(d, "dataset-metadata.json"))
    json.dump({"title": "ut-match", "id": f"{USER}/ut-match", "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(d, "dataset-metadata.json"), "w"))
    shutil.copy(os.path.join(ROOT, "tools", "referee.cpp"), d)
    bots = []
    for sp in specs:
        sp, _, params = sp.partition("#")
        f, _, flags = sp.partition("@")
        dst = name + "_" + os.path.basename(f)
        shutil.copy(f, os.path.join(d, dst))
        bots.append((dst, flags, params))
    r = kg("datasets", "create", "-p", d) if new else kg("datasets", "version", "-p", d, "-m", name)
    print((r.stdout + r.stderr).strip().splitlines()[-1][:120])
    for _ in range(40):
        time.sleep(6)
        if "ready" in kg("datasets", "status", f"{USER}/ut-match").stdout:
            break
    time.sleep(10)
    for i in range(0, len(bots), 2):
        k = os.path.join(ROOT, "kaggle", f"m{SFX}_{name}_{i // 2 + 1}")
        os.makedirs(k, exist_ok=True)
        seed = 777 + 1000 * (i // 2) + sum(map(ord, name + SFX))
        open(os.path.join(k, "match.py"), "w").write(
            SCRIPT % (bots[i][0], bots[i][1], bots[i + 1][0], bots[i + 1][1], games, iters, ms, seed, opn,
                      bots[i][2], bots[i + 1][2]))
        json.dump({"id": f"{USER}/ut-match-{name}-{i // 2 + 1}", "title": f"ut-match-{name}-{i // 2 + 1}",
                   "code_file": "match.py", "language": "python", "kernel_type": "script", "is_private": True,
                   "enable_gpu": False, "enable_internet": False, "dataset_sources": [f"{USER}/ut-match"]},
                  open(os.path.join(k, "kernel-metadata.json"), "w"))
        r = kg("kernels", "push", "-p", k)
        print(i // 2 + 1, (r.stdout + r.stderr).strip().splitlines()[-1][:110], flush=True)


def kernels(name):
    return sorted(glob.glob(os.path.join(ROOT, "kaggle", f"m{SFX}_{name}_*")), key=lambda k: int(k.rsplit("_", 1)[1]))


def status(name):
    st = []
    for k in kernels(name):
        r = kg("kernels", "status", f"{USER}/ut-match-{name}-{k.rsplit('_', 1)[1]}")
        m = re.search(r"KernelWorkerStatus\.(\w+)", r.stdout + r.stderr)
        st.append(m.group(1) if m else "?")
    print(" ".join(st))
    return st


def fetch(name):
    for k in kernels(name):
        i = k.rsplit("_", 1)[1]
        d = os.path.join(ROOT, "kaggle", "out", f"m{SFX}_{name}_{i}")
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        kg("kernels", "output", f"{USER}/ut-match-{name}-{i}", "-p", d)
        for f in glob.glob(os.path.join(d, "*.log")):
            txt = open(f).read().replace('\\"', '"')
            m = re.search(r'RESULT (\S+?)@(.*?) (\S+?)@(.*?) (\{.*?\})', txt)
            if not m:
                print(i, "no result:", txt[-400:]); continue
            s = json.loads(m.group(5))
            w, l, dr = s["wins_a"], s["wins_b"], s["draws"]
            n = w + l + dr
            p = (w + dr / 2 + 0.5) / (n + 1)
            var = (w * (1 - p) ** 2 + l * p ** 2 + dr * (0.5 - p) ** 2) / max(n, 1)
            se = math.sqrt(var / max(n, 1))
            elo = -400 * math.log10(1 / p - 1)
            err = (-400 * math.log10(1 / min(p + 1.96 * se, .999) - 1) + 400 * math.log10(1 / max(p - 1.96 * se, .001) - 1)) / 2
            print(f"{m.group(1)} [{m.group(2)}] vs {m.group(3)} [{m.group(4)}]: +{w} ={dr} -{l} ({100 * p:.1f}%)  elo {elo:+.0f} +/- {err:.0f}"
                  f"  faults {s['faults_a']}/{s['faults_b']}" +
                  (f"  [{s['ms']} ms, max {s['max_ms_a']:.0f}/{s['max_ms_b']:.0f}]" if not s["iters"] else f"  [{s['iters']} it]"))


if __name__ == "__main__":
    c, name, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    if c == "push":
        games, iters, ms, opn, files = 300, 20000, 0, 2, []
        i = 0
        while i < len(rest):
            if rest[i] == "--games": games = int(rest[i + 1]); i += 2
            elif rest[i] == "--iters": iters = int(rest[i + 1]); i += 2
            elif rest[i] == "--ms": ms = int(rest[i + 1]); i += 2
            elif rest[i] == "--open": opn = int(rest[i + 1]); i += 2
            else: files.append(rest[i]); i += 1
        push(name, files, games, iters, ms, opn)
    elif c == "status": status(name)
    elif c == "fetch": fetch(name)
