#!/usr/bin/env python3
"""Self-play data generation on Kaggle CPU kernels (4 vCPUs each, one generator per vCPU).

  kaggle_gen.py upload [NET.bin]                  refresh the private dataset ut-engine (bot.cpp [+ net])
  kaggle_gen.py push TAG N ITERS SEED HOURS [FLAGS] [NET]
                                                  start N kernels ut-TAG-1..N; each runs for HOURS
  kaggle_gen.py status TAG N
  kaggle_gen.py fetch TAG N                       finished outputs -> data/TAG_<seed>.bin
UT_KP=2 selects the second account (~/.kaggle/user_2, ~/.kaggle/access_token_2).
"""
import glob, json, os, re, shutil, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROF = os.environ.get("UT_KP", "1")
_SUF = "" if PROF == "1" else f"_{PROF}"
USER = open(os.path.expanduser(f"~/.kaggle/user{_SUF}")).read().strip()
SFX = "" if PROF == "1" else f"_p{PROF}"
ENV = {**os.environ, "KAGGLE_API_TOKEN": open(os.path.expanduser(f"~/.kaggle/access_token{_SUF}")).read().strip(),
       "PATH": os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"]}
KERNEL = '''import glob, os, signal, subprocess, time
ITERS, SEED, HOURS, FLAGS, NET = %d, %d, %f, %r, %r
src = glob.glob("/kaggle/input/**/bot.cpp", recursive=True)[0]
d = os.path.dirname(src)
subprocess.check_call(["g++", "-O3", "-std=c++17", "-march=native", "-DLOCAL"] + FLAGS.split() + ["-o", "/kaggle/working/bot", src])
n = os.cpu_count()
print(subprocess.run("lscpu | grep -E 'Model name|^CPU\\\\(s\\\\)'", shell=True, capture_output=True, text=True).stdout, flush=True)
env = dict(os.environ)
if NET:
    env["UT_NET"] = d + "/" + NET
t0 = time.time()
procs = [subprocess.Popen(["/kaggle/working/bot"], env={**env, "UT_GEN": "1000000:%%d:%%d:/kaggle/working/k_%%d.bin" %% (ITERS, SEED + i, SEED + i)})
         for i in range(n)]
while time.time() - t0 < HOURS * 3600 and any(p.poll() is None for p in procs):
    time.sleep(20)
for p in procs:
    if p.poll() is None:
        p.send_signal(signal.SIGKILL)
    p.wait()
os.remove("/kaggle/working/bot")
el = time.time() - t0
pos = sum(os.path.getsize(f) for f in glob.glob("/kaggle/working/k_*.bin")) // 44
print("positions %%d in %%.0fs = %%.0f per minute" %% (pos, el, pos / el * 60), flush=True)
'''


def kg(*a):
    return subprocess.run(["kaggle", *a], env=ENV, capture_output=True, text=True)


def upload(net=None):
    d = os.path.join(ROOT, "kaggle", "engine" + SFX)
    new = not os.path.isdir(d)
    os.makedirs(d, exist_ok=True)
    json.dump({"title": "ut-engine", "id": f"{USER}/ut-engine", "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(d, "dataset-metadata.json"), "w"))
    shutil.copy(os.path.join(ROOT, "bot.cpp"), d)
    if net:
        shutil.copy(net, d)
    r = kg("datasets", "create", "-p", d) if new else kg("datasets", "version", "-p", d, "-m", "update")
    print((r.stdout + r.stderr).strip().splitlines()[-1])


def push(tag, n, iters, seed, hours, flags="", net=""):
    for i in range(1, n + 1):
        d = os.path.join(ROOT, "kaggle", f"k{SFX}_{tag}_{i}")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "gen.py"), "w").write(KERNEL % (iters, seed + 100 * i, hours, flags, net))
        json.dump({"id": f"{USER}/ut-{tag}-{i}", "title": f"ut-{tag}-{i}", "code_file": "gen.py", "language": "python",
                   "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": False,
                   "dataset_sources": [f"{USER}/ut-engine"]}, open(os.path.join(d, "kernel-metadata.json"), "w"))
        r = kg("kernels", "push", "-p", d)
        print(i, (r.stdout + r.stderr).strip().splitlines()[-1][:110], flush=True)


def status(tag, n):
    out = []
    for i in range(1, n + 1):
        r = kg("kernels", "status", f"{USER}/ut-{tag}-{i}")
        m = re.search(r"KernelWorkerStatus\.(\w+)", r.stdout + r.stderr)
        out.append(m.group(1) if m else "?")
    print(" ".join(out))
    return out


def fetch(tag, n):
    st = status(tag, n)
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    for i in range(1, n + 1):
        if st[i - 1] != "COMPLETE":
            continue
        d = os.path.join(ROOT, "kaggle", "out", f"{tag}{SFX}_{i}")
        if os.path.isdir(d):
            continue
        os.makedirs(d)
        kg("kernels", "output", f"{USER}/ut-{tag}-{i}", "-p", d)
        for f in glob.glob(os.path.join(d, "k_*.bin")):
            shutil.move(f, os.path.join(ROOT, "data", f"{tag}{SFX}_{os.path.basename(f)[2:]}"))
        print("fetched", i, flush=True)


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    if c == "upload": upload(a[0] if a else None)
    elif c == "push": push(a[0], int(a[1]), int(a[2]), int(a[3]), float(a[4]), a[5] if len(a) > 5 else "", a[6] if len(a) > 6 else "")
    elif c == "status": status(a[0], int(a[1]))
    elif c == "fetch": fetch(a[0], int(a[1]))
