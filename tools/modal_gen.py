"""Self-play data generation on Modal.

  modal run tools/modal_gen.py --workers 2 --iters 4000 --seed 50000 --hours 1.5 --tag g0m [--net tools/nets/X.bin --hid 256]
Each worker is one container; it compiles the engine and runs one generator per hardware
thread, appending to the Modal volume "ut-data" under /<tag>/ (committed every 90 s, so a
stopped container loses little). Download:  modal volume get ut-data /<tag> data/
"""
import os
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = os.environ.get("UT_MODAL_NET", "")
app = modal.App("ut-gen")
image = modal.Image.debian_slim().apt_install("g++").add_local_file(os.path.join(ROOT, "bot.cpp"), "/src/bot.cpp")
if NET:
    image = image.add_local_file(os.path.join(ROOT, NET), "/src/net.bin")
vol = modal.Volume.from_name("ut-data", create_if_missing=True)


@app.function(image=image, cpu=8, memory=8192, timeout=4 * 3600, volumes={"/out": vol})
def gen(seed: int, iters: int, hours: float, tag: str, hid: int = 0):
    import glob, signal, subprocess, time
    flags = [f"-DNN_H={hid}"] if hid else []
    subprocess.check_call(["g++", "-O3", "-std=c++17", "-march=native", "-DLOCAL"] + flags + ["-o", "/tmp/bot", "/src/bot.cpp"])
    n = len(os.sched_getaffinity(0))
    os.makedirs(f"/out/{tag}", exist_ok=True)
    env = {**os.environ}
    if hid:
        env["UT_NET"] = "/src/net.bin"
    t0 = time.time()
    procs = [subprocess.Popen(["/tmp/bot"], env={**env, "UT_GEN": f"1000000:{iters}:{seed + i}:/out/{tag}/m_{seed + i}.bin"},
                              stderr=subprocess.DEVNULL) for i in range(n)]
    last = time.time()
    while time.time() - t0 < hours * 3600 and any(p.poll() is None for p in procs):
        time.sleep(5)
        if time.time() - last > 90:
            vol.commit()
            last = time.time()
    for p in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGKILL)
        p.wait()
    el = time.time() - t0
    pos = sum(os.path.getsize(f) // 44 for f in glob.glob(f"/out/{tag}/m_*.bin") if int(f.rsplit("_", 1)[1][:-4]) // 100 == seed // 100)
    vol.commit()
    cpu = subprocess.run("grep -m1 'model name' /proc/cpuinfo", shell=True, capture_output=True, text=True).stdout.strip()
    return {"seed": seed, "threads": n, "positions": pos, "seconds": round(el), "per_min": round(pos / el * 60), "cpu": cpu}


@app.local_entrypoint()
def main(workers: int = 1, iters: int = 4000, seed: int = 50000, hours: float = 1.0, tag: str = "test", hid: int = 0):
    args = [(seed + 100 * w, iters, hours, tag, hid) for w in range(workers)]
    tot = 0
    for r in gen.starmap(args, return_exceptions=True):
        print(r, flush=True)
        if isinstance(r, dict):
            tot += r["positions"]
    print("total positions", tot)
