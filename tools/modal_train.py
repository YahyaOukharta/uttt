"""Train a net on a Modal GPU, on data in the Modal volume "ut-data".

  modal run tools/modal_train.py --name n1a --data 'g0m/*.bin' [--extra 'up/*.bin'] -- (train.py args via --args)
  e.g. modal run tools/modal_train.py --name n1a --data 'g0m/*.bin,up/*.bin' --args '--hidden 256 --epochs 20 --lam 0.5 --dedup'
The net is written to the volume as nets/NAME.bin(.pt) and printed log lines stream back.
Download:  modal volume get ut-data /nets/NAME.bin tools/nets/
Upload local data first if needed:  modal volume put ut-data data/X.bin /up/X.bin
"""
import os
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = modal.App("ut-train")
image = (modal.Image.debian_slim().pip_install("torch", "numpy")
         .add_local_file(os.path.join(ROOT, "tools", "train.py"), "/src/train.py"))
vol = modal.Volume.from_name("ut-data", create_if_missing=True)


@app.function(image=image, gpu="A10G", memory=32768, timeout=3 * 3600, volumes={"/data": vol})
def train(name: str, data: str, args: str):
    import shlex, subprocess
    os.makedirs("/data/nets", exist_ok=True)
    globs = ["/data/" + d.strip() for d in data.split(",") if d.strip()]
    cmd = ["python3", "/src/train.py", "-o", f"/data/nets/{name}.bin"] + globs + shlex.split(args)
    p = subprocess.run(cmd, capture_output=True, text=True)
    vol.commit()
    return p.stdout[-6000:] + p.stderr[-3000:]


@app.local_entrypoint()
def main(name: str, data: str, args: str = "--hidden 256 --epochs 20 --lam 0.5"):
    print(train.remote(name, data, args))
