#!/usr/bin/env python3
"""Deep net (train.py --deep L) checkpoint -> files for the float test version of the engine's deep head.

  deep_export.py NET.bin.pt OUT_PREFIX
writes OUT_PREFIX.bin (first layer as int16 like train.py's export, output layer zero; pack.py -n) and
OUT_PREFIX.deep.h (#define NN_DEEP L and the float arrays; pack.py --deep).
"""
import sys
import numpy as np
import torch

QA, NB = 255, 10
sd = torch.load(sys.argv[1], map_location="cpu")
pre = sys.argv[2]
w = np.round(sd["ft.weight"].numpy().T * QA).astype(np.int16)
b = np.round(sd["ft.bias"].numpy() * QA).astype(np.int16)
h = w.shape[1]
with open(pre + ".bin", "wb") as f:
    f.write(w.tobytes()); f.write(b.tobytes())
    f.write(np.zeros((NB, 2 * h), np.int16).tobytes()); f.write(np.zeros(NB, np.int32).tobytes())
w2, b2, wo, bo = (sd[k].numpy().astype(np.float32) for k in ("l2.weight", "l2.bias", "out.weight", "out.bias"))
L = w2.shape[0]
def arr(name, a):
    return f"static const float {name}[] = {{" + ",".join(f"{x:.6g}" for x in a.reshape(-1)) + "};\n"
open(pre + ".deep.h", "w").write(arr("DEEP_W2", w2) + arr("DEEP_B2", b2) + arr("DEEP_WO", wo) + arr("DEEP_BO", bo))
print(f"{pre}: hidden {h}, inputs {w.shape[0]}, deep {L}")
