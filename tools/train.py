#!/usr/bin/env python3
"""Train the NNUE-style value net for the Ultimate Tic-Tac-Toe bot.

Data: 44-byte records written by UT_GEN (runGen in bot.cpp), side-to-move frame:
  own u16[9], opp u16[9] (cells per small board), q i16 (root value, (v*2-1)*10000),
  next i8 (-1 = any board), result u8 (0 loss, 1 draw, 2 win), ply, best, flags, side u8.

Net: 189 inputs per perspective (81 own cells, 81 opponent cells, 9 own won boards, 9 opponent
won boards, 9 drawn boards) -> H shared feature transformer, SCReLU, both perspectives
concatenated (side to move first) -> 10 output heads chosen by the next board (9 = any).
Training uses the 8 symmetries of the board (the same 3x3 permutation on boards and cells).

Output file: int16 W[189][H], int16 B[H], int16 O[10][2H], int32 OB[10] (QA=255, QB=64),
then with --draw a second head predicting P(draw): int16 D[10][2H], int32 DB[10].

  train.py -o tools/nets/n0.bin 'data/g0*.bin' [--hidden 256] [--epochs 20] [--lam 0.5]
"""
import argparse, glob, os, time
import numpy as np
import torch
import torch.nn as nn

REC = np.dtype([("own", "<u2", 9), ("opp", "<u2", 9), ("q", "<i2"), ("next", "i1"), ("result", "u1"),
                ("ply", "u1"), ("best", "u1"), ("flags", "u1"), ("side", "u1")])
QA, QB = 255, 64
NIN, NB = 189, 10
LINES = [0x7, 0x38, 0x1C0, 0x49, 0x92, 0x124, 0x111, 0x54]


def syms():
    """The 8 symmetries of a 3x3 grid as index permutations."""
    g = np.arange(9).reshape(3, 3)
    out = []
    for k in range(4):
        r = np.rot90(g, k)
        out.append(r.reshape(-1))
        out.append(np.fliplr(r).reshape(-1))
    return np.array(out)


class Net(nn.Module):
    def __init__(self, h, draw=False, nin=NIN, deep=0):
        super().__init__()
        self.h = h
        self.ft = nn.Linear(nin, h)
        # deep > 0: a hidden layer of that width before the output (CReLU; .pt only for now)
        self.l2 = nn.Linear(2 * h, deep) if deep else None
        self.out = nn.Linear(deep if deep else 2 * h, NB)
        self.dout = nn.Linear(2 * h, NB) if draw else None
        nn.init.uniform_(self.ft.weight, -0.1, 0.1)
        nn.init.constant_(self.ft.bias, 0.05)

    def forward(self, a, b, bucket):
        ha = torch.clamp(self.ft(a), 0, 1) ** 2
        hb = torch.clamp(self.ft(b), 0, 1) ** 2
        hc = torch.cat([ha, hb], dim=1)
        if self.l2 is not None:
            z = self.out(torch.clamp(self.l2(hc), 0, 1)).gather(1, bucket[:, None]).squeeze(1)
            return z, None
        z = self.out(hc).gather(1, bucket[:, None]).squeeze(1)
        if self.dout is None:
            return z, None
        return z, self.dout(hc).gather(1, bucket[:, None]).squeeze(1)


def export(model, path):
    w = np.round(model.ft.weight.detach().cpu().numpy().T * QA).astype(np.int16)      # [189][H]
    b = np.round(model.ft.bias.detach().cpu().numpy() * QA).astype(np.int16)          # [H]
    o = np.round(model.out.weight.detach().cpu().numpy() * QB).astype(np.int16)       # [10][2H]
    ob = np.round(model.out.bias.detach().cpu().numpy() * QA * QB).astype(np.int32)   # [10]
    with open(path, "wb") as f:
        f.write(w.tobytes()); f.write(b.tobytes()); f.write(o.tobytes()); f.write(ob.tobytes())
        if model.dout is not None:
            f.write(np.round(model.dout.weight.detach().cpu().numpy() * QB).astype(np.int16).tobytes())
            f.write(np.round(model.dout.bias.detach().cpu().numpy() * QA * QB).astype(np.int32).tobytes())
    return int(np.abs(w).max()), int(np.abs(o).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam", type=float, default=0.5, help="weight of the game result vs the search value")
    ap.add_argument("--dedup", action="store_true", help="merge identical positions, averaging their targets")
    ap.add_argument("--max", type=int, default=0, help="use at most this many positions")
    ap.add_argument("--init", help="float checkpoint (.pt) to continue from")
    ap.add_argument("--draw", type=float, default=0, help="train a draw-probability head with this loss weight")
    ap.add_argument("--threats", action="store_true", help="18 extra inputs (small-board win threats per player)")
    ap.add_argument("--mthreats", action="store_true", help="experimental: 18 more inputs (open boards whose capture wins the game, per player)")
    ap.add_argument("--alive", action="store_true", help="experimental: 27 more inputs (open boards each player can still win; dead open boards)")
    ap.add_argument("--deep", type=int, default=0, help="hidden layer of this width between the accumulator and the output buckets")
    ap.add_argument("--teacher", help="float checkpoint (.pt) of a teacher net trained on the same files and options: "
                    "train on its predictions (averaged over the 8 symmetries)")
    ap.add_argument("--mu", type=float, default=1.0, help="weight of the teacher prediction vs the original target")
    ap.add_argument("--wclip", type=float, default=127 / 255, help="clip first-layer weights and biases (int8 at QA=255)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    a = ap.parse_args()
    t0 = time.time()

    paths = sorted(set(p for f in a.files for p in glob.glob(f, recursive=True)))
    def load(p):   # files may still be growing: drop a partial trailing record
        b = np.fromfile(p, dtype=np.uint8)
        return b[: len(b) // REC.itemsize * REC.itemsize].view(REC)
    data = np.concatenate([load(p) for p in paths])
    if a.max:
        data = data[np.random.default_rng(1).permutation(len(data))[:a.max]]
    n = len(data)
    result = data["result"].astype(np.float32) / 2
    q = (data["q"].astype(np.float32) + 10000) / 20000
    target = (a.lam * result + (1 - a.lam) * q).astype(np.float32)
    raw = np.empty((n, 19), dtype=np.int16)        # own[9], opp[9] (9-bit masks), next board (9 = any)
    raw[:, :9] = data["own"]
    raw[:, 9:18] = data["opp"]
    raw[:, 18] = np.where(data["next"] < 0, 9, data["next"])
    drawt = (data["result"] == 1).astype(np.float32)
    print(f"{n} positions from {len(paths)} files; stm result: win {np.mean(data['result'] == 2):.3f} "
          f"draw {np.mean(data['result'] == 1):.3f}; mean q {q.mean():.3f} ({time.time() - t0:.0f}s)", flush=True)
    del data
    if a.dedup:   # merge identical positions (64-bit hash of the 19 fields), averaging their targets
        keys = np.zeros(n, dtype=np.uint64)
        mul = np.uint64(0x9E3779B97F4A7C15)
        for j in range(19):
            keys ^= (raw[:, j].astype(np.uint64) + np.uint64(j * 1021 + 7)) * np.uint64((0xD6E8FEB86659FD93 + 2 * j * 0x632BE5AB) & 0xFFFFFFFFFFFFFFFF | 1)
            keys = (keys ^ (keys >> np.uint64(31))) * mul
        _, first, inv, cnt = np.unique(keys, return_index=True, return_inverse=True, return_counts=True)
        del keys
        target = (np.bincount(inv, weights=target) / cnt).astype(np.float32)
        result = (np.bincount(inv, weights=result) / cnt).astype(np.float32)
        drawt = (np.bincount(inv, weights=drawt) / cnt).astype(np.float32)
        raw = raw[first]
        print(f"dedup: {n} -> {len(raw)} unique ({100 * len(raw) / n:.1f}%) ({time.time() - t0:.0f}s)", flush=True)
        n = len(raw)
    perm0 = np.random.default_rng(2).permutation(n)
    raw, target, result, drawt = raw[perm0], target[perm0], result[perm0], drawt[perm0]
    nval = min(200000, n // 20)

    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    X = torch.from_numpy(raw).to(dev)
    Y = torch.from_numpy(target).to(dev)
    R = torch.from_numpy(result).to(dev)
    DT = torch.from_numpy(drawt).to(dev)
    bits = torch.tensor([[(v >> i) & 1 for i in range(9)] for v in range(512)], dtype=torch.float32, device=dev)
    win = torch.tensor([float(any((v & l) == l for l in LINES)) for v in range(512)], device=dev)
    S = torch.from_numpy(syms()).to(dev)                      # (8, 9)
    # permuting the 9 bits of a mask: PB[s][mask] = permuted mask
    pb = np.zeros((8, 512), dtype=np.int64)
    for s, p in enumerate(syms()):
        for v in range(512):
            pb[s, v] = sum(1 << i for i in range(9) if (v >> p[i]) & 1)
    PB = torch.from_numpy(pb).to(dev)

    def feats(xb, sym):
        own, opp, nxt = xb[:, :9].long(), xb[:, 9:18].long(), xb[:, 18].long()
        if sym is not None:                                   # apply symmetry s to boards and cells
            p = S[sym]                                        # (B, 9)
            own = PB[sym[:, None], torch.gather(own, 1, p)]
            opp = PB[sym[:, None], torch.gather(opp, 1, p)]
            inv = torch.argsort(p, dim=1)
            nxt = torch.where(nxt < 9, torch.gather(inv, 1, nxt.clamp(max=8)[:, None]).squeeze(1), nxt)
        wo, wp = win[own], win[opp]                           # (B, 9) won boards
        drawn = ((own | opp) == 511).float() * (1 - wo) * (1 - wp)
        co, cp = bits[own].reshape(-1, 81), bits[opp].reshape(-1, 81)
        if a.threats:   # a player threatens a small board: two in a line, third cell empty, board open
            open_ = ((own | opp) != 511).float() * (1 - wo) * (1 - wp)
            to, tp = THR[own, opp] * open_, THR[opp, own] * open_
            stm, oth = [co, cp, wo, wp, drawn, to, tp], [cp, co, wp, wo, drawn, tp, to]
            if a.mthreats:   # board bd is open and winning it completes a line of won boards
                p2 = 2 ** torch.arange(9, device=dev)
                mo, mp = (wo.long() * p2).sum(1), (wp.long() * p2).sum(1)
                mto, mtp = bits[TWB[mo]] * open_, bits[TWB[mp]] * open_
                stm, oth = stm + [mto, mtp], oth + [mtp, mto]
            if a.alive:   # open board a player can still win (a line free of opponent stones); dead open boards
                ao, ap_ = ALV[opp] * open_, ALV[own] * open_
                dead = open_ * (1 - ao) * (1 - ap_)
                stm, oth = stm + [ao, ap_, dead], oth + [ap_, ao, dead]
            return torch.cat(stm, dim=1), torch.cat(oth, dim=1), nxt
        stm = torch.cat([co, cp, wo, wp, drawn], dim=1)
        oth = torch.cat([cp, co, wp, wo, drawn], dim=1)
        return stm, oth, nxt

    if a.threats:
        thr = np.zeros((512, 512), dtype=np.float32)
        for m in range(512):
            for o in range(512):
                if m & o:
                    continue
                e = 511 & ~(m | o)
                thr[m, o] = float(any(bin(m & l).count("1") == 2 and (e & l) for l in LINES))
        THR = torch.from_numpy(thr).to(dev)
        twb = np.zeros(512, dtype=np.int64)   # cells that complete a line for this mask
        for m in range(512):
            for c in range(9):
                if not (m >> c) & 1 and any(((m | 1 << c) & l) == l for l in LINES):
                    twb[m] |= 1 << c
        TWB = torch.from_numpy(twb).to(dev)
        # ALV[o] = 1 if some line holds none of the opponent's stones o
        ALV = torch.tensor([float(any((o & l) == 0 for l in LINES)) for o in range(512)], device=dev)
    nin = NIN + (18 if a.threats else 0) + (18 if a.mthreats else 0) + (27 if a.alive else 0)
    YT = Y   # training target (the validation loss always uses the original target)
    if a.teacher:
        if not os.path.exists(a.teacher):
            a.teacher = glob.glob("/kaggle/input/**/" + os.path.basename(a.teacher), recursive=True)[0]
        sd = torch.load(a.teacher, map_location=dev)
        tm = Net(sd["ft.weight"].shape[0], False, nin, sd["l2.weight"].shape[0] if "l2.weight" in sd else 0).to(dev)
        tm.load_state_dict({k: v for k, v in sd.items() if not k.startswith("dout")})
        tm.eval()
        YT = torch.empty_like(Y)
        with torch.no_grad():
            for i in range(0, n, 65536):
                idx = torch.arange(i, min(i + 65536, n), device=dev)
                acc = torch.zeros(len(idx), device=dev)
                for sy in range(8):
                    s_, o_, k_ = feats(X[idx], torch.full((len(idx),), sy, device=dev, dtype=torch.long))
                    acc += torch.sigmoid(tm(s_, o_, k_)[0])
                YT[idx] = a.mu * acc / 8 + (1 - a.mu) * Y[idx]
        del tm
        print(f"teacher {a.teacher}: mean target {float(YT.mean()):.3f} vs {float(Y.mean()):.3f}, "
              f"mean |diff| {float((YT - Y).abs().mean()):.3f} ({time.time() - t0:.0f}s)", flush=True)
    model = Net(a.hidden, a.draw > 0, nin, a.deep).to(dev)
    if a.init and not os.path.exists(a.init):   # Kaggle: find the checkpoint wherever the dataset is mounted
        a.init = glob.glob("/kaggle/input/**/" + os.path.basename(a.init), recursive=True)[0]
    if a.init:
        sd = torch.load(a.init, map_location=dev)
        w = sd["ft.weight"]
        if w.shape[1] < model.ft.weight.shape[1]:   # widen: new inputs start with zero weights
            print(f"init: inputs {w.shape[1]} -> {model.ft.weight.shape[1]} (new weights 0)", flush=True)
            sd["ft.weight"] = torch.cat([w, torch.zeros(w.shape[0], model.ft.weight.shape[1] - w.shape[1], device=dev)], 1)
        own = model.state_dict()
        drop = [k for k in sd if k in own and sd[k].shape != own[k].shape]   # e.g. the output layer of a shallow net
        for k in drop:
            del sd[k]
        missing = model.load_state_dict(sd, strict=False).missing_keys
        if missing:
            print(f"init: new parameters {missing}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    nb = (n - nval) // a.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * nb, eta_min=a.lr / 400)
    lossf = nn.BCEWithLogitsLoss()

    def evaluate():
        model.eval()
        tot = tr = td = 0.0
        with torch.no_grad():
            for i in range(0, nval, 65536):
                idx = torch.arange(i, min(i + 65536, nval), device=dev)
                s, o, k = feats(X[idx], None)
                z, zd = model(s, o, k)
                tot += float(lossf(z, Y[idx])) * len(idx)
                tr += float(lossf(z, R[idx])) * len(idx)
                if zd is not None:
                    td += float(lossf(zd, DT[idx])) * len(idx)
        model.train()
        return tot / nval, tr / nval, td / nval

    print(f"loaded in {time.time() - t0:.0f}s, training on {a.device}", flush=True)
    t0 = time.time()
    perm = None
    for ep in range(a.epochs):
        del perm   # int32 on the device, shuffled on the host: 200M+ positions fit beside the data
        perm = torch.randperm(n - nval).to(torch.int32).to(dev)
        perm += nval
        run = torch.zeros((), device=dev)
        for i in range(nb):
            idx = perm[i * a.batch:(i + 1) * a.batch].long()
            sym = torch.randint(0, 8, (len(idx),), device=dev)
            s, o, k = feats(X[idx], sym)
            z, zd = model(s, o, k)
            loss = lossf(z, YT[idx])
            if zd is not None:
                loss = loss + a.draw * lossf(zd, DT[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                # quantization ranges of the exported net (deep nets: same first layer, int8 hidden layer)
                model.ft.weight.clamp_(-a.wclip, a.wclip)
                model.ft.bias.clamp_(-a.wclip, a.wclip)
                if a.deep:
                    model.l2.weight.clamp_(-1.98, 1.98)
                else:
                    model.out.weight.clamp_(-1.98, 1.98)
                    if model.dout is not None:
                        model.dout.weight.clamp_(-1.98, 1.98)
                run += loss.detach()
        vl, vr, vd = evaluate()
        print(f"epoch {ep + 1}/{a.epochs} train {float(run) / max(nb, 1):.4f} val {vl:.4f} val-result {vr:.4f} "
              + (f"val-draw {vd:.4f} " if a.draw > 0 else "") +
              f"({time.time() - t0:.0f}s)", flush=True)
        torch.save(model.state_dict(), a.out + ".pt")
        if not a.deep:
            mw, mo = export(model, a.out)
    print(f"saved {a.out}" + (f" (max |W| {mw}, max |O| {mo})" if not a.deep else " (.pt only)"))


if __name__ == "__main__":
    main()
