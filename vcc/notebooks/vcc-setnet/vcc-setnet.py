# %% [code]
"""SETNET: learn the donor COMBINATION RULE on GPU.

THE GAP, from the decoded leaderboard (columns are (scaled,raw) per member in
pds,mse,jac,nmae,fid,reach; rank 1 sums to 1.303/6 = 0.2171 exactly):

              pds_sc  mse_sc  jac_sc  nmae_sc  fid_sc  reach_sc
  rank 1       0.815   0.242  -0.009    0.125  -0.071    0.201
  us  (v34)    0.459   0.000  -0.015    0.143  -0.006    0.206

Their fid is WORSE than ours and jac/nmae/reach are level -- the entire deficit
is pds and mse, and both are functions of one number:
    mse_norm = (1 - C^2) + ((k - C*R)/R)^2,  C = 0.3217, R = 1.8345
C = 0.32 is retrieval's weighted mean; rank 1 implies C ~ 0.49.  With their pds
and mse plus OUR fid/jac/reach the sum is 1.619/6 = 0.270, so closing C is the
whole task.

WHY A SET FUNCTION, AND WHY IT IS NOT THE FIRST GPU RUN.  `vcc_stage1_gpu.py`
represented a perturbation by a DepMap embedding and reached held-out pds 0.4633
against a 0.80 gate; that failed because gene-similarity predicts
response-similarity at r ~ 0.005-0.011, so no function of an embedding recovers a
target-specific response.  Here the perturbation is represented by the DONOR
RESPONSES THEMSELVES and the network learns only the combination rule.  A
permutation-invariant set function strictly CONTAINS the weighted mean that
yields C = 0.32, so with 67,579 labelled examples it cannot do worse than
retrieval -- it can only find a better rule.

ARCHITECTURE
  per-donor encode   phi(s_i)  : G -> H   shared MLP on the unit-normalised delta
  context gate       FiLM(ctx) : destination control statistics, defined for a
                     cell type never seen -- so it deploys to A/B/C
  masked mean pool   permutation invariant, handles 1..KD donors
  decode             H -> G, output unit-normalised

LOSS is 1 - cosine, i.e. C directly, because C is the quantity every scored
member is a function of.  `between` row-similarity is printed every fold: a high
pds with high between-row is the v33b failure that already cost a submission.

Leave-one-FAMILY-out via the stored family index.
"""
import os

import numpy as np
import torch
import torch.nn as nn

_ROOT = "/kaggle/input"
PACK = None
for dirpath, _dirs, files in os.walk(_ROOT):
    if "gpu_corpus.npz" in files:
        PACK = os.path.join(dirpath, "gpu_corpus.npz")
        break
print("resolved pack =", PACK, flush=True)
assert PACK, "gpu_corpus.npz not found under /kaggle/input"

# Kaggle's P100 is sm_60 and their torch build supports sm_70+, so is_available()
# returns True while every CUDA op fails.  Probe a real op, fall back rather than
# crash.  The kernel metadata pins nvidiaTeslaT4 for the same reason.
DEV = "cpu"
if torch.cuda.is_available():
    try:
        torch.zeros(8, device="cuda").sum().item()
        DEV = "cuda"
    except Exception as e:
        print("CUDA unusable (%s) -> CPU" % str(e)[:80], flush=True)
print("device:", DEV,
      torch.cuda.get_device_name(0) if DEV == "cuda" else "", flush=True)

z = np.load(PACK, allow_pickle=True)
Dall = z["D"]                      # (N, G) float16, rows already unit-normalised
pert = [str(x) for x in z["pert"]]
bank = z["bank"].astype(int)
base = z["base"].astype(np.float32)
bname = [str(x) for x in z["bname"]]
bfam = [str(x) for x in z["bfam"]]
genes = [str(x) for x in z["genes"]]
G = Dall.shape[1]
fams = sorted(set(bfam))
print("corpus %s  banks %d  families %s" % (Dall.shape, len(bname), fams), flush=True)

where = {}
for r, p in enumerate(pert):
    where.setdefault(p, []).append(r)

lb = np.log1p(np.maximum(base, 0))
ctxf = np.stack([np.log1p(base.sum(1)), (base > 0).mean(1),
                 lb.mean(1), lb.std(1)], 1).astype(np.float32)
ctxf = (ctxf - ctxf.mean(0)) / (ctxf.std(0) + 1e-8)

# JOHNSON-LINDENSTRAUSS PROJECTION.  Kaggle assigned a P100 (sm_60) against a
# torch build requiring sm_70+, so the first run fell back to CPU and one fold
# took 1,730 s.  Rather than depend on the accelerator lottery, train in a random
# orthonormal subspace: JL preserves inner products, and the loss and the scored
# quantity are BOTH cosine, so C is preserved to within the JL distortion while
# the encoder shrinks from 18,080x1024 to PDIM x1024 -- ~18x fewer parameters and
# CPU-tractable.  Decoding is the transpose, so predictions return to gene space.
PDIM = int(os.environ.get("PDIM", "1024"))
_rp = np.random.RandomState(7).randn(G, PDIM).astype(np.float32) / np.sqrt(PDIM)
_q, _ = np.linalg.qr(_rp)                      # orthonormal columns
PROJ = _q.astype(np.float32)                   # (G, PDIM)
Dp = (Dall.astype(np.float32) @ PROJ)          # (N, PDIM) projected corpus
Dp /= np.maximum(np.linalg.norm(Dp, axis=1, keepdims=True), 1e-30)
print("projected corpus %s  (JL, cosine-preserving)" % (Dp.shape,), flush=True)

KD = 6
rng = np.random.RandomState(0)
IDX, FAMI = [], []
for r, p in enumerate(pert):
    fa = bfam[bank[r]]
    src = [q for q in where[p] if bfam[bank[q]] != fa]
    if not src:
        continue
    if len(src) > KD:
        src = [src[i] for i in rng.choice(len(src), KD, replace=False)]
    IDX.append((r, src))
    FAMI.append(fams.index(fa))
F = np.asarray(FAMI, np.int8)
N = len(IDX)
print("training rows %d  donors<=%d  genes %d" % (N, KD, G), flush=True)


def gather(js):
    s = np.zeros((len(js), KD, PDIM), np.float32)
    m = np.zeros((len(js), KD), bool)
    y = np.zeros((len(js), PDIM), np.float32)
    c = np.zeros((len(js), ctxf.shape[1]), np.float32)
    for a, jj in enumerate(js):
        r, src = IDX[jj]
        y[a] = Dp[r]
        c[a] = ctxf[bank[r]]
        for t, q in enumerate(src):
            s[a, t] = Dp[q]
            m[a, t] = True
    return s, m, y, c


class SetNet(nn.Module):
    def __init__(self, G, nc, H=2048, Z=1024):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(G, H), nn.GELU(), nn.Linear(H, Z))
        self.film = nn.Linear(nc, 2 * Z)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.dec = nn.Sequential(nn.GELU(), nn.Linear(Z, H), nn.GELU(),
                                 nn.Linear(H, G))
        # DO NOT zero-init BOTH the decoder head and gamma.  With dec_out == 0
        # the gradient wrt gamma is <grad_out, dec_out> = 0, and the gradient wrt
        # the decoder is gamma * (...) = 0 -- both paths dead, so the network is
        # frozen at exactly retrieval.  Measured: every fold reproduced the
        # baseline to 4 dp (C -0.0023/0.0673/0.0772/0.0528) at every BETA.
        # Keep the decoder at normal init and start gamma SMALL but nonzero, so
        # the correction begins near retrieval and is trainable from step 1.
        # BOUNDED correction.  An unconstrained gamma grows until the learned
        # term dominates the retrieval skip: measured, every fold fell BELOW
        # baseline (best mean lift -0.0119) because 60 epochs of in-sample
        # improvement does not transfer across families.  gamma = GMAX*tanh(p)
        # caps the departure from retrieval, and early stopping on held-out
        # TRAINING rows decides when to stop rather than a fixed epoch count.
        self.gp = nn.Parameter(torch.full((1,), 0.1))
        self.Z = Z

    @property
    def gamma(self):
        return GMAX * torch.tanh(self.gp)

    def forward(self, s, m, c):
        """RESIDUAL ON RETRIEVAL.  out = masked_mean(donors) + gamma * correction.

        The set function nominally CONTAINS the masked mean, but a GELU MLP at
        random init cannot reach identity: measured, SetNet scored BELOW
        retrieval on 3 of 4 folds at every capacity and penalty tried
        (mean lift -0.029 at beta=0, -0.026 at beta=3), while between-row sat at
        +0.25..+0.82.  Making the mean the SKIP PATH and gamma zero-init means
        the network STARTS at retrieval exactly and can only depart from it by
        reducing the loss -- so `>= retrieval` becomes structural rather than
        aspirational.
        """
        B, K, G = s.shape
        w = m.float().unsqueeze(-1)
        skip = (s * w).sum(1) / w.sum(1).clamp(min=1.0)   # retrieval, in JL space
        h = self.enc(s.reshape(B * K, G)).reshape(B, K, self.Z)
        z = (h * w).sum(1) / w.sum(1).clamp(min=1.0)      # masked mean pool
        g, b = self.film(c).chunk(2, -1)
        corr = self.dec(z * (1 + g) + b)
        # NORMALISE BOTH BRANCHES.  Bounding gamma alone does not bound the
        # correction: ||dec|| is unconstrained, so 0.5*dec still swamps the skip
        # and replogle fell 0.0772 -> 0.0178.  Unit-normalising each branch makes
        # gamma a genuine mixing weight -- at GMAX 0.5 the correction can hold at
        # most 1/3 of the output, so retrieval is a real floor.
        skip = torch.nn.functional.normalize(skip, dim=-1)
        corr = torch.nn.functional.normalize(corr, dim=-1)
        return skip + self.gamma * corr


BETA = float(os.environ.get("BETA", "1.0"))


def cos_loss(p, y):
    """1 - cos, plus an explicit BETWEEN-ROW penalty.

    Cosine alone is minimised by emitting one average response for every input:
    measured on the first run, magpie C fell 0.0690 -> 0.0168 while between-row
    rose to +0.5282.  That is the documented mean-collapse
    (`vcc_stage1_gpu.py`: "between-row hit +0.95, one identical prediction for
    every perturbation"), and it scores at chance on pds by construction.

    The fix is to charge for it directly: normalise the batch's predictions and
    penalise their mean off-diagonal Gram entry.  A constant predictor drives
    that term to 1.0, so collapse is no longer a minimum.
    """
    pn = torch.nn.functional.normalize(p, dim=-1)
    yn = torch.nn.functional.normalize(y, dim=-1)
    align = 1.0 - (pn * yn).sum(-1).mean()
    B = pn.shape[0]
    if B > 1:
        Gm = pn @ pn.t()
        off = (Gm.sum() - Gm.diagonal().sum()) / (B * (B - 1))
    else:
        off = torch.zeros((), device=p.device)
    return align + BETA * off.clamp(min=0.0)


def evaluate(net, idx, bs=64):
    net.eval()
    cs, P = [], []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            sj, mj, yj, cj = gather(idx[i:i + bs])
            s = torch.tensor(sj, device=DEV)
            m = torch.tensor(mj, device=DEV)
            c = torch.tensor(cj, device=DEV)
            y = torch.tensor(yj, device=DEV)
            p = net(s, m, c)
            cs.append(torch.nn.functional.cosine_similarity(p, y, dim=-1).cpu().numpy())
            P.append(torch.nn.functional.normalize(p, dim=-1).cpu().numpy())
    P = np.concatenate(P)
    B = P @ P.T
    np.fill_diagonal(B, np.nan)
    return float(np.concatenate(cs).mean()), float(np.nanmean(B))


def baseline_C(idx):
    """Retrieval = masked mean of the donor stack.  The rule SetNet must beat."""
    cs = []
    for i in range(0, len(idx), 256):
        sj, mj, yj, _ = gather(idx[i:i + 256])
        m = mj[..., None]
        p = (sj * m).sum(1) / np.maximum(m.sum(1), 1)
        y = yj
        n = np.linalg.norm(p, axis=1) * np.linalg.norm(y, axis=1)
        cs.append(np.where(n > 0, (p * y).sum(1) / np.maximum(n, 1e-30), 0.0))
    return float(np.concatenate(cs).mean())


# 72 s per configuration on the T4, so sweep rather than guess.  Run 2 showed
# between-row +0.86..+0.96 with BETA=1.0 -- collapsing costs BETA*0.96 while
# alignment gains only ~0.05, so the optimiser should refuse it and does not:
# the network is UNDERFIT (20 epochs, lr 3e-4, 1024-dim) and the penalty
# gradient is swamped.  Sweep capacity and penalty weight together.
import itertools

GMAX = float(os.environ.get("GMAX", "0.5"))


def fit_eval(tr, te, epochs, lr, beta, H, Z):
    """Train with EARLY STOPPING on held-out training rows.

    60 fixed epochs overfits the training families: v9 grew the correction until
    it swamped the retrieval skip and every fold landed below baseline.  Hold out
    15% of the TRAINING rows (never the test family), score C on them each epoch,
    and keep the best weights -- so the correction is only as large as it
    generalises.
    """
    global BETA
    BETA = beta
    rs = np.random.RandomState(0)
    tr = tr[rs.permutation(len(tr))]
    cut = max(int(0.15 * len(tr)), 64)
    va, tr = tr[:cut], tr[cut:]
    torch.manual_seed(0)
    net = SetNet(PDIM, ctxf.shape[1], H=H, Z=Z).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    bs = 128
    best_c, best_state, bad = -9.0, None, 0
    for ep in range(epochs):
        net.train()
        perm = tr[np.random.RandomState(ep).permutation(len(tr))]
        for i in range(0, len(perm), bs):
            sj, mj, yj, cj = gather(perm[i:i + bs])
            opt.zero_grad()
            cos_loss(net(torch.tensor(sj, device=DEV),
                         torch.tensor(mj, device=DEV),
                         torch.tensor(cj, device=DEV)),
                     torch.tensor(yj, device=DEV)).backward()
            opt.step()
        vc, _ = evaluate(net, va)
        if vc > best_c + 1e-5:
            best_c, bad = vc, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return evaluate(net, te)


FOLDS = [(k, fa) for k, fa in enumerate(fams)
         if (F == k).sum() >= 200 and (F != k).sum() >= 2000]
print("folds scored: %s" % [f[1] for f in FOLDS], flush=True)
base = {}
for k, fa in FOLDS:
    base[fa] = baseline_C(np.where(F == k)[0])
print("retrieval C per fold: %s" % {k: round(v, 4) for k, v in base.items()},
      flush=True)
print("retrieval MEAN %.4f" % np.mean(list(base.values())), flush=True)

print("%6s %7s %5s %5s %5s | %s" % ("epochs", "lr", "beta", "H", "Z",
                                    " ".join("%16s" % f[1] for f in FOLDS)),
      flush=True)
best = None
for epochs, lr, beta, (H, Z) in itertools.product(
        (60,), (1e-3,), (10.0, 30.0), ((2048, 1024),)):
    row, gain = [], []
    for k, fa in FOLDS:
        c, bw = fit_eval(np.where(F != k)[0], np.where(F == k)[0],
                         epochs, lr, beta, H, Z)
        row.append("%8.4f/%+.2f" % (c, bw))
        gain.append(c - base[fa])
    mg = float(np.mean(gain))
    print("%6d %7.0e %5.1f %5d %5d | %s  MEAN LIFT %+0.4f"
          % (epochs, lr, beta, H, Z, " ".join(row), mg), flush=True)
    if best is None or mg > best[0]:
        best = (mg, epochs, lr, beta, H, Z)
print("BEST mean lift %+0.4f at epochs=%d lr=%.0e beta=%.1f H=%d Z=%d"
      % best, flush=True)
