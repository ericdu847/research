# %% [code]
"""Predict the 300 targets in contexts A/B/C from PUBLIC weights only -- no training.

WHY THIS EXISTS.  Every checkpoint Arc released has a ONE-HOT perturbation encoder over
2,024 Replogle genes (`pert_dim == 2024`), and measured coverage of this competition's 300
targets is 0/300 on all seven.  Feeding our controls into any of them produces nothing,
because the model has no input slot for ABCD1 or ACLY.  That is the entire reason this
project kept trying to TRAIN a replacement.

`VirtualCell2025/st-tahoe-finetune-0826` is a third-party checkpoint that already made the
swap: `pert_dim 5120` (ESM2 features over 19,792 genes), and it covers **300/300** of our
targets.  So the model we spent a day trying to train already exists in public.

    input_dim 2058   gene_dim 18080   output_dim 2058   pert_dim 5120
    cell types: ARC_H1, hepg2, jurkat, k562, rpe1
    pert_col target_gene | cell_type_key cell_type | batch_col batch_var
    control_pert non-targeting | embed_key X_state | output_space all

THE ONE DEPENDENCY.  `embed_key=X_state` means every input cell must first be embedded by
the State Embedding model (SE-600M, 11.5 GB).  On this project's CPU that ran 200 cells for
25 minutes without emitting, which is why this is a GPU kernel and not a local script.

WHAT IS NOT DONE HERE.  The repo also ships `infer/pred_4000.prep.vcc` -- their finished
submission.  That is their answer, not ours, and it is not touched.  Only the weights are
used, which is the same footing as using STATE itself.

CELL-TYPE TOKEN.  A/B/C are not among the model's five known contexts.  The context still
reaches the model through the basal cells (the zero-shot path it was built for), but the
token must be set to something, so it follows Chorus' own measured lineage affinity:
A lymphoid -> jurkat, B and C epithelial -> hepg2.  Swept via CT_FOR.

OUTPUT is a per-context delta npz on the 18,533 submission axis, ready for `submit26.py`.
The model emits 18,080 genes; the ~456 outside it are left at exactly zero, never imputed.
"""
import glob
import os
import subprocess
import sys
import threading
import time

T0 = time.time()
os.makedirs("/kaggle/working", exist_ok=True)
_LOG = open("/kaggle/working/run.log", "a", buffering=1, errors="replace")


class _Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, d):
        for x in self.s:
            try:
                x.write(d)
            except Exception:
                pass

    def flush(self):
        for x in self.s:
            try:
                x.flush()
            except Exception:
                pass


sys.stdout = _Tee(sys.__stdout__, _LOG)
sys.stderr = _Tee(sys.__stderr__, _LOG)


def el():
    return "%5.1fm" % ((time.time() - T0) / 60)


def sh(cmd, check=True):
    print("\n$ " + cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, errors="replace",
                         bufsize=1)
    for line in p.stdout:
        print(line.rstrip(), flush=True)
    p.wait()
    if check and p.returncode != 0:
        raise SystemExit("FAILED (%d): %s" % (p.returncode, cmd))
    return p.returncode


def heartbeat(every=60):
    def loop():
        while True:
            time.sleep(every)
            try:
                fs = sorted(glob.glob("/kaggle/temp/**/*.h5ad", recursive=True))
                print("[HB %s] files=%d %s" % (el(), len(fs),
                      os.path.basename(fs[-1]) if fs else ""), flush=True)
            except Exception as e:
                print("[HB %s] %s" % (el(), str(e)[:100]), flush=True)
    threading.Thread(target=loop, daemon=True).start()


SCRATCH = "/kaggle/temp"
os.makedirs(SCRATCH, exist_ok=True)
def _find_controls():
    """Locate the control files wherever Kaggle mounted them.

    The dataset was uploaded with --dir-mode zip, so thefive files are not necessarily at
    /kaggle/input/<slug>/ -- they may sit one level down or inside an extracted folder.
    Search for pert_counts.csv and take its directory rather than assuming a path; a
    wrong guess here wastes the whole GPU session on a FileNotFoundError twelve GB in.
    """
    import glob as _g
    hits = _g.glob("/kaggle/input/**/pert_counts.csv", recursive=True)
    if not hits:
        raise SystemExit("ABORT: pert_counts.csv not found under /kaggle/input; got %s"
                         % _g.glob("/kaggle/input/**", recursive=True)[:40])
    d = os.path.dirname(hits[0])
    print("  [ctl] controls at %s -> %s" % (d, sorted(os.listdir(d))[:8]), flush=True)
    return d


CTL = _find_controls()
NCELL = int(os.environ.get("VI_NCELL", "16"))
NCTRL = int(os.environ.get("VI_NCTRL", "128"))
CT_FOR = {"A": "jurkat", "B": "hepg2", "C": "hepg2"}

sh("pip -q install arc-state 2>&1 | tail -2", check=False)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", check=False)


def _gpu_guard():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("ABORT: no CUDA device. Re-run on the T4 accelerator.")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    try:
        torch.zeros(1024, 1024, device="cuda").matmul(
            torch.zeros(1024, 1024, device="cuda")).sum().item()
    except Exception as e:
        raise SystemExit("ABORT: CUDA present but unusable (%s) -- almost certainly an "
                         "sm_60 P100 against an sm_70+ torch build." % str(e)[:150])
    print("  [guard] CUDA OK on %s" % names, flush=True)


_gpu_guard()
heartbeat()

# ---------------------------------------------------------------- weights
from huggingface_hub import hf_hub_download   # noqa: E402
import shutil                                  # noqa: E402

TX = os.path.join(SCRATCH, "stft")
os.makedirs(os.path.join(TX, "checkpoints"), exist_ok=True)
for f in ["config.yaml", "var_dims.pkl", "pert_onehot_map.pt",
          "cell_type_onehot_map.pkl", "batch_onehot_map.pkl", "data_module.torch",
          "checkpoints/step=8000.ckpt"]:
    d = os.path.join(TX, f)
    if not os.path.exists(d):
        shutil.copy(hf_hub_download("VirtualCell2025/st-tahoe-finetune-0826", f), d)
    print("  tx  %-24s %8.1f MB" % (os.path.basename(f), os.path.getsize(d) / 1e6),
          flush=True)
shutil.copy(os.path.join(TX, "checkpoints", "step=8000.ckpt"),
            os.path.join(TX, "checkpoints", "final.ckpt"))

SE = os.path.join(SCRATCH, "se600m")
os.makedirs(SE, exist_ok=True)
for f in ["config.yaml", "protein_embeddings.pt", "se600m_epoch16.ckpt"]:
    d = os.path.join(SE, f)
    if not os.path.exists(d):
        shutil.copy(hf_hub_download("arcinstitute/SE-600M", f), d)
    print("  se  %-24s %8.2f GB" % (f, os.path.getsize(d) / 1e9), flush=True)

# ---------------------------------------------------------------- slim the SE checkpoint
# OOM (exit 137) happens INSIDE `torch.load` of se600m_epoch16.ckpt, before any data is
# touched: 4,928 cells and batch 8 still died 60 s after "Loading model from checkpoint".
# The file is 11.5 GB but a 600M-parameter model is only ~2.4 GB of weights -- the rest is
# TRAINING STATE (optimizer moments, etc.), and torch.load pulls all of it into RAM before
# the model is even constructed.
#
# `Inference.load_model` ultimately calls Lightning's `StateEmbeddingModel.
# load_from_checkpoint`, which needs exactly two keys: `state_dict` and
# `hyper_parameters`.  So mmap-load (which defers tensor storage rather than reading it),
# keep those two, and write a slim checkpoint.  The 11.5 GB original is then deleted.
import torch  # noqa: E402
# The checkpoint pickles classes from `vci`, this package's former name.
# `state.emb.__init__._setup_vci_aliases()` installs sys.modules["vci"] -> state.emb, so
# importing it first is what makes the unpickle resolvable.
import state.emb  # noqa: E402,F401

SLIM = os.path.join(SE, "se600m_slim.ckpt")
BIG = os.path.join(SE, "se600m_epoch16.ckpt")
if not os.path.exists(SLIM):
    print("[%s] slimming SE checkpoint (mmap, weights only)" % el(), flush=True)
    try:
        ck = torch.load(BIG, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        ck = torch.load(BIG, map_location="cpu", weights_only=False)
    # `Inference.load_model` reads cfg from `cfg_yaml` or `hyper_parameters`, so those
    # must survive.  `protein_embeds_dict` (411 MB) is dropped because we pass
    # --protein-embeddings explicitly; optimizer_states/lr_schedulers are the bulk.
    keep = {k: ck[k] for k in ("state_dict", "hyper_parameters", "cfg_yaml") if k in ck}
    if "state_dict" not in keep:
        raise SystemExit("ABORT: no state_dict in checkpoint; keys=%s" % list(ck)[:12])
    print("   kept %s | dropped %s" % (sorted(keep), sorted(set(ck) - set(keep))),
          flush=True)
    # 11.55 -> 6.72 GB was not enough; it still died during model setup. The state_dict
    # is fp32, and `Inference.load_model` casts the model to a lower precision via
    # `get_precision_config` immediately after loading anyway -- so store fp16 and halve
    # both the file and the peak. Cast tensor-by-tensor so the fp32 copy is never
    # materialised whole.
    sd = keep.get("state_dict", {})
    n16 = 0
    for k in list(sd.keys()):
        t = sd[k]
        if hasattr(t, "dtype") and t.dtype == torch.float32:
            sd[k] = t.half()
            n16 += 1
    print("   cast %d/%d state_dict tensors to fp16" % (n16, len(sd)), flush=True)
    import psutil  # noqa: E402
    print("   RAM available before save: %.1f GB"
          % (psutil.virtual_memory().available / 1e9), flush=True)
    torch.save(keep, SLIM)
    del ck, keep
    import gc
    gc.collect()
    print("   %.2f GB -> %.2f GB" % (os.path.getsize(BIG) / 1e9,
                                     os.path.getsize(SLIM) / 1e9), flush=True)
    os.remove(BIG)

# ---------------------------------------------------------------- build inputs
import anndata as ad     # noqa: E402
import numpy as np       # noqa: E402
import pandas as pd      # noqa: E402
import pickle            # noqa: E402
import scipy.sparse as sp  # noqa: E402
import csv               # noqa: E402

v = pickle.load(open(os.path.join(TX, "var_dims.pkl"), "rb"))
MG = [str(x) for x in v["gene_names"]]
bm = pickle.load(open(os.path.join(TX, "batch_onehot_map.pkl"), "rb"))
BATCH = str(list(bm.keys())[0])
T = [r[0] for r in csv.reader(open(os.path.join(CTL, "pert_counts.csv")))][1:]
T = [t for t in T if t]
SG = [l.strip() for l in open(os.path.join(CTL, "gene_names.csv")) if l.strip()][1:]
print("\n[%s] model genes %d | submission genes %d | targets %d | batch %s"
      % (el(), len(MG), len(SG), len(T), BATCH), flush=True)

rng = np.random.RandomState(0)
out = {}
for ctx in ["A", "B", "C"]:
    print("\n=== context %s (token %s) ===" % (ctx, CT_FOR[ctx]), flush=True)
    a = ad.read_h5ad(os.path.join(CTL, "context_%s.h5ad" % ctx))
    pos = {g: i for i, g in enumerate([str(x) for x in a.var_names])}
    take = np.array([pos.get(g, -1) for g in MG])
    have = take >= 0
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    sub = X[:, take[have]]
    if not have.all():
        full = sp.lil_matrix((X.shape[0], len(MG)), dtype=X.dtype)
        full[:, np.nonzero(have)[0]] = sub
        sub = full.tocsr()
    print("  controls %s -> model axis, %d/%d genes present"
          % (a.shape, int(have.sum()), len(MG)), flush=True)

    n = sub.shape[0]
    blocks, labs = [], []
    ci = rng.choice(n, min(NCTRL, n), replace=False)
    blocks.append(sub[ci]); labs += ["non-targeting"] * len(ci)
    for t in T:
        idx = rng.choice(n, min(NCELL, n), replace=False)
        blocks.append(sub[idx]); labs += [t] * len(idx)
    M = sp.vstack(blocks).tocsr()
    obs = pd.DataFrame({"target_gene": pd.Categorical(labs),
                        "cell_type": pd.Categorical([CT_FOR[ctx]] * M.shape[0]),
                        "batch_var": pd.Categorical([BATCH] * M.shape[0])})
    obs.index = ["c%d" % i for i in range(M.shape[0])]
    raw = os.path.join(SCRATCH, "raw_%s.h5ad" % ctx)
    emb = os.path.join(SCRATCH, "emb_%s.h5ad" % ctx)
    prd = os.path.join(SCRATCH, "pred_%s.h5ad" % ctx)
    ad.AnnData(X=M, obs=obs, var=pd.DataFrame(index=pd.Index(MG))).write_h5ad(raw)
    print("  wrote %s %s" % (raw, M.shape), flush=True)

    # PASS --config. `Inference.load_model` calls torch.load on the FULL checkpoint three
    # times: once at line 103 to recover cfg (skipped when --config is given), once inside
    # load_from_checkpoint, and once at line 138 for protein embeds (already skipped, we
    # pass --protein-embeddings). At 5.33 GB each against 17.7 GB available, dropping one
    # is the difference. SE-600M ships its own config.yaml, which we already downloaded.
    sh(("python -m state emb transform"
        " --checkpoint %s"
        " --config %s"
        " --input %s"
        " --output %s"
        " --embed-key X_state"
        " --protein-embeddings %s/protein_embeddings.pt"
        " --batch-size 8") % (SLIM, os.path.join(SE, "config.yaml"), raw, emb, SE))
    try:
        os.remove(raw)
    except OSError:
        pass
    sh("python -m state tx infer --model-dir %s --adata %s --output %s"
       " --embed-key X_state --pert-col target_gene --celltype-col cell_type"
       " --batch-col batch_var --control-pert non-targeting" % (TX, emb, prd))

    pa = ad.read_h5ad(prd)
    P = np.asarray(pa.X.todense()) if hasattr(pa.X, "todense") else np.asarray(pa.X)
    lab = np.asarray(pa.obs["target_gene"]).astype(str)
    base = P[lab == "non-targeting"].mean(0)
    D = np.zeros((len(T), P.shape[1]), np.float32)
    ok = 0
    for i, t in enumerate(T):
        m = lab == t
        if m.any():
            D[i] = P[m].mean(0) - base
            ok += 1
    # lift onto the submission axis; genes the model does not emit stay at zero
    pos2 = {g: i for i, g in enumerate(MG)}
    tk = np.array([pos2.get(g, -1) for g in SG]); hv = tk >= 0
    F = np.zeros((len(T), len(SG)), np.float32)
    F[:, np.nonzero(hv)[0]] = D[:, tk[hv]]
    nz = np.linalg.norm(F, axis=1)
    U = F / np.maximum(nz[:, None], 1e-30)
    S = U @ U.T
    print("  %d/%d targets | mean||d|| %.4f | between-row %.4f | genes %d/%d"
          % (ok, len(T), nz.mean(), S[~np.eye(len(T), dtype=bool)].mean(),
             int(hv.sum()), len(SG)), flush=True)
    out[ctx] = F

np.savez_compressed("/kaggle/working/pred26_state.npz",
                    targets=np.array(T), genes=np.array(SG), **out)
print("\n[%s] wrote /kaggle/working/pred26_state.npz  (%.1f MB)"
      % (el(), os.path.getsize("/kaggle/working/pred26_state.npz") / 1e6), flush=True)
print("[%s] DONE" % el(), flush=True)
