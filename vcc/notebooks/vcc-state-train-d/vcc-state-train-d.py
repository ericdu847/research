"""Train STATE on Arc's own competition support set, with ESM2 perturbation features.

WHY THIS RUN EXISTS.  The 2026 board has four teams (ranks 3-6) whose six per-member
scores agree to three decimals at overall 0.207-0.211, against our 0.1362.  That is one
model being run off the shelf.  It is not a downloaded checkpoint: every released Arc
checkpoint is `pert_rep: onehot` over a fixed vocabulary and reads 0/300 coverage on the
competition's targets.  It is `state tx train` on the support set Arc publishes, whose
`starter.toml` and `ESM2_pert_features.pt` make the perturbation encoder EMBEDDING-based
and therefore able to address genes never seen in training.

That single swap -- one-hot -> ESM2 -- is the whole difference between 0.136 and 0.21.

WHAT THIS BUYS BEYOND THE BASELINE.  Their weakness is ours inverted:

              pds     mse     jac     nmae    fid     reach
  cluster     0.738   0.247  -0.022   0.118  -0.013   0.196
  us (v34)    0.459   0.000  -0.015   0.143  -0.006   0.206

They own direction (pds, mse = C ~ 0.49 vs our 0.32); we own emission (fid, reach), and
`VCC_KNOWLEDGE.md` S5 measures `disp 0.4` at fid +0.117 / jac +0.100 scaled on the
122-target panel.  Their direction under our emitter sums to 1.505/6 = 0.2508.  So this
run is not the destination -- it is the direction half of it.

DISK.  Kaggle gives ~20 GB persisted at /kaggle/working and a larger scratch at
/kaggle/temp.  The zip is 8.7 GB and the members we need total ~29 GB, so everything heavy
lives on scratch and the zip is deleted the moment extraction finishes.  Only the trained
checkpoint and the maps needed to run inference are written to /kaggle/working.

TIME.  `training/default.yaml` ships max_steps 400000, far past a 12 h T4 session.  We cap
the budget and checkpoint every CKPT_EVERY steps into /kaggle/working, so a session that
is cut off still leaves a usable model and the next session can resume from it.

SCHEMA.  The obs column names of the support-set h5 files are detected at run time and
asserted before training starts.  Hardcoding them risks discovering a typo eleven hours in.
"""
import os
import subprocess
import sys
import time

T0 = time.time()

# SELF-LOGGING.  Kaggle's log endpoint returned 0 bytes for the run that mattered, so the
# only record of why v5 died was lost.  Tee everything into /kaggle/working, which comes
# back with `kaggle kernels output` regardless of whether the log API cooperates.  Child
# processes inherit the tee by having their output captured and re-emitted here.
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


def sh(cmd, check=True):
    """Run a child and fold its output into our own log, so a crash inside
    `state tx train` is recoverable from /kaggle/working even if Kaggle's log is empty."""
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


import glob
import threading


def _heartbeat(rundir, every=60):
    def loop():
        while True:
            time.sleep(every)
            try:
                cks = sorted(glob.glob(os.path.join(rundir, "**", "*.ckpt"),
                                       recursive=True))
                mets = sorted(glob.glob(os.path.join(rundir, "**", "metrics.csv"),
                                        recursive=True))
                last = ""
                if mets:
                    with open(mets[-1], "r", errors="replace") as f:
                        rows = [r.rstrip() for r in f if r.strip()]
                    if len(rows) > 1:
                        last = " | " + rows[-1][:200]
                print("[HB %s] ckpts=%d%s%s" % (
                    el(), len(cks),
                    (" newest=" + os.path.basename(cks[-1])) if cks else "",
                    last), flush=True)
            except Exception as e:
                print("[HB %s] %s" % (el(), str(e)[:120]), flush=True)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def el():
    return "%5.1fm" % ((time.time() - T0) / 60)


# ---------------------------------------------------------------- configuration
WORK = "/kaggle/working"


def _scratch():
    """Pick the roomiest writable scratch.  We need ~38 GB peak (8.7 GB zip plus the
    ~29 GB of members we keep) and /kaggle/working is capped at 20 GB, so guessing wrong
    here costs a whole session to a disk-full forty minutes in."""
    best, bfree = None, -1
    for c in ("/kaggle/temp", "/tmp", "/kaggle/working"):
        try:
            os.makedirs(c, exist_ok=True)
            s = os.statvfs(c)
            free = s.f_bavail * s.f_frsize
            print("  scratch candidate %-18s %6.1f GB free" % (c, free / 1e9), flush=True)
            if free > bfree:
                best, bfree = c, free
        except Exception as e:
            print("  scratch candidate %-18s unusable (%s)" % (c, str(e)[:60]), flush=True)
    if bfree < 40e9:
        print("  WARNING: best scratch %s has only %.1f GB; extraction needs ~38 GB"
              % (best, bfree / 1e9), flush=True)
    return best


SCRATCH = _scratch()
SUP = os.path.join(SCRATCH, "competition_support_set")
RUNDIR = os.path.join(WORK, "strun")
URL = ("https://storage.googleapis.com/vcc_data_prod/datasets/state/"
       "competition_support_set.zip")

# members needed for TRAINING only.  The val/test templates are ~20 GB of emission
# scaffolding we do not use here -- inference runs against our own context controls.
NEED = ["ESM2_pert_features.pt", "gene_names.csv", "starter.toml",
        "competition_train.h5", "k562_gwps.h5", "rpe1.h5", "jurkat.h5",
        "k562.h5", "hepg2.h5"]

MAX_STEPS = int(os.environ.get("ST_MAX_STEPS", "8000"))
CKPT_EVERY = int(os.environ.get("ST_CKPT_EVERY", "500"))
BATCH = int(os.environ.get("ST_BATCH", "16"))
CELL_SET = int(os.environ.get("ST_CELL_SET", "64"))

os.makedirs(SCRATCH, exist_ok=True)
os.makedirs(RUNDIR, exist_ok=True)

# ---------------------------------------------------------------- 1. environment
sh("pip -q install arc-state 2>&1 | tail -3")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", check=False)

# ACCELERATOR GUARD.  Kaggle can hand out a P100 (sm_60) against a torch build that needs
# sm_70+, and in that state `torch.cuda.is_available()` returns True while every real CUDA
# op fails -- `kernel/vcc_setnet.py` already lost a run to exactly this.  A silent fallback
# to CPU would burn the whole 12 h slot, and there is nobody awake to reassign the
# accelerator, so probe with an ACTUAL op and die in the first minute if the GPU is
# unusable or is not the T4 this kernel is pinned to.
def _gpu_guard():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("ABORT: no CUDA device. Re-run with the T4 accelerator.")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print("  [guard] devices: %s" % names, flush=True)
    try:
        torch.zeros(1024, 1024, device="cuda").matmul(
            torch.zeros(1024, 1024, device="cuda")).sum().item()
    except Exception as e:
        raise SystemExit("ABORT: CUDA present but unusable (%s). Almost certainly an "
                         "sm_60 P100 against an sm_70+ torch build. Re-run on T4."
                         % str(e)[:160])
    if not any("T4" in n for n in names):
        raise SystemExit("ABORT: expected a Tesla T4, got %s. Re-run on T4 rather than "
                         "training for hours on the wrong accelerator." % names)
    print("  [guard] real CUDA matmul OK on %d T4(s)" % len(names), flush=True)


_gpu_guard()

# ---------------------------------------------------------------- 2. support set
zp = os.path.join(SCRATCH, "css.zip")
if not os.path.exists(os.path.join(SUP, "competition_train.h5")):
    if not os.path.exists(zp):
        print("[%s] downloading support set (8.7 GB)" % el(), flush=True)
        sh("curl -sL --retry 3 -C - -o %s %s" % (zp, URL))
    import zipfile
    z = zipfile.ZipFile(zp)
    for i in z.infolist():
        base = os.path.basename(i.filename)
        if i.is_dir() or base not in NEED:
            continue
        dst = os.path.join(SUP, base)
        if os.path.exists(dst) and os.path.getsize(dst) == i.file_size:
            continue
        t = time.time()
        os.makedirs(SUP, exist_ok=True)
        with z.open(i) as src, open(dst, "wb") as out:
            while True:
                b = src.read(1 << 24)
                if not b:
                    break
                out.write(b)
        print("[%s] %-32s %6.2f GB %5.0fs" % (el(), base, i.file_size / 1e9,
                                              time.time() - t), flush=True)
    z.close()
    os.remove(zp)
sh("df -h %s | tail -1" % SCRATCH, check=False)
sh("ls -la %s" % SUP, check=False)

# ---------------------------------------------------------------- 3. schema probe
import h5py  # noqa: E402
import numpy as np  # noqa: E402


def obs_keys(path):
    with h5py.File(path, "r") as f:
        if "obs" not in f:
            return []
        return list(f["obs"].keys())


def cats(path, col, n=12):
    with h5py.File(path, "r") as f:
        g = f["obs"][col]
        if isinstance(g, h5py.Group) and "categories" in g:
            c = g["categories"][:]
            return [x.decode() if isinstance(x, bytes) else str(x) for x in c[:n]], len(c)
    return None, None


train_h5 = os.path.join(SUP, "competition_train.h5")
ks = obs_keys(train_h5)
print("\n[%s] obs columns in competition_train.h5:\n  %s" % (el(), ks), flush=True)
for k in ks:
    c, n = cats(train_h5, k)
    if c is not None:
        print("   %-22s categorical n=%-6d ex=%s" % (k, n, c[:6]), flush=True)


def pick(cands, keys, what):
    for c in cands:
        if c in keys:
            return c
    raise SystemExit("could not find a %s column among %s" % (what, keys))


PERT_COL = pick(["target_gene", "gene", "perturbation", "condition"], ks, "perturbation")
CT_COL = pick(["cell_type", "cell_line", "celltype"], ks, "cell type")
BATCH_COL = pick(["batch_var", "gem_group", "batch", "gene_batch"], ks, "batch")
allc, _ = cats(train_h5, PERT_COL, n=100000)
CTRL = "non-targeting"
for cand in ["non-targeting", "non_targeting", "control", "NTC", "DMSO_TF"]:
    if allc and cand in allc:
        CTRL = cand
        break
print("\n[%s] pert_col=%s  cell_type_key=%s  batch_col=%s  control_pert=%s"
      % (el(), PERT_COL, CT_COL, BATCH_COL, CTRL), flush=True)

# the ESM2 map must cover the perturbations we will deploy on, not just the ones we train
# on -- that is the entire reason this run is not a one-hot model.
import torch  # noqa: E402
pf = os.path.join(SUP, "ESM2_pert_features.pt")
feats = torch.load(pf, map_location="cpu", weights_only=False)
print("[%s] ESM2 features: %d perturbations x %d dims"
      % (el(), len(feats), next(iter(feats.values())).shape[-1]), flush=True)
if allc:
    miss = [p for p in allc if p not in feats and p != CTRL]
    print("   training perts missing from ESM2: %d / %d" % (len(miss), len(allc)),
          flush=True)

# ---------------------------------------------------------------- 3b. log1p consistency
# `cell_load` REFUSES output_space='all' when some datasets carry uns/log1p and others do
# not (perturbation_dataloader.py:541).  Arc's own starter.toml lists all six files, so the
# published starter hits this too -- which means the resolution is a data fact, not a config
# trick, and guessing it would assert something false about the expression space.  So:
# measure every file, classify it, and act only on what the numbers say.
#
# A raw-count matrix is integer-valued with a large maximum.  A log1p matrix is not
# integer-valued and lives around 0-10.  If every file really is log1p, the ones missing
# uns/log1p are mislabelled and the marker can be added honestly.  If they genuinely differ
# they are in DIFFERENT SPACES and must not be pooled -- we then train on the largest
# consistent group, preferring the one holding competition_train.h5, because that is H1 and
# H1 is the nearest context we have to the destination.
def probe(path):
    with h5py.File(path, "r") as f:
        uns = f.get("uns")
        has = uns is not None and "log1p" in uns
        X = f["X"]
        d = X["data"] if isinstance(X, h5py.Group) else X
        s = np.asarray(d[: min(200000, d.shape[0])])
    return has, float(s.max()), float(s.mean()), float(np.mean(s == np.round(s)))


H5 = sorted(f for f in os.listdir(SUP) if f.endswith(".h5"))
print("\n[%s] log1p probe" % el(), flush=True)
info = {}
for fn in H5:
    has, mx, mu, fi = probe(os.path.join(SUP, fn))
    looks_log1p = (fi < 0.99) or (mx < 20.0)
    info[fn] = (has, looks_log1p)
    print("   %-24s uns/log1p=%-5s max=%9.2f mean=%7.4f integer=%.3f  -> %s"
          % (fn, has, mx, mu, fi, "log1p" if looks_log1p else "RAW COUNTS"), flush=True)

if all(v[1] for v in info.values()):
    for fn, (has, _) in info.items():
        if not has:
            with h5py.File(os.path.join(SUP, fn), "a") as f:
                f.require_group("uns").create_group("log1p")
            print("   + added uns/log1p to %s (verified log1p by value distribution)" % fn,
                  flush=True)
    USE_H5 = H5
else:
    want = info.get("competition_train.h5", (False, True))[1]
    USE_H5 = [fn for fn, (_, lg) in info.items() if lg == want]
    print("   MIXED SPACES -- training on the %s group only: %s"
          % ("log1p" if want else "raw-count", USE_H5), flush=True)
    for fn in USE_H5:
        if not info[fn][0] and want:
            with h5py.File(os.path.join(SUP, fn), "a") as f:
                f.require_group("uns").create_group("log1p")
        if info[fn][0] and not want:
            with h5py.File(os.path.join(SUP, fn), "a") as f:
                del f["uns"]["log1p"]
print("[%s] training on %d files: %s" % (el(), len(USE_H5), USE_H5), flush=True)

# ---------------------------------------------------------------- 4. TOML
toml = os.path.join(SCRATCH, "vcc.toml")
# cell_load printed "Dataset path does not exist" for starter.toml's brace form, so point it
# at a DIRECTORY holding exactly the files we mean to train on.  Hard links cost nothing and
# leave the originals intact for the inference step.
DSDIR = os.path.join(SCRATCH, "ds")
os.makedirs(DSDIR, exist_ok=True)
for fn in USE_H5:
    lp = os.path.join(DSDIR, fn)
    if not os.path.exists(lp):
        try:
            os.link(os.path.join(SUP, fn), lp)
        except OSError:
            os.symlink(os.path.join(SUP, fn), lp)
with open(toml, "w") as f:
    f.write("[datasets]\nreplogle_h1 = \"%s\"\n\n" % DSDIR)
    f.write("[training]\nreplogle_h1 = \"train\"\n\n")
    # A whole cell type is held out, not a perturbation split: the deployment regime is a
    # context with zero perturbations measured in it, so validation must be zeroshot.
    if "hepg2.h5" in USE_H5:
        f.write("[zeroshot]\n\"replogle_h1.hepg2\" = \"test\"\n\n")
    else:
        f.write("[zeroshot]\n\n")
    f.write("[fewshot]\n")
print("\n[%s] toml:\n%s" % (el(), open(toml).read()), flush=True)

# ---------------------------------------------------------------- 5. train
ov = [
    "data.kwargs.toml_config_path=%s" % toml,
    "data.kwargs.perturbation_features_file=%s" % pf,
    "data.kwargs.embed_key=null",
    "data.kwargs.output_space=all",
    "data.kwargs.pert_col=%s" % PERT_COL,
    "data.kwargs.cell_type_key=%s" % CT_COL,
    "data.kwargs.batch_col=%s" % BATCH_COL,
    "data.kwargs.control_pert=%s" % CTRL,
    "data.kwargs.num_workers=2",
    "data.kwargs.basal_mapping_strategy=random",
    "data.kwargs.should_yield_control_cells=true",
    "model=state_sm",
    "model.kwargs.cell_set_len=%d" % CELL_SET,
    "model.kwargs.transformer_backbone_kwargs.max_position_embeddings=%d" % CELL_SET,
    "training.batch_size=%d" % BATCH,
    "training.max_steps=%d" % MAX_STEPS,
    "training.ckpt_every_n_steps=%d" % CKPT_EVERY,
    "training.val_freq=%d" % CKPT_EVERY,
    "training.wandb_track=false",
    "+trainer.enable_progress_bar=false",
    "training.devices=1",
    "output_dir=%s" % RUNDIR,
    "name=vccstate",
]
_heartbeat(RUNDIR, int(os.environ.get("ST_HB", "60")))
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
sh("python -m state tx train " + " ".join("'%s'" % o for o in ov))

# ---------------------------------------------------------------- 6. keep results
print("\n[%s] run dir contents:" % el(), flush=True)
sh("find %s -maxdepth 3 -printf '%%10s  %%p\\n' | sort -rn | head -40" % RUNDIR,
   check=False)
sh("du -sh %s" % RUNDIR, check=False)
print("\n[%s] DONE" % el(), flush=True)
