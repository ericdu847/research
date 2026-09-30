# %% [code]
"""Reproduce Arc's OFFICIAL Virtual Cell Challenge training run, verbatim.

Ranks 3-6 on the board carry per-member scores identical to three decimals at overall
0.207-0.211.  That is not four teams tuning; `state`'s default `train_seed` is 42, so four
teams running Arc's published recipe on Arc's published data get the SAME WEIGHTS.  The
recipe is the Colab linked from github.com/ArcInstitute/state:

    https://colab.research.google.com/drive/1QKOtYP7bMpdgDJEipDxaJqOchv7oQ-_l

This script is that notebook, unchanged where it matters.  Every earlier attempt of mine
deviated -- `model=state_sm` instead of `model=state`, explicit `output_space`/`embed_key`
overrides the notebook never sets, 2 workers instead of 8, pypi `arc-state` instead of git
main -- and every deviation was mine, not theirs.  All of them are reverted here.

WHAT IS KEPT FROM MY VERSION, AND WHY
  * heartbeat thread -- Kaggle's log endpoint returned 0 bytes on the one run that
    mattered, and Lightning's rich progress bar redraws with carriage returns and emits no
    newlines, so neither stdout nor Kaggle's log is a reliable progress signal.  The
    heartbeat prints checkpoint count plus the metrics.csv header and last row.
  * accelerator guard -- Kaggle can hand out an sm_60 P100 against an sm_70+ torch build,
    where is_available() returns True and only a real op reveals the problem.  Abort in
    minute one rather than burn a 12 h slot.
  * log1p harmonisation -- Arc's own files disagree: competition_train.h5 carries
    uns/log1p and the other five do not, which makes cell_load refuse output_space='all'.
    Verified by value distribution that all six ARE log1p (max 6.7-8.3, integer fraction
    0.000 among nonzeros), so the missing markers are a labelling gap, not a space
    mismatch.
  * devices=1 -- devices=2 deadlocked an earlier run at step 3049 with byte-identical
    metrics across three consecutive heartbeats.

CHECKPOINTS every 2000 rather than the notebook's 20000: a Kaggle session is capped at 12
hours where the notebook assumes an uninterrupted Colab, so intermediate checkpoints are
the difference between a usable model and nothing.
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


def heartbeat(rundir, every=60):
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
                    with open(mets[-1], errors="replace") as f:
                        rows = [r.rstrip() for r in f if r.strip()]
                    if len(rows) > 1:
                        last = " | " + rows[0][:90] + " || " + rows[-1][:120]
                print("[HB %s] ckpts=%d%s%s"
                      % (el(), len(cks),
                         (" newest=" + os.path.basename(cks[-1])) if cks else "", last),
                      flush=True)
            except Exception as e:
                print("[HB %s] %s" % (el(), str(e)[:120]), flush=True)

    threading.Thread(target=loop, daemon=True).start()


WORK = "/kaggle/working"
SCRATCH = "/kaggle/temp"
os.makedirs(SCRATCH, exist_ok=True)
os.chdir(SCRATCH)
SUP = os.path.join(SCRATCH, "competition_support_set")
URL = ("https://storage.googleapis.com/vcc_data_prod/datasets/state/"
       "competition_support_set.zip")
MAX_STEPS = int(os.environ.get("ST_MAX_STEPS", "40000"))
CKPT_EVERY = int(os.environ.get("ST_CKPT_EVERY", "2000"))

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
    print("  [guard] real CUDA matmul OK on %s" % names, flush=True)


sh("pip -q install uv 2>&1 | tail -2", check=False)
if not os.path.isdir(os.path.join(SCRATCH, "state")):
    sh("git clone -q https://github.com/ArcInstitute/state.git")
_gpu_guard()

zp = os.path.join(SCRATCH, "competition_support_set.zip")
if not os.path.exists(os.path.join(SUP, "competition_train.h5")):
    if not os.path.exists(zp):
        print("[%s] downloading support set" % el(), flush=True)
        sh("curl -sL --retry 3 -C - -o %s %s" % (zp, URL))
    import zipfile
    z = zipfile.ZipFile(zp)
    z.extractall(SCRATCH)
    z.close()
    os.remove(zp)
sh("ls -la %s" % SUP, check=False)

import h5py           # noqa: E402
import numpy as np    # noqa: E402

print("\n[%s] log1p probe" % el(), flush=True)
H5 = sorted(f for f in os.listdir(SUP) if f.endswith(".h5"))
info = {}
for fn in H5:
    with h5py.File(os.path.join(SUP, fn), "r") as f:
        uns = f.get("uns")
        has = uns is not None and "log1p" in uns
        X = f["X"]
        d = X["data"] if isinstance(X, h5py.Group) else X
        s = np.asarray(d[: min(200000, d.shape[0])])
    mx, fi = float(s.max()), float(np.mean(s == np.round(s)))
    lg = (fi < 0.99) or (mx < 20.0)
    info[fn] = (has, lg)
    print("   %-24s uns/log1p=%-5s max=%8.2f integer=%.3f -> %s"
          % (fn, has, mx, fi, "log1p" if lg else "RAW"), flush=True)
if all(v[1] for v in info.values()):
    for fn, (has, _) in info.items():
        if not has:
            with h5py.File(os.path.join(SUP, fn), "a") as f:
                f.require_group("uns").create_group("log1p")
            print("   + uns/log1p -> %s" % fn, flush=True)

DS = os.path.join(SCRATCH, "ds")
os.makedirs(DS, exist_ok=True)
for fn in H5:
    lp = os.path.join(DS, fn)
    if not os.path.exists(lp):
        try:
            os.link(os.path.join(SUP, fn), lp)
        except OSError:
            os.symlink(os.path.join(SUP, fn), lp)
TOML = os.path.join(SCRATCH, "vcc.toml")
with open(TOML, "w") as f:
    f.write('[datasets]\nreplogle_h1 = "%s"\n\n[training]\nreplogle_h1 = "train"\n\n'
            '[zeroshot]\n"replogle_h1.hepg2" = "test"\n\n[fewshot]\n' % DS)
print("\n[%s] toml:\n%s" % (el(), open(TOML).read()), flush=True)

RUN = os.path.join(WORK, "competition")
os.makedirs(RUN, exist_ok=True)
heartbeat(os.path.join(RUN, "first_run"))
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

os.chdir(os.path.join(SCRATCH, "state"))
cmd = (
    "uv run state tx train"
    " data.kwargs.toml_config_path=" + TOML +
    " data.kwargs.num_workers=8"
    " data.kwargs.batch_col=batch_var"
    " data.kwargs.pert_col=target_gene"
    " data.kwargs.cell_type_key=cell_type"
    " data.kwargs.control_pert=non-targeting"
    " data.kwargs.perturbation_features_file=" + SUP + "/ESM2_pert_features.pt"
    " training.max_steps=" + str(MAX_STEPS) +
    " training.ckpt_every_n_steps=" + str(CKPT_EVERY) +
    " training.devices=1"
    " training.wandb_track=false"
    " model=state"
    " output_dir=" + RUN +
    " name=first_run"
)
sh(cmd)

print("\n[%s] checkpoints:" % el(), flush=True)
sh("ls -la %s/first_run/checkpoints/ || true" % RUN, check=False)
print("[%s] DONE" % el(), flush=True)
