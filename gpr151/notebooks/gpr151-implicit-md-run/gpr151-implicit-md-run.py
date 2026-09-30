"""
kaggle_run_all.py -- Kaggle GPU kernel entrypoint for GPR151/GPR88 vs
opioid-receptor GBn2 implicit-solvent MD screening (7 conditions).

Installs a CUDA-enabled OpenMM via conda-forge (avoiding the pip-OpenMM
CPU/OpenCL-only trap and the libstdc++ ABI clash with Kaggle's system
Python), then runs all_conditions_driver.py -- which in turn calls
run_implicit_md.py per condition -- entirely inside the conda env via
subprocess, keeping this launcher process on system Python throughout.

Push as a Kaggle "script" kernel with:
  - GPU enabled (T4 x1 or x2)
  - Internet enabled (needed for the conda/miniforge install)
  - Dataset "ericdu847/gpr151-implicit-md-systems" attached as input
Kaggle auto-saves everything written under the kernel's working directory
(including ./out/<condition>/*) as kernel output when the run finishes.
"""
import subprocess, sys, os, glob, time, re, shutil

def sh(cmd, **kw):
    print("+", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"command failed (exit {r.returncode}): {cmd}")
    return r

t0 = time.time()

print("=== GPU / driver check ===", flush=True)
smi = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
print(smi.stdout, smi.stderr, flush=True)

CONDA_PREFIX = "/opt/miniforge"
if not os.path.exists(f"{CONDA_PREFIX}/bin/conda"):
    print("=== Installing miniforge ===", flush=True)
    sh("wget -q https://github.com/conda-forge/miniforge/releases/latest/download/"
       "Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh")
    sh(f"bash /tmp/miniforge.sh -b -p {CONDA_PREFIX}")

conda = f"{CONDA_PREFIX}/bin/conda"

# driver-aware CUDA toolkit selection (Issue #3 in project history: cudatoolkit
# 11.8 installed against a CUDA 12.x driver silently falls back to CPU speed)
m = re.search(r"CUDA Version:\s*([\d.]+)", smi.stdout)
cuda_ver = "12.0"
if m:
    driver_cuda = float(m.group(1))
    cuda_ver = "12.0" if driver_cuda >= 12.0 else "11.8"
print(f"Detected driver CUDA {m.group(1) if m else '?'} -> using cuda-version={cuda_ver}", flush=True)

ENV_NAME = "openmm_env"
env_python = f"{CONDA_PREFIX}/envs/{ENV_NAME}/bin/python"
if not os.path.exists(env_python):
    print("=== Creating conda env with CUDA-enabled OpenMM ===", flush=True)
    sh(f"{conda} create -y -n {ENV_NAME} -c conda-forge python=3.11 openmm "
       f"cuda-version={cuda_ver} mdanalysis numpy")

print("=== Verifying OpenMM platforms ===", flush=True)
check = subprocess.run(
    [env_python, "-c",
     "import openmm as mm; print([mm.Platform.getPlatform(i).getName() "
     "for i in range(mm.Platform.getNumPlatforms())])"],
    capture_output=True, text=True,
)
print("stdout:", check.stdout, flush=True)
print("stderr:", check.stderr, flush=True)
if "CUDA" not in check.stdout:
    print("WARNING: CUDA platform not available -- will run at CPU speed!", flush=True)

INPUT_ROOT = "/kaggle/input"
# Find the directory that actually contains run_implicit_md.py -- Kaggle's
# input layout has varied (sometimes /kaggle/input/<slug>/, sometimes
# nested under /kaggle/input/datasets/<user>/<slug>/), so search a few
# levels deep rather than assume a fixed depth.
DATA_DIR = None
for depth_glob in ["*", "*/*", "*/*/*", "*/*/*/*"]:
    for d in glob.glob(f"{INPUT_ROOT}/{depth_glob}"):
        if (os.path.isdir(d)
                and os.path.exists(os.path.join(d, "run_implicit_md.py"))
                and glob.glob(os.path.join(d, "*__system.xml*"))):
            DATA_DIR = d
            break
    if DATA_DIR:
        break
assert DATA_DIR, f"Could not find run_implicit_md.py + system files under {INPUT_ROOT} (searched up to 4 levels deep)"
print("DATA_DIR:", DATA_DIR, flush=True)
print("contents:", sorted(os.listdir(DATA_DIR)), flush=True)

shutil.copy(os.path.join(DATA_DIR, "run_implicit_md.py"), "run_implicit_md.py")
shutil.copy(os.path.join(DATA_DIR, "all_conditions_driver.py"), "all_conditions_driver.py")

# If a prior partial run's output was attached as a second dataset input
# (e.g. "gpr151-implicit-md-partial-out"), copy its out/ tree in first so
# already-completed conditions are skipped this session.
all_input_dirs = [d for d in glob.glob(f"{INPUT_ROOT}/**", recursive=True) if os.path.isdir(d)]
for d in all_input_dirs:
    if d != DATA_DIR and os.path.isdir(os.path.join(d, "out")):
        print(f"Found prior partial output at {d}/out -- copying in for resume", flush=True)
        shutil.copytree(os.path.join(d, "out"), "out", dirs_exist_ok=True)

run_env = os.environ.copy()
lib_dir = f"{CONDA_PREFIX}/envs/{ENV_NAME}/lib"
run_env["LD_LIBRARY_PATH"] = lib_dir + ":" + run_env.get("LD_LIBRARY_PATH", "")
# Pass the already-resolved dataset dir down explicitly, rather than have
# all_conditions_driver.py re-search /kaggle/input independently (that
# search matched the wrong, shallower directory in an earlier run).
run_env["MD_DATA_DIR"] = DATA_DIR
# Observed: gpr88_vs_mor took ~4.35h wall time for 2ns equil + 10ns production
# on a single T4/P100 (GBn2 implicit solvent, ~10.4k atoms). Kaggle's hard
# session cap is ~9-12h; with 2 GPUs running conditions in parallel (a "wave"
# = up to NUM_GPUS conditions simultaneously), budget for ~2 waves per
# session while leaving a safety margin below the hard cap. If the internal
# budget check is too conservative we simply stop early and resume next
# session (safe); if too generous, Kaggle's own hard kill still won't corrupt
# completed conditions' done.json files (only the in-flight one is lost).
run_env.setdefault("MD_TIME_BUDGET_HOURS", "8.7")

print("=== LAUNCHING all_conditions_driver.py ===", flush=True)
ret = subprocess.run([env_python, "all_conditions_driver.py"], env=run_env)
print(f"driver exit code: {ret.returncode}", flush=True)
print(f"total kernel elapsed: {time.time()-t0:.1f}s", flush=True)
