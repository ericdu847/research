#!/usr/bin/env python3
"""Pull notebooks and dataset metadata from several Kaggle accounts into one repo.

Credentials come from environment variables named KAGGLE_ACCOUNT_<username>. Each
one holds the full contents of a legacy kaggle.json file, a bare 32-character
legacy key (the username is then taken from the variable name), or a new-style
API token.

Each account runs in its own `kaggle` CLI subprocess with an isolated config
dir, because the kaggle package authenticates once, at import time.

Routing: the first regex in ROUTES that matches a notebook's slug or title picks
its project folder. Anything unmatched lands in unsorted/<owner>/.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ROUTES: list[tuple[str, str]] = [
    (r"gpr151|gpcr|dock|openmm|psoralen|q8tdv0", "gpr151"),
    (r"vcc|virtual[-_ ]?cell|chorus|perturb|crispr", "vcc"),
    (r"nano|e[-_]?prop|cortex|chatbot", "nano"),
]
PAGE_SIZE = 100
MAX_PAGES = 50


def route(ref: str, title: str) -> str:
    text = f"{ref} {title}".lower()
    for pattern, folder in ROUTES:
        if re.search(pattern, text):
            return folder
    return f"unsorted/{ref.split('/')[0]}"


def account_env(name: str, secret: str, cfg_dir: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("KAGGLE_ACCOUNT_", "KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"))}
    env["KAGGLE_CONFIG_DIR"] = cfg_dir
    secret = secret.strip()
    if secret.startswith("{"):
        creds = json.loads(secret)
        env["KAGGLE_USERNAME"], env["KAGGLE_KEY"] = creds["username"], creds["key"]
    elif re.fullmatch(r"[0-9a-f]{32}", secret):
        env["KAGGLE_USERNAME"], env["KAGGLE_KEY"] = name, secret
    else:
        env["KAGGLE_API_TOKEN"] = secret
    return env


def kaggle(env: dict[str, str], *args: str) -> str:
    r = subprocess.run(["kaggle", *args], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def list_mine(env: dict[str, str], kind: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for page in range(1, MAX_PAGES + 1):
        out = kaggle(env, kind, "list", "--mine", "--csv", "--page-size", str(PAGE_SIZE), "-p", str(page))
        start = out.find("ref,")
        if start < 0:
            break  # "No kernels found" / "No datasets found"
        batch = list(csv.DictReader(io.StringIO(out[start:])))
        rows += batch
        if len(batch) < PAGE_SIZE:
            break
    return rows


def sync_account(name: str, secret: str, manifest: list[dict]) -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as cfg:
        env = account_env(name, secret, cfg)
        try:
            kernels = list_mine(env, "kernels")
            datasets = list_mine(env, "datasets")
        except RuntimeError as e:
            print(f"[{name}] listing failed: {e}", file=sys.stderr)
            return 1
        print(f"[{name}] {len(kernels)} notebooks, {len(datasets)} datasets")

        for k in kernels:
            ref, title = k["ref"], k.get("title", "")
            owner, slug = ref.split("/", 1)
            dest = ROOT / route(ref, title) / "notebooks" / slug
            tmp = Path(tempfile.mkdtemp())
            try:
                kaggle(env, "kernels", "pull", ref, "-p", str(tmp), "-m")
                shutil.rmtree(dest, ignore_errors=True)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(tmp), dest)
                manifest.append({"type": "notebook", "ref": ref, "title": title,
                                 "account": name, "path": str(dest.relative_to(ROOT))})
            except RuntimeError as e:
                failures += 1
                print(f"[{name}] pull {ref} failed: {e}", file=sys.stderr)
                shutil.rmtree(tmp, ignore_errors=True)

        for d in datasets:
            ref, title = d["ref"], d.get("title", "")
            slug = ref.split("/", 1)[1]
            dest = ROOT / route(ref, title) / "datasets" / slug
            dest.mkdir(parents=True, exist_ok=True)
            try:
                kaggle(env, "datasets", "metadata", ref, "-p", str(dest))
                manifest.append({"type": "dataset", "ref": ref, "title": title,
                                 "account": name, "path": str(dest.relative_to(ROOT)),
                                 "url": f"https://www.kaggle.com/datasets/{ref}"})
            except RuntimeError as e:
                failures += 1
                print(f"[{name}] metadata {ref} failed: {e}", file=sys.stderr)
    return failures


def main() -> int:
    accounts = sorted((k, v) for k, v in os.environ.items() if k.startswith("KAGGLE_ACCOUNT_") and v.strip())
    if not accounts:
        print("No KAGGLE_ACCOUNT_* secrets set.", file=sys.stderr)
        return 1
    manifest: list[dict] = []
    failures = sum(sync_account(name.removeprefix("KAGGLE_ACCOUNT_"), secret, manifest)
                   for name, secret in accounts)
    manifest.sort(key=lambda m: (m["path"], m["ref"]))
    (ROOT / "kaggle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Synced {len(manifest)} items, {failures} failures.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
