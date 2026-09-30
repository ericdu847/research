#!/usr/bin/env python3
"""Pull notebooks and dataset metadata from several Kaggle accounts into one repo.

Credentials come from environment variables named KAGGLE_ACCOUNT_<username>. Each
one holds the full contents of a legacy kaggle.json file, a bare 32-character
legacy key (the username is then taken from the variable name), or a new-style
API token.

Each account runs in its own `kaggle` CLI subprocess with an isolated config
dir, because the kaggle package authenticates once, at import time.

Routing is configured in scripts/routes.json and is content-based: every pulled
notebook's source is scanned for project keywords, plus its slug/title for name
patterns; overrides pin a ref to a folder. Datasets go wherever the notebooks
that attach them went. Each run writes routing_report.md explaining every
decision, and removes the previous copy of anything that moved.
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
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "kaggle_manifest.json"
REPORT = ROOT / "routing_report.md"
CONFIG = json.loads((ROOT / "scripts" / "routes.json").read_text())
PAGE_SIZE = 100
MAX_PAGES = 50
MAX_SOURCE_BYTES = 3_000_000


# ---------------------------------------------------------------- routing

def notebook_source(folder: Path) -> str:
    """Concatenate code and markdown from a pulled notebook folder."""
    parts: list[str] = []
    for f in sorted(folder.iterdir()):
        if f.name == "kernel-metadata.json" or not f.is_file():
            continue
        raw = f.read_bytes()[:MAX_SOURCE_BYTES].decode("utf-8", "ignore")
        if f.suffix == ".ipynb":
            try:
                nb = json.loads(raw)
                for cell in nb.get("cells", []):
                    src = cell.get("source", "")
                    parts.append("".join(src) if isinstance(src, list) else src)
                continue
            except json.JSONDecodeError:
                pass
        parts.append(raw)
    return "\n".join(parts).lower()


def score(ref: str, title: str, source: str) -> tuple[str, str]:
    """Return (folder, reason)."""
    if ref in CONFIG["overrides"]:
        return CONFIG["overrides"][ref], "override"
    name = f"{ref} {title}".lower()
    scores: dict[str, int] = {}
    hits: dict[str, list[str]] = {}
    for project, spec in CONFIG["projects"].items():
        n = [p for p in spec["name_patterns"] if re.search(p, name)]
        k = [w for w in spec["keywords"] if w in source]
        scores[project] = 3 * len(n) + len(k)
        hits[project] = [f"name:{p}" for p in n] + k
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    (best, s1), (_, s2) = ranked[0], ranked[1]
    detail = "; ".join(f"{p}={s} [{', '.join(hits[p][:6])}]" for p, s in ranked if s)
    if s1 >= CONFIG["min_score"] and s1 - s2 >= CONFIG["min_margin"]:
        return best, detail
    return CONFIG["fallback"], detail or "no signal"


# ---------------------------------------------------------------- kaggle

def full_ref(row: dict[str, str], account: str) -> str:
    """Normalise a listed ref to owner/slug (CLI versions differ: URL, slug, or owner/slug)."""
    ref = row["ref"].strip().rstrip("/")
    if ref.startswith("http"):
        ref = "/".join(ref.split("/")[-2:])
    return ref if "/" in ref else f"{account}/{ref}"


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


# ---------------------------------------------------------------- sync

def pull_account(name: str, secret: str, stage: Path, items: list[dict]) -> int:
    """Pull everything for one account into the staging dir; append item records."""
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
            if not k["ref"].strip():
                print(f"[{name}] skipped notebook with no ref (unsaved?): {k.get('title', '')!r}")
                continue
            ref = full_ref(k, name)
            tmp = stage / "notebooks" / ref.replace("/", "__")
            tmp.mkdir(parents=True, exist_ok=True)
            try:
                kaggle(env, "kernels", "pull", ref, "-p", str(tmp), "-m")
            except RuntimeError as e:
                failures += 1
                print(f"[{name}] pull {ref} failed: {e}", file=sys.stderr)
                continue
            meta = json.loads((tmp / "kernel-metadata.json").read_text()) if (tmp / "kernel-metadata.json").exists() else {}
            items.append({"type": "notebook", "ref": ref, "title": k.get("title", ""), "account": name,
                          "stage": tmp, "datasets": meta.get("dataset_sources", [])})

        for d in datasets:
            if not d["ref"].strip():
                continue
            ref = full_ref(d, name)
            tmp = stage / "datasets" / ref.replace("/", "__")
            tmp.mkdir(parents=True, exist_ok=True)
            try:
                kaggle(env, "datasets", "metadata", ref, "-p", str(tmp))
            except RuntimeError as e:
                failures += 1
                print(f"[{name}] metadata {ref} failed: {e}", file=sys.stderr)
                continue
            items.append({"type": "dataset", "ref": ref, "title": d.get("title", ""), "account": name,
                          "stage": tmp, "url": f"https://www.kaggle.com/datasets/{ref}"})
    return failures


def route_all(items: list[dict]) -> None:
    notebooks = [i for i in items if i["type"] == "notebook"]
    for nb in notebooks:
        nb["folder"], nb["reason"] = score(nb["ref"], nb["title"], notebook_source(nb["stage"]))
    users: dict[str, Counter] = {}
    for nb in notebooks:
        for ds in nb["datasets"]:
            users.setdefault(ds, Counter())[nb["folder"]] += 1
    for ds in (i for i in items if i["type"] == "dataset"):
        text = " ".join(p.read_text(errors="ignore") for p in ds["stage"].glob("*.json")).lower()
        folder, reason = score(ds["ref"], ds["title"], text)
        if reason != "override":
            votes = users.get(ds["ref"], Counter())
            votes.pop(CONFIG["fallback"], None)
            if votes:
                folder, n = votes.most_common(1)[0]
                reason = f"attached by {n} {folder} notebook(s)"
        ds["folder"], ds["reason"] = folder, reason


def safe_repo_path(rel: str) -> Path | None:
    p = (ROOT / rel).resolve()
    ok = p.is_relative_to(ROOT) and p.parent.name in ("notebooks", "datasets")
    return p if ok else None


def install(items: list[dict], old: list[dict]) -> None:
    old_paths = {o["ref"] + "|" + o["type"]: o["path"] for o in old}
    for it in items:
        slug = it["ref"].split("/", 1)[1]
        dest = ROOT / it["folder"] / f"{it['type']}s" / slug
        prev = old_paths.get(it["ref"] + "|" + it["type"])
        if prev and (p := safe_repo_path(prev)):
            shutil.rmtree(p, ignore_errors=True)
        shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(it["stage"]), dest)
        it["path"] = str(dest.relative_to(ROOT))
    # prune directories left empty by moves
    for d in sorted((p for p in ROOT.rglob("*") if p.is_dir() and ".git" not in p.parts),
                    key=lambda p: -len(p.parts)):
        if not any(d.iterdir()):
            d.rmdir()


def write_outputs(items: list[dict]) -> None:
    items.sort(key=lambda i: (i["path"], i["ref"]))
    manifest = [{k: i[k] for k in ("type", "ref", "title", "account", "path", "url") if k in i} for i in items]
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    counts = Counter((i["folder"], i["type"]) for i in items)
    lines = ["# Routing report", "",
             "Generated by `scripts/kaggle_sync.py`. To move something, add its ref to `overrides` in "
             "`scripts/routes.json`.", "",
             "| Folder | Notebooks | Datasets |", "|---|---|---|"]
    for f in sorted({i["folder"] for i in items}):
        lines.append(f"| {f} | {counts[(f, 'notebook')]} | {counts[(f, 'dataset')]} |")
    lines += ["", "| Ref | Title | Type | Folder | Why |", "|---|---|---|---|---|"]
    for i in items:
        why = i["reason"].replace("|", "\\|")
        lines.append(f"| {i['ref']} | {i['title'].replace('|', '/')} | {i['type']} | {i['folder']} | {why} |")
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> int:
    accounts = sorted((k, v) for k, v in os.environ.items() if k.startswith("KAGGLE_ACCOUNT_") and v.strip())
    if not accounts:
        print("No KAGGLE_ACCOUNT_* secrets set.", file=sys.stderr)
        return 1
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    items: list[dict] = []
    with tempfile.TemporaryDirectory() as stage_dir:
        stage = Path(stage_dir)
        failures = sum(pull_account(n.removeprefix("KAGGLE_ACCOUNT_"), s, stage, items) for n, s in accounts)
        route_all(items)
        install(items, old)
    # keep manifest entries for anything not re-synced this run (e.g. an account failed)
    synced = {i["ref"] + "|" + i["type"] for i in items}
    for o in old:
        if o["ref"] + "|" + o["type"] not in synced and (ROOT / o["path"]).exists():
            items.append({**o, "folder": o["path"].split(f"/{o['type']}s/")[0], "reason": "kept from previous run"})
    write_outputs(items)
    print(f"Synced {len(synced)} items, {failures} failures.")
    for f, n in sorted(Counter(i["folder"] for i in items).items()):
        print(f"  {f}: {n}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
