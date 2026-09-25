"""Publish each JW-language tier to HuggingFace as it finishes.

    .venv/bin/python upload_jw.py            # once
    .venv/bin/python upload_jw.py --watch    # until the run is done

Uploads out/gemma-jw-tier<N>.jsonl.gz and keeps the dataset README's language
table honest about what is in there.
"""
import argparse, gzip, json, os, shutil, sys, time
from pathlib import Path

HERE = Path("/mnt/volume_d2wey28/projects/shola-translate")
OUT = HERE / "out"
STATE = HERE / "uploaded-jw.json"
CKPT = HERE / "checkpoint-jw.json"
PASS = HERE / "jw-final-langs.tsv"
REPO = "AfriSpeech/shola-machine-translations"
os.environ.setdefault("HF_HOME", "/mnt/volume_d2wey28/hf-cache")

from huggingface_hub import HfApi        # noqa: E402

api = HfApi()


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(d):
    STATE.write_text(json.dumps(d, indent=1, sort_keys=True))


def n_languages():
    if not PASS.exists():
        return 0
    return len([l for l in PASS.read_text(encoding="utf-8").splitlines() if l.strip()])


def tier_finished(tier):
    """A tier is done when every language has a checkpoint entry for it."""
    if not CKPT.exists() or not PASS.exists():
        return False
    done = json.loads(CKPT.read_text())
    codes = [l.split("\t")[0] for l in
             PASS.read_text(encoding="utf-8").splitlines() if l.strip()]
    return all(f"{tier}:{c}" in done for c in codes) and bool(codes)


def upload(tier, done):
    src = OUT / f"gemma-jw-tier{tier}.jsonl"
    if not src.exists():
        return False
    dest = f"gemma-4-31b-it-jw/tier{tier}.jsonl.gz"
    size = src.stat().st_size
    if done.get(dest) == size:
        return False
    gz = src.with_suffix(".jsonl.gz")
    print(f"compressing tier {tier} ({size/1e6:.0f} MB)...", flush=True)
    with open(src, "rb") as fi, gzip.open(gz, "wb", compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
    api.upload_file(path_or_fileobj=str(gz), path_in_repo=dest,
                    repo_id=REPO, repo_type="dataset",
                    commit_message=f"Gemma 4 31B, {n_languages()} more "
                                   f"African languages, tier {tier}")
    done[dest] = size
    save(done)
    print(f"  uploaded {dest}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="upload what exists, finished or not")
    args = ap.parse_args()
    while True:
        done = state()
        for tier in (1, 2, 3, 4):
            if args.force or tier_finished(tier):
                upload(tier, done)
        if not args.watch:
            return
        if all(tier_finished(t) for t in (1, 2, 3, 4)):
            print("all tiers uploaded", flush=True)
            return
        time.sleep(600)


if __name__ == "__main__":
    sys.exit(main())
