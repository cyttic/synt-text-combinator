#!/usr/bin/env python3
"""Push the verifier dataset to HF so Kaggle can train on it.

Four splits in `cyttic/heb-bigram-verifier` (private):
  train / test      — the 77k augmented font crops (out/verifier_ds)
  poc_font          — the 100 POC font renders (gate-coverage probe)
  diffused_probe    — every verified diffused unit from all out/verified* runs
                      (handwriting-domain probe; NEVER for training)

    /mnt/ssd2/cyttic/ml_env/bin/python 08_push_verifier_ds.py
"""
import glob, os

from datasets import Dataset, Features, Image as HfImage, Value

REPO = "cyttic/heb-bigram-verifier"


def rows_from_tsv(ds_dir, split):
    for line in open(os.path.join(ds_dir, f"labels_{split}.tsv"), encoding="utf-8"):
        if line.strip():
            name, text = line.rstrip("\n").split("\t")
            yield os.path.join(ds_dir, "images", name), text


def rows_poc():
    for line in open("out/font/manifest.tsv", encoding="utf-8"):
        if line.strip():
            name, text = line.split("\t")[:2]
            yield os.path.join("out/font", name), text


def rows_diffused():
    seen = set()
    for d in sorted(glob.glob("out/verified*/[0-9]*_*")):
        text = os.path.basename(d).split("_", 1)[1]
        for p in sorted(glob.glob(os.path.join(d, "s*.png"))):
            if p not in seen:
                seen.add(p)
                yield p, text


def push(split, pairs):
    pairs = list(pairs)
    ds = Dataset.from_dict(
        {"image": [{"bytes": open(p, "rb").read(), "path": None} for p, _ in pairs],
         "text": [t for _, t in pairs]},
        features=Features({"image": HfImage(), "text": Value("string")}))
    ds.push_to_hub(REPO, split=split, private=True)
    print(f"pushed {split}: {len(pairs):,} rows")


if __name__ == "__main__":
    push("train", rows_from_tsv("out/verifier_ds", "train"))
    push("test", rows_from_tsv("out/verifier_ds", "test"))
    push("poc_font", rows_poc())
    push("diffused_probe", rows_diffused())
    print(f"done -> https://huggingface.co/datasets/{REPO}")
