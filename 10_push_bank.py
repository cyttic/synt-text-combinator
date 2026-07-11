#!/usr/bin/env python3
"""Push a verified bigram bank (out/verified_*) to HF as a standalone dataset.

    /mnt/ssd2/cyttic/ml_env/bin/python 10_push_bank.py                       # out/verified_exp10 -> cyttic/heb-bigram-bank
    /mnt/ssd2/cyttic/ml_env/bin/python 10_push_bank.py --bank out/verified_v2
"""
import argparse, glob, os

from datasets import Dataset, Features, Image as HfImage, Value

ap = argparse.ArgumentParser()
ap.add_argument("--bank", default="out/verified_exp10")
ap.add_argument("--repo", default="cyttic/heb-bigram-bank")
args = ap.parse_args()

rows = []
for d in sorted(glob.glob(os.path.join(args.bank, "[0-9]*_*"))):
    bigram = os.path.basename(d).split("_", 1)[1]
    for p in sorted(glob.glob(os.path.join(d, "s*.png"))):
        style = int(os.path.basename(p)[1:4])
        rows.append((p, bigram, style))
assert rows, f"no verified units in {args.bank}"

ds = Dataset.from_dict(
    {"image": [{"bytes": open(p, "rb").read(), "path": None} for p, _, _ in rows],
     "text": [t for _, t, _ in rows],
     "style_id": [s for _, _, s in rows]},
    features=Features({"image": HfImage(), "text": Value("string"), "style_id": Value("int32")}))
ds.push_to_hub(args.repo, private=True)
print(f"pushed {len(rows)} verified units ({len(set(t for _, t, _ in rows))} bigrams) "
      f"-> https://huggingface.co/datasets/{args.repo}")
