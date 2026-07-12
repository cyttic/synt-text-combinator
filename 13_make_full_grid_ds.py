#!/usr/bin/env python3
"""Full systematic bigram+letter grid rendered with all 57 fonts.

Units = every valid Hebrew bigram (22 base-letter firsts x 27 all seconds = 594) +
27 single letters = 621 units. x 57 fonts x --variants. Labels perfect by construction.

    python 13_make_full_grid_ds.py                       # 1 variant  -> 35,397 imgs
    python 13_make_full_grid_ds.py --variants 3          # 3 variants -> 106,191 imgs
    python 13_make_full_grid_ds.py --push cyttic/heb-bigram-grid
"""
import argparse, glob, itertools, os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

FONTS_DIR = "/mnt/ssd2/cyttic/projects/fontsVisualizer/fonts"
BASE   = "אבגדהוזחטיכלמנסעפצקרשת"   # 22, may start a bigram
FINALS = "ךםןףץ"                    # 5, word-final only -> never first
ALL27  = BASE + FINALS


def units():
    bigrams = ["".join(p) for p in itertools.product(BASE, ALL27)]   # 22*27 = 594
    letters = list(ALL27)                                            # 27
    return bigrams + letters                                         # 621


def render(font_path, text, pt, pad=10):
    try:
        font = ImageFont.truetype(font_path, pt)
    except Exception:
        return None
    tmp = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    bb = tmp.textbbox((0, 0), text, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    if w <= 0 or h <= 0:
        return None
    c = Image.new("RGB", (w + 2 * pad, h + 2 * pad), "white")
    ImageDraw.Draw(c).text((pad - bb[0], pad - bb[1]), text, font=font, fill="black")
    return c


def augment(img, rng):
    a = np.array(img.convert("L"))
    r = rng.random()
    if r < 0.35:   a = cv2.erode(a, np.ones((2, 2), np.uint8))
    elif r < 0.6:  a = cv2.dilate(a, np.ones((2, 2), np.uint8))
    img = Image.fromarray(a).convert("RGB")
    img = img.rotate(rng.uniform(-10, 10), expand=True, fillcolor="white", resample=Image.BICUBIC)
    if rng.random() < 0.3: img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.1)))
    ink = ImageOps.invert(img.convert("L")).getbbox()
    return img.crop(ink) if ink else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/grid_ds")
    ap.add_argument("--fonts", default=FONTS_DIR)
    ap.add_argument("--variants", type=int, default=1, help="renders per (unit,font)")
    ap.add_argument("--augment", action="store_true", help="apply rotation/stroke/blur augs")
    ap.add_argument("--test-frac", type=float, default=0.04, dest="test_frac")
    ap.add_argument("--push", default=None, help="HF repo id to push to (e.g. cyttic/heb-bigram-grid)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    fonts = sorted(sum((glob.glob(os.path.join(args.fonts, e)) for e in
                        ("*.ttf", "*.otf", "*.TTF", "*.OTF")), []))
    U = units()
    print(f"{len(U)} units (594 bigrams + 27 letters) x {len(fonts)} fonts x {args.variants} "
          f"= {len(U) * len(fonts) * args.variants:,} target images")

    imdir = os.path.join(args.out, "images"); os.makedirs(imdir, exist_ok=True)
    rows, n, miss = [], 0, 0
    for text in U:
        for fp in fonts:
            for _ in range(args.variants):
                img = render(fp, text, pt=int(rng.integers(60, 111)) if args.augment else 90)
                if img is None:
                    miss += 1; continue
                if args.augment:
                    img = augment(img, rng)
                name = f"{n:06d}.png"
                img.save(os.path.join(imdir, name)); rows.append(f"{name}\t{text}"); n += 1
        if len(rows) % 10000 < len(fonts) * args.variants:
            print(f"  {len(rows):,}...", flush=True)

    rng.shuffle(rows)
    n_test = int(len(rows) * args.test_frac)
    for split, chunk in (("test", rows[:n_test]), ("train", rows[n_test:])):
        open(os.path.join(args.out, f"labels_{split}.tsv"), "w", encoding="utf-8").write(
            "\n".join(chunk) + "\n")
    print(f"DONE: {len(rows):,} images ({miss} font-glyph misses) -> {args.out} "
          f"(train {len(rows)-n_test:,} / test {n_test:,})")

    if args.push:
        from datasets import Dataset, Features, Image as HfImage, Value
        for split in ("train", "test"):
            pairs = [l.split("\t") for l in
                     open(os.path.join(args.out, f"labels_{split}.tsv"), encoding="utf-8") if l.strip()]
            ds = Dataset.from_dict(
                {"image": [{"bytes": open(os.path.join(imdir, nm), "rb").read(), "path": None}
                           for nm, _ in pairs],
                 "text": [t for _, t in pairs]},
                features=Features({"image": HfImage(), "text": Value("string")}))
            ds.push_to_hub(args.push, split=split, private=True)
            print(f"pushed {split}: {len(pairs):,}")
        print(f"-> https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()
