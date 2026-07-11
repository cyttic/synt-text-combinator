#!/usr/bin/env python3
"""Build the short-crop VERIFIER dataset: single letters + ALL corpus bigrams x 57 fonts.

Labels are perfect by construction (font rendering). Augmentations mirror how verification
stresses images: rotation (+-10 deg; the 3-view check rotates +-7), font size, stroke
thickness (morphological dilate/erode ~ pen weight), mild blur/noise.

    python 06_make_verifier_ds.py            # ~77k crops -> out/verifier_ds/
"""
import argparse, collections, glob, os, random, re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

FONTS_DIR = "/mnt/ssd2/cyttic/projects/fontsVisualizer/fonts"
SENTS = "/mnt/ssd2/cyttic/projects/TrOCR_Hebrew/sentences_modern.txt"
LETTERS = "אבגדהוזחטיכךלמםנןסעפףצץקרשת"
HEB = re.compile(r"[א-ת]+")


def render(font_path, text, pt, pad=10):
    try:
        font = ImageFont.truetype(font_path, pt)
    except Exception:
        return None
    tmp = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    bbox = tmp.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return None
    canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), "white")
    ImageDraw.Draw(canvas).text((pad - bbox[0], pad - bbox[1]), text, font=font, fill="black")
    return canvas


def augment(img, rng):
    a = np.array(img.convert("L"))
    # stroke weight: dilate or erode ink by 1px
    r = rng.random()
    if r < 0.35:
        a = cv2.erode(a, np.ones((2, 2), np.uint8))     # ink is dark -> erode thickens
    elif r < 0.6:
        a = cv2.dilate(a, np.ones((2, 2), np.uint8))    # thins
    img = Image.fromarray(a).convert("RGB")
    img = img.rotate(rng.uniform(-10, 10), expand=True, fillcolor="white",
                     resample=Image.BICUBIC)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.1)))
    if rng.random() < 0.3:
        a = np.array(img).astype(np.int16)
        a += rng.integers(-18, 18, a.shape, dtype=np.int16)
        img = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    ink = ImageOps.invert(img.convert("L")).getbbox()
    return img.crop(ink) if ink else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/verifier_ds")
    ap.add_argument("--bigram-variants", type=int, default=2, dest="bg_var")
    ap.add_argument("--letter-variants", type=int, default=6, dest="lt_var")
    ap.add_argument("--test-frac", type=float, default=0.04, dest="test_frac")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed); random.seed(args.seed)

    freq = collections.Counter()
    with open(SENTS, encoding="utf-8") as f:
        for line in f:
            for w in HEB.findall(line):
                for a, b in zip(w, w[1:]):
                    freq[a + b] += 1
    bigrams = sorted(freq)
    fonts = sorted(sum((glob.glob(os.path.join(FONTS_DIR, e)) for e in
                        ("*.ttf", "*.otf", "*.TTF", "*.OTF")), []))
    print(f"{len(bigrams)} bigrams + {len(LETTERS)} letters x {len(fonts)} fonts "
          f"(x{args.bg_var}/x{args.lt_var} variants)")

    imdir = os.path.join(args.out, "images"); os.makedirs(imdir, exist_ok=True)
    jobs = [(t, args.bg_var) for t in bigrams] + [(t, args.lt_var) for t in LETTERS]
    rows, n = [], 0
    for text, nvar in jobs:
        for fp in fonts:
            for _ in range(nvar):
                img = render(fp, text, pt=int(rng.integers(60, 111)))
                if img is None:
                    continue
                img = augment(img, rng)
                if img.width < 4 or img.height < 4:
                    continue
                name = f"{n:06d}.png"
                img.save(os.path.join(imdir, name))
                rows.append(f"{name}\t{text}")
                n += 1
        if len(rows) % 10000 < nvar * len(fonts):
            print(f"  {len(rows):,} images...", flush=True)

    random.shuffle(rows)
    n_test = int(len(rows) * args.test_frac)
    for split, chunk in (("test", rows[:n_test]), ("train", rows[n_test:])):
        with open(os.path.join(args.out, f"labels_{split}.tsv"), "w", encoding="utf-8") as f:
            f.write("\n".join(chunk) + "\n")
    print(f"DONE: {len(rows):,} images -> {args.out} "
          f"(train {len(rows) - n_test:,} / test {n_test:,})")


if __name__ == "__main__":
    main()
