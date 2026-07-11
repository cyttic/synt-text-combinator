#!/usr/bin/env python3
"""Render each bigram with a random curated Hebrew font -> tight ink crops.

Pillow in ml_env is built with libraqm: feed LOGICAL order, it handles RTL itself
(same convention as test-diff-pen/font_validation.py — do NOT pre-reverse).

    python 02_render_fonts.py
"""
import argparse, glob, os, random
from PIL import Image, ImageDraw, ImageFont, ImageOps

FONTS_DIR = "/mnt/ssd2/cyttic/projects/fontsVisualizer/fonts"


def render(font_path, text, pt=90, pad=10):
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
    ink = ImageOps.invert(canvas.convert("L")).getbbox()
    return canvas.crop(ink) if ink else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bigrams", default="out/bigrams.txt")
    ap.add_argument("--fonts", default=FONTS_DIR)
    ap.add_argument("--out", default="out/font")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    fonts = sorted(sum((glob.glob(os.path.join(args.fonts, e)) for e in
                        ("*.ttf", "*.otf", "*.TTF", "*.OTF")), []))
    assert fonts, f"no fonts in {args.fonts}"
    bigrams = [l.strip() for l in open(args.bigrams, encoding="utf-8") if l.strip()]
    os.makedirs(args.out, exist_ok=True)

    manifest = []
    for i, bg in enumerate(bigrams):
        img, tries = None, 0
        while img is None and tries < 10:  # some fonts miss glyphs -> retry another
            fp = random.choice(fonts)
            img = render(fp, bg)
            tries += 1
        if img is None:
            print(f"SKIP {bg}: no font renders it")
            continue
        name = f"{i:03d}_{bg}.png"
        img.save(os.path.join(args.out, name))
        manifest.append(f"{name}\t{bg}\t{os.path.basename(fp)}")
    with open(os.path.join(args.out, "manifest.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"rendered {len(manifest)}/{len(bigrams)} bigrams -> {args.out}")


if __name__ == "__main__":
    main()
