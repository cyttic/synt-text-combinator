#!/usr/bin/env python3
"""Aligned letter grid: rows = 57 fonts, columns = each Hebrew letter, one glyph per cell.
Scan a column to spot the font whose version of that letter is bad.

    python font_letter_grid.py
"""
import argparse, glob, os
from PIL import Image, ImageDraw, ImageFont

ALPHABET = "אבגדהוזחטיכךלמםנןסעפףצץקרשת"
PT = 40
CELL = 58
LABEL_W = 150
HDR_H = 46


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", default="/mnt/ssd2/cyttic/projects/fontsVisualizer/fonts")
    ap.add_argument("--out", default="out/font_letter_grid.png")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    args = ap.parse_args()
    FONTS_DIR, OUT = args.fonts, args.out
    fonts = sorted(sum((glob.glob(os.path.join(FONTS_DIR, e)) for e in
                        ("*.ttf", "*.otf", "*.TTF", "*.OTF")), []))
    all_n = len(fonts)
    fonts = fonts[args.offset:args.offset + args.count] if args.count else fonts[args.offset:]
    base_idx = args.offset
    print(f"({all_n} fonts total; showing {base_idx}..{base_idx+len(fonts)-1})")
    letters = list(ALPHABET)
    ncol = len(letters)
    W = LABEL_W + ncol * CELL
    H = HDR_H + len(fonts) * CELL
    sheet = Image.new("RGB", (W, H), (252, 252, 252))
    d = ImageDraw.Draw(sheet)
    small = ImageFont.load_default()

    # header row: the letter for each column (use a clean known font if present, else first)
    hdr_font = ImageFont.truetype(fonts[0], 30)
    for j, ch in enumerate(letters):
        x = LABEL_W + j * CELL
        bb = d.textbbox((0, 0), ch, font=hdr_font)
        d.text((x + (CELL - (bb[2] - bb[0])) // 2 - bb[0], 6), ch, font=hdr_font, fill=(20, 20, 150))
    d.line([(0, HDR_H), (W, HDR_H)], fill=(180, 180, 180))

    for i, fp in enumerate(fonts):
        y = HDR_H + i * CELL
        d.text((6, y + CELL // 2 - 4), f"{base_idx + i:03d} {os.path.basename(fp)[:17]}",
               fill=(90, 90, 90), font=small)
        d.line([(0, y), (W, y)], fill=(230, 230, 230))
        try:
            f = ImageFont.truetype(fp, PT)
        except Exception:
            continue
        for j, ch in enumerate(letters):
            x = LABEL_W + j * CELL
            bb = d.textbbox((0, 0), ch, font=f)
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            d.text((x + (CELL - w) // 2 - bb[0], y + (CELL - h) // 2 - bb[1]),
                   ch, font=f, fill=(0, 0, 0))
    for j in range(ncol + 1):
        x = LABEL_W + j * CELL
        d.line([(x, HDR_H), (x, H)], fill=(235, 235, 235))
    os.makedirs("out", exist_ok=True)
    sheet.save(OUT)
    print(f"wrote {OUT}  ({len(fonts)} fonts x {ncol} letters, {W}x{H})")


if __name__ == "__main__":
    main()
