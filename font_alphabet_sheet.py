#!/usr/bin/env python3
"""Contact sheet: full Hebrew alphabet rendered by each of the 57 fonts, one row per font,
font filename labelled at left. For eyeballing which font has the bad glyph.

    python font_alphabet_sheet.py
"""
import glob, os
from PIL import Image, ImageDraw, ImageFont

FONTS_DIR = "/mnt/ssd2/cyttic/projects/fontsVisualizer/fonts"
ALPHABET = "אבגדהוזחטיכךלמםנןסעפףצץקרשת"   # 22 base + 5 finals (display order)
PT = 46
ROW_H = 74
LABEL_W = 260
PAD = 16
OUT = "out/font_alphabet.png"


def main():
    fonts = sorted(sum((glob.glob(os.path.join(FONTS_DIR, e)) for e in
                        ("*.ttf", "*.otf", "*.TTF", "*.OTF")), []))
    label_font = ImageFont.load_default()
    # measure widest alphabet render to size the canvas
    tmp = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    maxw = 0
    glyph_imgs = []
    for fp in fonts:
        try:
            f = ImageFont.truetype(fp, PT)
            bb = tmp.textbbox((0, 0), ALPHABET, font=f)
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            cell = Image.new("RGB", (w + 2 * PAD, ROW_H), "white")
            ImageDraw.Draw(cell).text((PAD - bb[0], (ROW_H - h) // 2 - bb[1]),
                                      ALPHABET, font=f, fill="black")
            glyph_imgs.append((os.path.basename(fp), cell))
            maxw = max(maxw, cell.width)
        except Exception as e:
            glyph_imgs.append((os.path.basename(fp) + " [ERR]", None))

    W = LABEL_W + maxw + PAD
    H = ROW_H * len(glyph_imgs)
    sheet = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(sheet)
    for i, (name, cell) in enumerate(glyph_imgs):
        y = i * ROW_H
        d.line([(0, y), (W, y)], fill=(225, 225, 225))
        d.text((8, y + ROW_H // 2 - 4), f"{i:02d} {name[:34]}", fill=(90, 90, 90), font=label_font)
        if cell is not None:
            sheet.paste(cell, (LABEL_W + maxw - cell.width, y))   # RTL right-align
    os.makedirs("out", exist_ok=True)
    sheet.save(OUT)
    print(f"wrote {OUT}  ({len(glyph_imgs)} fonts, {sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
