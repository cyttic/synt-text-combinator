#!/usr/bin/env python3
"""Build out/gallery.html — one row per bigram: font render | diffused variants.

    python 04_make_gallery.py && xdg-open out/gallery.html
"""
import argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("--diffused", default="out/diffused")
ap.add_argument("--out", default="out/gallery.html")
args = ap.parse_args()

rows = [l.rstrip("\n").split("\t") for l in
        open(os.path.join(args.diffused, "manifest.tsv"), encoding="utf-8") if l.strip()]

html = ["""<!doctype html><meta charset="utf-8"><title>bigram restyle gallery</title>
<style>
 body{font-family:sans-serif;background:#1e1e1e;color:#ddd;margin:20px}
 table{border-collapse:collapse} td,th{padding:6px 10px;border-bottom:1px solid #333;text-align:center}
 img{height:64px;background:#fff;image-rendering:auto}
 .bg{font-size:26px;direction:rtl} .meta{color:#888;font-size:11px}
</style>
<h2>font-anchored DiffusionPen bigrams</h2>
<table><tr><th>#</th><th>bigram</th><th>font render</th><th colspan=9>diffused variants</th></tr>"""]

for stem, bigram, fontfile, sids in rows:
    d = os.path.join(args.diffused, stem)
    variants = sorted(f for f in os.listdir(d) if f.startswith("s") and f.endswith(".png"))
    cells = "".join(
        f'<td><img src="{os.path.join(os.path.basename(args.diffused), stem, v)}">'
        f'<div class="meta">{v[:-4]}</div></td>' for v in variants)
    html.append(
        f'<tr><td class="meta">{stem.split("_")[0]}</td><td class="bg">{bigram}'
        f'<div class="meta">{fontfile}</div></td>'
        f'<td><img src="{os.path.join(os.path.basename(args.diffused), stem, "font.png")}"></td>'
        f'{cells}</tr>')

html.append("</table>")
with open(args.out, "w", encoding="utf-8") as f:
    f.write("\n".join(html))
print(f"wrote {args.out} ({len(rows)} rows)")
