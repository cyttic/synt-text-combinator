#!/usr/bin/env python3
"""Build STYLE-LOCKED unit banks: for each writer id, one full bank of
bigrams + single letters (margin-free, verified CER-0 x 3 views).

    python 12_build_style_banks.py                          # styles from out/top_styles.txt
    python 12_build_style_banks.py --styles 43,191 --variants 3
"""
import argparse, os, subprocess, sys, time

PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--styles", default=None, help="CSV writer ids (default: out/top_styles.txt)")
    ap.add_argument("--variants", type=int, default=3, help="verified variants per unit per style")
    ap.add_argument("--max-attempts", type=int, default=6, dest="max_attempts")
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--fonts", default="out/font_all")
    ap.add_argument("--htr", default="cyttic/heb-shortcrop-verifier")
    args = ap.parse_args()

    styles = (args.styles or open("out/top_styles.txt").read().strip()).split(",")
    print(f"building {len(styles)} style-locked banks: {styles}")
    t0 = time.time()
    for sid in styles:
        sid = int(sid)
        out = f"out/bank_s{sid:03d}"
        print(f"\n===== style {sid} -> {out} =====", flush=True)
        r = subprocess.run([PY, "05_gen_verified.py",
                            "--htr", args.htr, "--fonts", args.fonts,
                            "--styles", str(sid),
                            "--variants", str(args.variants),
                            "--max-attempts", str(args.max_attempts),
                            "--strength", str(args.strength),
                            "--out", out])
        if r.returncode != 0:
            print(f"style {sid} FAILED (rc={r.returncode}), continuing")
    print(f"\nALL DONE in {(time.time()-t0)/3600:.1f}h")


if __name__ == "__main__":
    main()
