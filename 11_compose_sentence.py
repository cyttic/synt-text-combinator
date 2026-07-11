#!/usr/bin/env python3
"""Compose full sentence LINE images from the verified unit banks (bigrams + letters).

Tiling: each word is segmented by DP over the banks — bigrams preferred, verified single
letters as fallback — so any word whose letters exist in the bank is composable.
Assembly: units keep their FULL 64px canvas height (baseline preserved by generation),
are ink-cropped horizontally only, and pasted RIGHT-TO-LEFT with jittered intra-word /
word gaps and small vertical jitter.

Built-in quality check: each composed line is OCR'd by a LINE-domain HTR (default exp10,
matan-trained — the strictest realistic reader) and per-line CER vs the target is printed.

    python 11_compose_sentence.py --text "..."             # compose one sentence, 3 variants
    python 11_compose_sentence.py --from-corpus 10         # 10 random tileable corpus sentences
"""
import argparse, glob, os, random, re, sys

import numpy as np
import torch
from PIL import Image, ImageOps

TROCR = "/mnt/ssd2/cyttic/projects/TrOCR_Hebrew"
SENTS = os.path.join(TROCR, "sentences_modern.txt")
HTR_DEFAULT = os.path.join(TROCR, "exp10_matan_finetune/exp10-trocr-hebrew-matan-full")
sys.path.insert(0, TROCR)
HEB = re.compile(r"[א-ת]+")
H = 64


def load_banks(dirs):
    bank = {}
    for root in dirs:
        for d in glob.glob(os.path.join(root, "[0-9]*_*")):
            text = os.path.basename(d).split("_", 1)[1]
            for p in glob.glob(os.path.join(d, "s*.png")):
                bank.setdefault(text, []).append(p)
    return bank


def expand_bank_dirs(spec):
    """comma-separated paths, each may be a glob (e.g. out/bank_s*)."""
    dirs = []
    for part in spec.split(","):
        hits = sorted(glob.glob(part)) or [part]
        dirs += [h for h in hits if os.path.isdir(h)]
    return dirs


def tile(word, bank):
    """DP segmentation of `word` into bank units; bigrams preferred. None if impossible."""
    n = len(word)
    INF = 1e9
    cost = [INF] * (n + 1); back = [None] * (n + 1)
    cost[0] = 0
    for i in range(n):
        if cost[i] >= INF:
            continue
        for L, c in ((2, 1.0), (1, 1.6)):          # bigram cheaper than single
            if i + L <= n and word[i:i + L] in bank and cost[i] + c < cost[i + L]:
                cost[i + L] = cost[i] + c; back[i + L] = L
    if cost[n] >= INF:
        return None
    segs, i = [], n
    while i > 0:
        L = back[i]; segs.append(word[i - L:i]); i -= L
    return segs[::-1]                               # logical order


INK_THR = 200  # grayscale < thr counts as ink (diffused backgrounds are noisy, not pure white)


def ink_bbox(img):
    a = np.asarray(img.convert("L")) < INK_THR
    cols = np.where(a.any(axis=0))[0]
    rows = np.where(a.any(axis=1))[0]
    if len(cols) == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def hcrop(img, margin=2):
    """crop horizontally to REAL ink (thresholded), KEEP full height (baseline preserved)."""
    bb = ink_bbox(img)
    if bb is None:
        return img
    x0, _, x1, _ = bb
    return img.crop((max(0, x0 - margin), 0, min(img.width, x1 + margin), img.height))


def compose(sentence, bank, rng):
    words = HEB.findall(sentence)
    pieces = []                                     # (PIL, is_word_start) in RTL paste order
    for word in words:
        segs = tile(word, bank)
        if segs is None:
            return None, f"cannot tile: {word}"
        for k, seg in enumerate(segs):
            img = hcrop(Image.open(rng.choice(bank[seg])).convert("RGB"))
            pieces.append((img, k == 0))
    intra = lambda: int(rng.integers(2, 8))
    wordgap = lambda: int(rng.integers(16, 30))
    width = sum(im.width for im, _ in pieces) \
        + sum(wordgap() if ws else intra() for _, ws in pieces[1:]) + 40
    canvas = Image.new("RGB", (width, H + 12), "white")
    x = width - 20                                  # start at the RIGHT (RTL)
    for i, (im, word_start) in enumerate(pieces):
        if i > 0:
            x -= wordgap() if word_start else intra()
        x -= im.width
        canvas.paste(im, (x, 6 + int(rng.integers(-2, 3))))
    bb = ink_bbox(canvas)
    if bb:
        x0, y0, x1, y1 = bb
        canvas = canvas.crop((max(0, x0 - 10), max(0, y0 - 4),
                              min(canvas.width, x1 + 10), min(canvas.height, y1 + 4)))
    return canvas, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None, help="sentence to compose")
    ap.add_argument("--from-corpus", type=int, default=0, dest="n_corpus",
                    help="compose N random tileable corpus sentences instead")
    ap.add_argument("--variants", type=int, default=3, help="renders per sentence")
    ap.add_argument("--banks", default="out/verified_v2,out/verified_letters",
                    help="comma-separated bank dirs, globs OK (e.g. 'out/bank_s*')")
    ap.add_argument("--lock-style", action="store_true", dest="lock_style",
                    help="treat each bank dir as ONE style; compose every line from a single "
                         "dir that covers the whole sentence (no style mixing)")
    ap.add_argument("--out", default="out/sentences")
    ap.add_argument("--htr", default=HTR_DEFAULT, help="line-domain OCR for the quality check")
    ap.add_argument("--no-ocr", action="store_true", dest="no_ocr")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    bank_dirs = expand_bank_dirs(args.banks)
    if args.lock_style:
        style_banks = {os.path.basename(d): load_banks([d]) for d in bank_dirs}
        style_banks = {k: v for k, v in style_banks.items() if v}
        bank = {}                                   # union, for tileability screening
        for b in style_banks.values():
            for t, ps in b.items():
                bank.setdefault(t, []).extend(ps)
        print(f"style-locked: {len(style_banks)} banks: {list(style_banks)}")
    else:
        style_banks = None
        bank = load_banks(bank_dirs)
    n_units = sum(len(v) for v in bank.values())
    letters_cov = sorted({t for t in bank if len(t) == 1})
    print(f"bank: {len(bank)} unit texts, {n_units} images | single letters: "
          f"{''.join(letters_cov)} ({len(letters_cov)}/27)")

    if args.text:
        sentences = [args.text.strip()]
    else:
        alls = [l.strip() for l in open(SENTS, encoding="utf-8") if l.strip()]
        idx = rng.permutation(len(alls))
        sentences = []
        for i in idx:
            s = alls[i]
            words = HEB.findall(s)
            if words and 3 <= len(words) <= 9 and all(tile(w, bank) for w in words):
                sentences.append(s)
            if len(sentences) >= args.n_corpus:
                break
        print(f"picked {len(sentences)} tileable corpus sentences")

    os.makedirs(args.out, exist_ok=True)
    made = []
    for si, s in enumerate(sentences):
        for v in range(args.variants):
            if style_banks:
                words = HEB.findall(s)
                covering = [k for k, b in style_banks.items()
                            if all(tile(w, b) for w in words)]
                if not covering:
                    print(f"[{si}] no single style covers: {s[:40]}"); break
                key = covering[int(rng.integers(len(covering)))]
                use_bank, tag = style_banks[key], key
            else:
                use_bank, tag = bank, "mixed"
            img, err = compose(s, use_bank, rng)
            if err:
                print(f"[{si}] {err}"); break
            p = os.path.join(args.out, f"{si:03d}_{v}_{tag}.png")
            img.save(p); made.append((p, s))
    print(f"composed {len(made)} line images -> {args.out}")

    if made and not args.no_ocr:
        from transformers import VisionEncoderDecoderModel, AutoTokenizer
        from block_processor import HebrewBlockProcessor
        import jiwer
        HebrewBlockProcessor.AUTOCROP = False
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = VisionEncoderDecoderModel.from_pretrained(args.htr).to(dev).half().eval()
        tok = AutoTokenizer.from_pretrained(args.htr)
        gc = model.generation_config
        gc.decoder_start_token_id = tok.cls_token_id; gc.pad_token_id = tok.pad_token_id
        gc.eos_token_id = tok.sep_token_id; gc.max_new_tokens = None
        proc = HebrewBlockProcessor()
        cers = []
        with open(os.path.join(args.out, "ocr_check.tsv"), "w", encoding="utf-8") as f:
            for s in range(0, len(made), 8):
                chunk = made[s:s + 8]
                pv = proc([Image.open(p).convert("RGB") for p, _ in chunk])["pixel_values"] \
                    .to(dev, dtype=model.dtype)
                with torch.no_grad():
                    ids = model.generate(pv, num_beams=4, max_length=64)
                for (p, gt), rd in zip(chunk, tok.batch_decode(ids, skip_special_tokens=True)):
                    gt_w = " ".join(HEB.findall(gt))
                    c = jiwer.cer(gt_w, rd.strip()); cers.append(c)
                    f.write(f"{os.path.basename(p)}\t{c:.3f}\t{gt_w}\t{rd.strip()}\n")
        print(f"OCR check ({os.path.basename(args.htr)}): mean CER {np.mean(cers)*100:.1f}% "
              f"| median {np.median(cers)*100:.1f}% | <=20%: {sum(c <= .2 for c in cers)}/{len(cers)}")
        print(f"details -> {args.out}/ocr_check.tsv")


if __name__ == "__main__":
    main()
