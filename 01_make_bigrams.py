#!/usr/bin/env python3
"""Sample realistic Hebrew bigrams, frequency-weighted from the modern sentence corpus.

Bigrams = ADJACENT in-word letter pairs, so final forms (ךםןףץ) only appear where they
appear in real words. Weighted sampling without replacement -> common pairs dominate but
tail pairs are represented, like real text.

    python 01_make_bigrams.py --count 100
"""
import argparse, collections, os, random, re

SENTS = "/mnt/ssd2/cyttic/projects/TrOCR_Hebrew/sentences_modern.txt"
HEB = re.compile(r"[א-ת]+")  # letters only, no niqqud/punct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--sents", default=SENTS)
    ap.add_argument("--out", default="out/bigrams.txt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    freq = collections.Counter()
    with open(args.sents, encoding="utf-8") as f:
        for line in f:
            for word in HEB.findall(line):
                for a, b in zip(word, word[1:]):
                    freq[a + b] += 1
    print(f"corpus: {sum(freq.values()):,} bigram tokens, {len(freq):,} unique")

    # weighted sample WITHOUT replacement
    pool = list(freq.items())
    chosen = []
    while pool and len(chosen) < args.count:
        total = sum(c for _, c in pool)
        r = random.uniform(0, total)
        acc = 0
        for i, (bg, c) in enumerate(pool):
            acc += c
            if acc >= r:
                chosen.append(bg)
                pool.pop(i)
                break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(chosen) + "\n")
    finals = sum(1 for bg in chosen if bg[-1] in "ךםןףץ")
    print(f"wrote {len(chosen)} bigrams -> {args.out}  (word-final forms in {finals})")
    print("sample:", " ".join(chosen[:15]))


if __name__ == "__main__":
    main()
