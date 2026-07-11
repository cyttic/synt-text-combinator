#!/usr/bin/env python3
"""Finetune exp23 into a SHORT-CROP verifier (single letters + bigrams).

Warm-start `cyttic/exp23-directfit-unfrozen`, train gently (unfrozen, LR 1e-5, 1 epoch)
on out/verifier_ds. fp16 for the local RTX 2080 (8GB): batch 4 x grad-accum 4.

Exit report — exact-match rate (the metric verification actually uses) on:
  1. held-out font crops (out/verifier_ds/labels_test.tsv)
  2. the 100 POC font renders (out/font) — exp23 baseline was 88% on 3-view
  3. handwriting-domain probe: the verified diffused units (out/verified*/...) — never trained on

    python 07_train_verifier.py                  # -> out/verifier/
    python 07_train_verifier.py --encoder-frozen # lighter fallback if OOM
"""
import argparse, glob, os, sys

import numpy as np
import torch
from PIL import Image

TROCR = "/mnt/ssd2/cyttic/projects/TrOCR_Hebrew"
sys.path.insert(0, TROCR)
from block_processor import HebrewBlockProcessor  # noqa: E402

HebrewBlockProcessor.AUTOCROP = False


class CropDS(torch.utils.data.Dataset):
    def __init__(self, ds_dir, split):
        self.imdir = os.path.join(ds_dir, "images")
        self.rows = [l.rstrip("\n").split("\t") for l in
                     open(os.path.join(ds_dir, f"labels_{split}.tsv"), encoding="utf-8")
                     if l.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        name, text = self.rows[i]
        return {"image": Image.open(os.path.join(self.imdir, name)).convert("RGB"),
                "text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cyttic/exp23-directfit-unfrozen")
    ap.add_argument("--ds", default="out/verifier_ds")
    ap.add_argument("--out", default="out/verifier")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4, dest="grad_accum")
    ap.add_argument("--encoder-frozen", action="store_true", dest="frozen")
    ap.add_argument("--eval-n", type=int, default=1500, dest="eval_n")
    args = ap.parse_args()

    from transformers import (VisionEncoderDecoderModel, AutoTokenizer,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)
    import jiwer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, info = VisionEncoderDecoderModel.from_pretrained(args.model, output_loading_info=True)
    assert not [k for k in info["missing_keys"] if k.startswith("encoder")], "encoder not loaded!"
    tok = AutoTokenizer.from_pretrained(args.model)
    model.generation_config.decoder_start_token_id = tok.cls_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.sep_token_id
    model.generation_config.max_new_tokens = None
    model.generation_config.max_length = 16
    if args.frozen:
        for p in model.encoder.parameters():
            p.requires_grad = False
    model = model.to(dev)
    proc = HebrewBlockProcessor()

    train_ds, test_ds = CropDS(args.ds, "train"), CropDS(args.ds, "test")
    print(f"train {len(train_ds):,} | test {len(test_ds):,} | "
          f"{'FROZEN' if args.frozen else 'unfrozen'} lr={args.lr}")

    def collate(b):
        pv = proc([e["image"] for e in b])["pixel_values"]
        lb = tok([e["text"] for e in b], padding="longest", truncation=True,
                 max_length=16, return_tensors="pt").input_ids
        lb[lb == tok.pad_token_id] = -100
        return {"pixel_values": pv, "labels": lb}

    def metrics(p):
        pi = np.where(p.predictions < 0, tok.pad_token_id, p.predictions)
        li = np.where(p.label_ids < 0, tok.pad_token_id, p.label_ids)
        ps = [s.strip() for s in tok.batch_decode(pi, skip_special_tokens=True)]
        ls = [s.strip() for s in tok.batch_decode(li, skip_special_tokens=True)]
        return {"cer": jiwer.cer(ls, ps),
                "exact": sum(a == b for a, b in zip(ps, ls)) / len(ls)}

    eval_sub = torch.utils.data.Subset(test_ds, range(min(args.eval_n, len(test_ds))))
    targs = Seq2SeqTrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch * 2,
        gradient_accumulation_steps=args.grad_accum, weight_decay=0.01, warmup_ratio=0.05,
        fp16=True, predict_with_generate=True, generation_max_length=16, generation_num_beams=1,
        eval_strategy="steps", eval_steps=1000, save_strategy="steps", save_steps=2000,
        save_total_limit=2, load_best_model_at_end=False, logging_steps=100,
        dataloader_num_workers=4, remove_unused_columns=False, report_to="none")
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train_ds,
                             eval_dataset=eval_sub, data_collator=collate,
                             compute_metrics=metrics)
    trainer.train()
    final = os.path.join(args.out, "final")
    trainer.save_model(final); tok.save_pretrained(final)
    print("saved ->", final)

    # ---- exit report: exact-match on the three domains ----
    @torch.no_grad()
    def exact_rate(pairs, tag):
        model.eval(); hits = 0
        for s in range(0, len(pairs), 16):
            chunk = pairs[s:s + 16]
            pv = proc([im for im, _ in chunk])["pixel_values"].to(dev, dtype=model.dtype)
            ids = model.generate(pv, num_beams=1, max_length=16)
            rd = [s.strip() for s in tok.batch_decode(ids, skip_special_tokens=True)]
            hits += sum(r == t for r, (_, t) in zip(rd, chunk))
        print(f"EXIT [{tag}] exact-match {hits}/{len(pairs)} = {100 * hits / len(pairs):.1f}%")

    holdout = [(test_ds[i]["image"], test_ds[i]["text"])
               for i in range(min(1000, len(test_ds)))]
    exact_rate(holdout, "held-out font crops")

    poc = []
    man = "out/font/manifest.tsv"
    if os.path.exists(man):
        for l in open(man, encoding="utf-8"):
            if l.strip():
                nm, bg = l.split("\t")[:2]
                poc.append((Image.open(os.path.join("out/font", nm)).convert("RGB"), bg))
        exact_rate(poc, "100 POC font renders")

    probe = []
    for d in sorted(glob.glob("out/verified*/[0-9]*_*")):
        bg = os.path.basename(d).split("_", 1)[1]
        for p in glob.glob(os.path.join(d, "s*.png")):
            probe.append((Image.open(p).convert("RGB"), bg))
    if probe:
        exact_rate(probe, f"diffused handwriting probe ({len(probe)} imgs, never trained)")


if __name__ == "__main__":
    main()
