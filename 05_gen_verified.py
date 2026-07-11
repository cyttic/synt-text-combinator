#!/usr/bin/env python3
"""Generate VERIFIED bigram variants: DiffusionPen restyle + strict TrOCR acceptance.

Accept rule (per variant): the HTR must read the bigram with CER == 0.0 on THREE views —
the image as generated, rotated +ROT deg, rotated -ROT deg. Any miss -> regenerate with
fresh noise + a fresh random writer style, until --max-attempts is exhausted.

Verifier calibration (built-in): before generating, every clean FONT render is put through
the same 3-check. Bigrams whose font render fails are counted separately — there the
verifier (line-trained TrOCR on 2-letter crops) is the bottleneck, not DiffusionPen.

    python 05_gen_verified.py --limit 5            # smoke
    python 05_gen_verified.py                      # full: 100 bigrams x 5 verified variants
    python 05_gen_verified.py --htr cyttic/exp23-directfit-unfrozen   # alt verifier (HF)
"""
import argparse, json, os, random, sys

DIFFPEN = "/mnt/ssd2/cyttic/projects/test-diff-pen"
TROCR = "/mnt/ssd2/cyttic/projects/TrOCR_Hebrew"
HTR_DEFAULT = os.path.join(TROCR, "exp10_matan_finetune/exp10-trocr-hebrew-matan-full")
sys.path.insert(0, DIFFPEN)
sys.path.insert(0, TROCR)

import numpy as np
import torch
from PIL import Image, ImageOps

from build_char_bank import build_args, load_model, to_pil  # noqa: E402

CANVAS_W, CANVAS_H = 256, 64


def pad_to_canvas(img):
    w, h = img.size
    img = img.resize((max(1, int(w * CANVAS_H / h)), CANVAS_H))
    if img.width < CANVAS_W:
        return ImageOps.pad(img, size=(CANVAS_W, CANVAS_H), color="white")
    while img.width > CANVAS_W:
        img = img.resize((img.width - 20, CANVAS_H))
    return ImageOps.pad(img, size=(CANVAS_W, CANVAS_H), color="white")


def tight(img, margin=4):
    bbox = ImageOps.invert(img.convert("L")).getbbox()
    if bbox is None:
        return img
    x0, y0, x1, y1 = bbox
    return img.crop((max(0, x0 - margin), max(0, y0 - margin),
                     min(img.width, x1 + margin), min(img.height, y1 + margin)))


class StyleBank:
    def __init__(self, transform, device):
        wr = json.load(open(f"{DIFFPEN}/hebrew/writers_dict_train.json"))
        self.rev = {v: k for k, v in wr.items()}
        rows = [l.strip().split(",") for l in open(f"{DIFFPEN}/hebrew/splits_words/heb_train_val.txt")]
        self.by_writer = {}
        for r in rows:
            self.by_writer.setdefault(r[1], []).append(r)
        self.transform, self.device = transform, device

    def images(self, style_id):
        rows = self.by_writer[self.rev[style_id]]
        good = [r for r in rows if len(r[2]) > 3] or rows
        five = random.sample(good, 5) if len(good) >= 5 else (good * 5)[:5]
        return torch.stack([self.transform(pad_to_canvas(
            Image.open(f"{DIFFPEN}/hebrew/words/{r[0]}").convert("RGB"))) for r in five])


class Verifier:
    """TrOCR 3-view exact-match check. AUTOCROP off (June-era processor convention)."""

    def __init__(self, model_id, device, rot_deg, beams):
        from transformers import VisionEncoderDecoderModel, AutoTokenizer
        from block_processor import HebrewBlockProcessor
        HebrewBlockProcessor.AUTOCROP = False
        self.proc = HebrewBlockProcessor()
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device).half().eval()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        gc = self.model.generation_config
        gc.decoder_start_token_id = self.tok.cls_token_id
        gc.pad_token_id = self.tok.pad_token_id
        gc.eos_token_id = self.tok.sep_token_id
        gc.max_new_tokens = None
        self.device, self.rot, self.beams = device, rot_deg, beams

    def views(self, img):
        t = tight(img)
        return [t,
                tight(t.rotate(+self.rot, expand=True, fillcolor="white")),
                tight(t.rotate(-self.rot, expand=True, fillcolor="white"))]

    @torch.no_grad()
    def read(self, pils):
        pv = self.proc(pils)["pixel_values"].to(self.device, dtype=self.model.dtype)
        ids = self.model.generate(pv, num_beams=self.beams, max_length=16)
        return [s.strip() for s in self.tok.batch_decode(ids, skip_special_tokens=True)]

    def passes(self, imgs, texts):
        """imgs: list of PIL, texts: expected per image -> list[bool], 3-view exact match."""
        flat, owner = [], []
        for i, im in enumerate(imgs):
            vs = self.views(im)
            flat += vs
            owner += [i] * len(vs)
        reads = self.read(flat)
        ok = [True] * len(imgs)
        for r, o in zip(reads, owner):
            if r != texts[o]:
                ok[o] = False
        return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", default="out/font")
    ap.add_argument("--out", default="out/verified")
    ap.add_argument("--variants", type=int, default=5, help="verified variants wanted per bigram")
    ap.add_argument("--max-attempts", type=int, default=8, dest="max_attempts",
                    help="generation attempts per variant slot")
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=6, help="diffusion candidates per forward")
    ap.add_argument("--rot-deg", type=float, default=7.0, dest="rot_deg")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--htr", default=f"{HTR_DEFAULT},cyttic/exp23-directfit-unfrozen",
                    help="comma-separated verifiers; a variant is accepted if ANY passes 3/3 "
                         "(exp10=matan-domain, exp23=font-domain; together they span both)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_path", default=f"{DIFFPEN}/hebrew_model")
    ap.add_argument("--style_path", default=f"{DIFFPEN}/style_models/iam_style_diffusionpen.pth")
    ap.add_argument("--stable_dif_path", default=f"{DIFFPEN}/stable-diffusion-v1-5")
    ap.add_argument("--device", default="cuda:0")
    cli = ap.parse_args()
    random.seed(cli.seed); torch.manual_seed(cli.seed)

    rows = [l.rstrip("\n").split("\t") for l in
            open(os.path.join(cli.fonts, "manifest.tsv"), encoding="utf-8") if l.strip()]
    if cli.limit:
        rows = rows[:cli.limit]

    vers = [Verifier(m.strip(), cli.device, cli.rot_deg, cli.beams)
            for m in cli.htr.split(",")]
    vnames = [m.strip().split("/")[-1][:24] for m in cli.htr.split(",")]
    print(f"verifiers (ANY passes 3/3 => accept): {vnames} | views: 0, +/-{cli.rot_deg} deg")

    def passes_any(imgs, texts):
        """-> (ok list, tag list: which verifier accepted first, '' if none)"""
        ok = [False] * len(imgs); tag = [""] * len(imgs)
        for v, nm in zip(vers, vnames):
            todo = [i for i in range(len(imgs)) if not ok[i]]
            if not todo:
                break
            res = v.passes([imgs[i] for i in todo], [texts[i] for i in todo])
            for i, r in zip(todo, res):
                if r:
                    ok[i] = True; tag[i] = nm
        return ok, tag

    # ---- calibration gate: ANY verifier must read the clean FONT render ----
    font_imgs = [Image.open(os.path.join(cli.fonts, r[0])).convert("RGB") for r in rows]
    texts = [r[1] for r in rows]
    font_ok = []
    for s in range(0, len(font_imgs), 16):
        font_ok += passes_any(font_imgs[s:s + 16], texts[s:s + 16])[0]
    n_ok = sum(font_ok)
    print(f"CALIBRATION: gate passes {n_ok}/{len(rows)} clean font renders "
          f"({100 * n_ok / len(rows):.0f}%). Failing bigrams are verifier-limited, skipped.")

    args = build_args(argparse.Namespace(steps=cli.steps, style=0, save_path=cli.save_path,
                      style_path=cli.style_path, stable_dif_path=cli.stable_dif_path,
                      device=cli.device))
    idx = int("".join(filter(str.isdigit, cli.device)) or 0)
    diffusion, ema, vae, ddim, fe, tok, te, transform = load_model(args, [idx])
    bank = StyleBank(transform, cli.device)
    n_writers = len(bank.rev)

    ddim.set_timesteps(cli.steps)
    t_idx = min(int(len(ddim.timesteps) * (1.0 - cli.strength)), len(ddim.timesteps) - 1)
    t_enc = ddim.timesteps[t_idx]
    run_ts = ddim.timesteps[t_idx:]
    os.makedirs(cli.out, exist_ok=True)

    def gen(canvas, bigram, style_ids):
        """one diffusion forward: len(style_ids) candidates for this bigram."""
        V = len(style_ids)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            x0 = transform(canvas).unsqueeze(0).to(cli.device)
            lat0 = vae.module.encode(x0.to(torch.float32)).latent_dist.sample() * 0.18215
            x = ddim.add_noise(lat0.repeat(V, 1, 1, 1), torch.randn((V, *lat0.shape[1:]),
                               device=cli.device), t_enc)
            tf = tok([bigram] * V, padding="max_length", truncation=True,
                     return_tensors="pt", max_length=40).to(cli.device)
            simgs = torch.cat([bank.images(s) for s in style_ids]).to(cli.device)
            sfeat = fe(simgs.reshape(-1, 3, 64, 256)).to(cli.device)
            labels = torch.tensor(style_ids).long().to(cli.device)
            for t in run_ts:
                tt = torch.full((V,), int(t), device=cli.device, dtype=torch.long)
                eps = ema(x, tt, tf, labels, original_images=simgs,
                          mix_rate=None, style_extractor=sfeat)
                x = ddim.step(eps, t, x).prev_sample
            img = vae.module.decode((1 / 0.18215 * x).to(torch.float32)).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return [to_pil(i.cpu()) for i in img]

    meta, stats, skipped = [], [], []
    total_gen = total_ok = 0
    for (name, bigram, fontfile), f_ok in zip([(r[0], r[1], r[2].strip()) for r in rows], font_ok):
        stem = os.path.splitext(name)[0]
        if not f_ok:
            skipped.append(stem)
            continue
        canvas = pad_to_canvas(Image.open(os.path.join(cli.fonts, name)).convert("RGB"))
        accepted, attempts = [], 0
        while len(accepted) < cli.variants and attempts < cli.max_attempts:
            need = min(cli.batch, cli.variants - len(accepted) + 2)  # small over-ask
            sids = random.sample(range(n_writers), need)
            cands = gen(canvas, bigram, sids)
            total_gen += need; attempts += 1
            oks, tags = passes_any(cands, [bigram] * need)
            for sid, im, ok, tg in zip(sids, cands, oks, tags):
                if ok and len(accepted) < cli.variants:
                    accepted.append((sid, im, tg))
        d = os.path.join(cli.out, stem); os.makedirs(d, exist_ok=True)
        canvas.save(os.path.join(d, "font.png"))
        for k, (sid, im, tg) in enumerate(accepted):
            im.save(os.path.join(d, f"s{sid:03d}_v{k}.png"))
        total_ok += len(accepted)
        by_tag = ",".join(f"{s}:{t}" for s, _, t in accepted)
        meta.append(f"{stem}\t{bigram}\t{fontfile}\t{','.join(str(s) for s, _, _ in accepted)}")
        stats.append(f"{stem}\t{bigram}\t{len(accepted)}/{cli.variants}\t{attempts} attempts\t{by_tag}")
        print(f"{stem} [{bigram}] {len(accepted)}/{cli.variants} verified "
              f"in {attempts} attempts [{by_tag}]", flush=True)

    with open(os.path.join(cli.out, "manifest.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(meta) + "\n")
    with open(os.path.join(cli.out, "stats.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(stats) + "\n")
        f.write(f"# skipped (verifier can't read even the font render): {','.join(skipped)}\n")
    keep = 100 * total_ok / max(1, total_gen)
    print(f"\nDONE: {total_ok} verified variants | {total_gen} generated | keep-rate {keep:.0f}% "
          f"| {len(skipped)} bigrams verifier-skipped")
    print(f"gallery: python 04_make_gallery.py --diffused {cli.out}")


if __name__ == "__main__":
    main()
