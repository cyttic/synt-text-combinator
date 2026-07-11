#!/usr/bin/env python3
"""Restyle font-rendered bigrams with the Hebrew DiffusionPen (SDEdit img2img).

For each font render: pad to the model's 64x256 canvas -> VAE-encode -> add noise up to an
intermediate timestep (strength) -> DDIM-denoise conditioned on (bigram text via CANINE,
writer style via 5 style images + class id). The font latent anchors the CONTENT; the
conditioning + noise inject the STYLE. strength: 0 = copy the font, 1 = ignore it entirely.

Reuses test-diff-pen verbatim: build_char_bank.load_model (UNet/EMA + VAE + style encoder),
hebrew/writers_dict_train.json + heb_train_val.txt + hebrew/words for style images —
exactly the assets train_hebrew.Diffusion.sampling uses.

    python 03_diffpen_restyle.py                     # full run: all bigrams x 5 styles
    python 03_diffpen_restyle.py --limit 2 --variants 2 --steps 20   # smoke test
"""
import argparse, json, os, random, sys

DIFFPEN = "/mnt/ssd2/cyttic/projects/test-diff-pen"
sys.path.insert(0, DIFFPEN)

import numpy as np
import torch
from PIL import Image, ImageOps

from build_char_bank import build_args, load_model, to_pil  # noqa: E402

CANVAS_W, CANVAS_H = 256, 64


def pad_to_canvas(img):
    """resize to 64px height then pad/center into 64x256 white — same as sampling()."""
    w, h = img.size
    img = img.resize((max(1, int(w * CANVAS_H / h)), CANVAS_H))
    if img.width < CANVAS_W:
        return ImageOps.pad(img, size=(CANVAS_W, CANVAS_H), color="white")
    while img.width > CANVAS_W:
        img = img.resize((img.width - 20, CANVAS_H))
    return ImageOps.pad(img, size=(CANVAS_W, CANVAS_H), color="white")


class StyleBank:
    """5 style images per writer id, from the SAME files train_hebrew.sampling reads."""

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
        ims = []
        for r in five:
            im = Image.open(f"{DIFFPEN}/hebrew/words/{r[0]}").convert("RGB")
            ims.append(self.transform(pad_to_canvas(im)))
        return torch.stack(ims)  # [5,3,64,256]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", default="out/font")
    ap.add_argument("--out", default="out/diffused")
    ap.add_argument("--variants", type=int, default=5, help="style variants per bigram")
    ap.add_argument("--strength", type=float, default=0.6, help="0=copy font .. 1=free generation")
    ap.add_argument("--steps", type=int, default=30, help="DDIM steps (full schedule)")
    ap.add_argument("--limit", type=int, default=None, help="only first N bigrams (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_path", default=f"{DIFFPEN}/hebrew_model")
    ap.add_argument("--style_path", default=f"{DIFFPEN}/style_models/iam_style_diffusionpen.pth")
    ap.add_argument("--stable_dif_path", default=f"{DIFFPEN}/stable-diffusion-v1-5")
    ap.add_argument("--device", default="cuda:0")
    cli = ap.parse_args()
    random.seed(cli.seed); torch.manual_seed(cli.seed)

    args = build_args(argparse.Namespace(steps=cli.steps, style=0, save_path=cli.save_path,
                      style_path=cli.style_path, stable_dif_path=cli.stable_dif_path,
                      device=cli.device))
    idx = int("".join(filter(str.isdigit, cli.device)) or 0)
    diffusion, ema, vae, ddim, fe, tok, te, transform = load_model(args, [idx])
    bank = StyleBank(transform, cli.device)
    n_writers = len(bank.rev)
    print(f"model loaded | {n_writers} writer styles | strength {cli.strength} | steps {cli.steps}")

    rows = [l.split("\t") for l in open(os.path.join(cli.fonts, "manifest.tsv"), encoding="utf-8")
            if l.strip()]
    if cli.limit:
        rows = rows[:cli.limit]
    os.makedirs(cli.out, exist_ok=True)

    ddim.set_timesteps(cli.steps)
    t_idx = min(int(len(ddim.timesteps) * (1.0 - cli.strength)), len(ddim.timesteps) - 1)
    t_enc = ddim.timesteps[t_idx]
    run_ts = ddim.timesteps[t_idx:]
    print(f"entering schedule at t={int(t_enc)} ({len(run_ts)}/{len(ddim.timesteps)} steps run)")

    meta = []
    for name, bigram, fontfile in [(r[0], r[1], r[2].strip()) for r in rows]:
        font_img = Image.open(os.path.join(cli.fonts, name)).convert("RGB")
        canvas = pad_to_canvas(font_img)
        style_ids = random.sample(range(n_writers), cli.variants)
        V = cli.variants

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            # content anchor: font image -> latent
            x0 = transform(canvas).unsqueeze(0).to(cli.device)
            lat0 = vae.module.encode(x0.to(torch.float32)).latent_dist.sample() * 0.18215
            lat0 = lat0.repeat(V, 1, 1, 1)
            noise = torch.randn_like(lat0)
            x = ddim.add_noise(lat0, noise, t_enc)

            # conditioning: text + per-variant writer style (5 imgs each, like sampling())
            text_features = tok([bigram] * V, padding="max_length", truncation=True,
                                return_tensors="pt", max_length=40).to(cli.device)
            style_images = torch.cat([bank.images(s) for s in style_ids]).to(cli.device)
            style_features = fe(style_images.reshape(-1, 3, 64, 256)).to(cli.device)
            labels = torch.tensor(style_ids).long().to(cli.device)

            for t in run_ts:
                tt = torch.full((V,), int(t), device=cli.device, dtype=torch.long)
                eps = ema(x, tt, text_features, labels, original_images=style_images,
                          mix_rate=None, style_extractor=style_features)
                x = ddim.step(eps, t, x).prev_sample

            img = vae.module.decode((1 / 0.18215 * x).to(torch.float32)).sample
            img = (img / 2 + 0.5).clamp(0, 1)

        stem = os.path.splitext(name)[0]
        d = os.path.join(cli.out, stem); os.makedirs(d, exist_ok=True)
        canvas.save(os.path.join(d, "font.png"))
        for sid, im in zip(style_ids, img):
            to_pil(im.cpu()).save(os.path.join(d, f"s{sid:03d}.png"))
        meta.append(f"{stem}\t{bigram}\t{fontfile}\t{','.join(str(s) for s in style_ids)}")
        print(f"{stem} [{bigram}] -> {V} variants (styles {style_ids})", flush=True)

    with open(os.path.join(cli.out, "manifest.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(meta) + "\n")
    print(f"done: {len(meta)} bigrams x {cli.variants} variants -> {cli.out}")


if __name__ == "__main__":
    main()
