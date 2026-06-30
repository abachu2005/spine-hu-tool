"""Generate the app icon set (PNG/ICO/ICNS) used by the installers.

Draws a simple, greyscale, medical-imaging-styled mark: a dark rounded square
with a light stylized vertebra glyph and "HU" label. Run once (committed
outputs are fine), or re-run to regenerate:

    python packaging/make_icons.py
"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(__file__), "icons")
BASE = 1024


def _font(size: int):
    for name in ("Helvetica.ttc", "Arial.ttf", "DejaVuSans-Bold.ttf",
                 "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw(px: int) -> Image.Image:
    s = px
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # rounded-square background, dark slate (PACS-like)
    pad = int(s * 0.06)
    radius = int(s * 0.20)
    d.rounded_rectangle([pad, pad, s - pad, s - pad], radius=radius,
                        fill=(28, 30, 34, 255))

    # central vertebral-body "core": a light ring with a brighter center disc
    cx, cy = s / 2, int(s * 0.42)
    r_out = int(s * 0.20)
    r_in = int(s * 0.135)
    d.ellipse([cx - r_out, cy - r_out, cx + r_out, cy + r_out],
              outline=(225, 228, 232, 255), width=max(2, int(s * 0.022)))
    d.ellipse([cx - r_in, cy - r_in, cx + r_in, cy + r_in],
              fill=(150, 156, 164, 255))

    # "HU" label beneath, the measurement the tool produces. Skip at tiny
    # sizes where it is illegible and the bitmap font misbehaves.
    if s >= 64:
        font = _font(max(8, int(s * 0.20)))
        text = "HU"
        bbox = d.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        d.text((cx - tw / 2 - bbox[0], int(s * 0.66) - bbox[1]), text,
               font=font, fill=(225, 228, 232, 255))
    return img


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    master = _draw(BASE)

    png_path = os.path.join(OUT, "spine_hu.png")
    master.save(png_path)

    ico_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    master.save(os.path.join(OUT, "spine_hu.ico"), sizes=ico_sizes)

    # ICNS via Pillow (expects square sizes up to 1024).
    icns_sizes = [16, 32, 64, 128, 256, 512, 1024]
    icns_imgs = [_draw(p) for p in icns_sizes]
    try:
        icns_imgs[-1].save(os.path.join(OUT, "spine_hu.icns"),
                           append_images=icns_imgs[:-1])
    except Exception as exc:  # noqa: BLE001 - icns is best-effort on non-mac
        print(f"icns generation skipped: {exc}")

    print(f"wrote icons to {OUT}")


if __name__ == "__main__":
    main()
