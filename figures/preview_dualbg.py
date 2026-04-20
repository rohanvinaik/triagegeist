"""Composite every transparent PNG against both Kaggle backgrounds for visual QA."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from style import DARK_BG, LIGHT_BG


def side_by_side(src: Path, out: Path, pad: int = 64) -> None:
    im = Image.open(src).convert("RGBA")
    W, H = im.size
    light = Image.new("RGBA", (W + 2 * pad, H + 2 * pad), LIGHT_BG + (255,))
    dark = Image.new("RGBA", (W + 2 * pad, H + 2 * pad), DARK_BG + (255,))
    light.alpha_composite(im, (pad, pad))
    dark.alpha_composite(im, (pad, pad))

    divider_w = 16
    canvas = Image.new(
        "RGBA",
        (light.width + dark.width + divider_w, light.height),
        (200, 200, 200, 0),
    )
    canvas.paste(light, (0, 0))
    canvas.paste(dark, (light.width + divider_w, 0))
    canvas.save(out)


def main() -> None:
    here = Path(__file__).parent
    out_dir = here / "_preview"
    out_dir.mkdir(exist_ok=True)
    for src in sorted(here.glob("*.png")):
        side_by_side(src, out_dir / f"{src.stem}_dualbg.png")
        print(f"  {src.name} → _preview/{src.stem}_dualbg.png")


if __name__ == "__main__":
    main()
