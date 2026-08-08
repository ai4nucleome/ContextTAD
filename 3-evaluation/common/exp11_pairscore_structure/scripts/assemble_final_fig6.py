#!/usr/bin/env python3
"""Assemble pair-score analyses and three pyGenomeTracks regions into final Fig. 6."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def load_font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def resize_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-figure", type=Path, required=True)
    parser.add_argument("--regions-table", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=350)
    args = parser.parse_args()

    main_figure = Image.open(args.main_figure).convert("RGB")
    regions = pd.read_csv(args.regions_table, sep="\t")
    region_images = []
    for row in regions.itertuples(index=False):
        chrom, start, end = str(row.chrom), int(row.start), int(row.end)
        path = args.asset_dir / f"{chrom}_{start}_{end}" / f"{chrom}_{start}_{end}.png"
        if not path.exists():
            raise FileNotFoundError(path)
        region_images.append(Image.open(path).convert("RGB"))

    target_width = max([main_figure.width, *[image.width for image in region_images]])
    region_images = [resize_width(image, target_width) for image in region_images]
    # Align the analytical plotting span (first pileup through Q4) with the
    # Hi-C plotting span in the representative-region panels. These normalized
    # coordinates are stable because both source figures use fixed layouts.
    analysis_left = 343 / 5205
    analysis_right = 5075 / 5205
    region_left = 320 / 6200
    region_right = 5256 / 6200
    analysis_span = analysis_right - analysis_left
    region_span = region_right - region_left
    main_width = round(target_width * region_span / analysis_span)
    main_figure = resize_width(main_figure, main_width)
    main_x = round(target_width * region_left - main_figure.width * analysis_left)

    section_gap = 65
    region_gap = 28
    total_height = (
        main_figure.height
        + section_gap
        + sum(image.height for image in region_images)
        + region_gap * (len(region_images) - 1)
    )
    canvas = Image.new("RGB", (target_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)
    panel_font = load_font(max(88, target_width // 55))

    canvas.paste(main_figure, (main_x, 0))
    panel_x = 26
    draw.text((panel_x, 10), "(a)", fill="black", font=panel_font)
    draw.text((panel_x, round(main_figure.height * 0.515)), "(b)", fill="black", font=panel_font)
    y = main_figure.height + section_gap
    for index, image in enumerate(region_images):
        canvas.paste(image, (0, y))
        draw.text((panel_x, y + 10), f"({chr(ord('c') + index)})", fill="black", font=panel_font)
        y += image.height + region_gap

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output_prefix.with_suffix(".png"), dpi=(args.dpi, args.dpi), optimize=True)
    canvas.save(args.output_prefix.with_suffix(".pdf"), resolution=args.dpi)
    print(f"Final Fig. 6: {canvas.width} x {canvas.height} pixels")


if __name__ == "__main__":
    main()
