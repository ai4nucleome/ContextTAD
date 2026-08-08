#!/usr/bin/env python3
"""Select and draw three reproducible ContextTAD representative regions."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import shutil
import subprocess

import cooler
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


RESOLUTION = 5_000
WINDOW_BP = 2_000_000
STEP_BP = 1_000_000
CHROMS = ("chr15", "chr16", "chr17")
TARGET_COMPLEXITY = {"chr15": 0.20, "chr16": 0.50, "chr17": 0.80}

def load_tads(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2])
    frame.columns = ["chrom", "left", "right_boundary"]
    frame = frame[frame["chrom"].isin(CHROMS)].copy()
    frame["right_exclusive"] = frame["right_boundary"] + RESOLUTION
    frame["length_bp"] = frame["right_exclusive"] - frame["left"]
    return frame


def topology_features(intervals: list[tuple[int, int]]) -> dict[str, float | int]:
    n_tads = len(intervals)
    nested = 0
    partial = 0
    for i, (left_i, right_i) in enumerate(intervals):
        nested += int(
            any(
                j != i
                and left_j <= left_i
                and right_i <= right_j
                and (left_j < left_i or right_i < right_j)
                for j, (left_j, right_j) in enumerate(intervals)
            )
        )
        for left_j, right_j in intervals[i + 1 :]:
            partial += int(
                (left_i < left_j < right_i < right_j)
                or (left_j < left_i < right_j < right_i)
            )

    endpoint_counts = Counter([x for interval in intervals for x in interval])
    shared_positions = sum(count >= 2 for count in endpoint_counts.values())
    shared_excess = sum(max(0, count - 1) for count in endpoint_counts.values())
    lengths = np.asarray([right - left for left, right in intervals], dtype=float)
    return {
        "n_tads": n_tads,
        "nested_tads": nested,
        "nested_fraction": nested / n_tads if n_tads else 0.0,
        "shared_boundary_positions": shared_positions,
        "shared_boundary_excess": shared_excess,
        "shared_boundary_fraction": shared_excess / (2 * n_tads) if n_tads else 0.0,
        "partial_overlap_pairs": partial,
        "median_tad_size": float(np.median(lengths)) if n_tads else np.nan,
        "tad_size_cv": float(np.std(lengths) / np.mean(lengths)) if n_tads and np.mean(lengths) else 0.0,
    }


def enumerate_candidates(tads: pd.DataFrame, mcool: Path) -> pd.DataFrame:
    clr = cooler.Cooler(f"{mcool}::/resolutions/{RESOLUTION}")
    rows: list[dict[str, float | int | str]] = []
    for chrom in CHROMS:
        chromsize = int(clr.chromsizes[chrom])
        bins = clr.bins().fetch(chrom)
        valid = np.isfinite(bins["weight"].to_numpy(dtype=float))
        chrom_tads = tads[tads["chrom"] == chrom]
        for start in range(0, chromsize - WINDOW_BP + 1, STEP_BP):
            end = start + WINDOW_BP
            lo = start // RESOLUTION
            hi = min(int(np.ceil(end / RESOLUTION)), len(valid))
            valid_fraction = float(valid[lo:hi].mean())
            subset = chrom_tads[
                (chrom_tads["left"] >= start)
                & (chrom_tads["right_exclusive"] <= end)
            ]
            intervals = list(zip(subset["left"].astype(int), subset["right_exclusive"].astype(int)))
            if intervals:
                prediction_span_fraction = (
                    max(right for _, right in intervals) - min(left for left, _ in intervals)
                ) / WINDOW_BP
            else:
                prediction_span_fraction = 0.0
            row: dict[str, float | int | str] = {
                "chrom": chrom,
                "start": start,
                "end": end,
                "valid_bin_fraction": valid_fraction,
                "prediction_span_fraction": prediction_span_fraction,
            }
            row.update(topology_features(intervals))
            rows.append(row)

    candidates = pd.DataFrame(rows)
    candidates = candidates[
        (candidates["valid_bin_fraction"] >= 0.98)
        & (candidates["n_tads"] >= 3)
    ].copy()
    for chrom in CHROMS:
        select = candidates["chrom"] == chrom
        group = candidates.loc[select]
        components = []
        for column in ("n_tads", "nested_fraction", "shared_boundary_fraction", "tad_size_cv"):
            components.append(group[column].rank(method="average", pct=True).to_numpy())
        candidates.loc[select, "complexity_score"] = np.mean(np.vstack(components), axis=0)
    candidates["target_complexity"] = candidates["chrom"].map(TARGET_COMPLEXITY)
    candidates["target_distance"] = (
        candidates["complexity_score"] - candidates["target_complexity"]
    ).abs()
    return candidates.sort_values(["chrom", "start"]).reset_index(drop=True)


def select_regions(candidates: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for chrom in CHROMS:
        group = candidates[candidates["chrom"] == chrom].copy()
        if chrom == "chr17":
            fixed = group[(group["start"] == 49_000_000) & (group["end"] == 51_000_000)]
            if len(fixed) != 1:
                raise RuntimeError("The fixed chr17:49-51 Mb representative region is unavailable")
            choice = fixed.iloc[0].copy()
            choice["selection_role"] = "predefined complex multiscale region"
            selected.append(choice)
            continue
        if chrom == "chr15":
            eligible = group[
                (group["nested_fraction"] <= group["nested_fraction"].median())
                & (group["shared_boundary_fraction"] <= group["shared_boundary_fraction"].median())
                & (group["prediction_span_fraction"] >= 0.65)
            ]
            role = "simple adjacent domains"
        elif chrom == "chr16":
            eligible = group[
                (group["nested_tads"] >= 1)
                & (group["shared_boundary_positions"] >= 1)
                & (group["prediction_span_fraction"] >= 0.65)
            ]
            role = "nested domains with a shared boundary"
        if eligible.empty:
            raise RuntimeError(f"No representative-region candidate satisfies the rules for {chrom}")
        if chrom == "chr16":
            # Among nearly equivalent medium-complexity windows, prefer a
            # representative prediction density rather than a crowded panel.
            min_distance = eligible["target_distance"].min()
            eligible = eligible[
                eligible["target_distance"] <= min_distance + 0.02
            ].copy()
            eligible["density_distance"] = (
                eligible["n_tads"] - group["n_tads"].median()
            ).abs()
            ordering = ["density_distance", "target_distance", "valid_bin_fraction", "start"]
            ascending = [True, True, False, True]
        else:
            ordering = ["target_distance", "valid_bin_fraction", "start"]
            ascending = [True, False, True]
        choice = eligible.sort_values(ordering, ascending=ascending).iloc[0].copy()
        choice["selection_role"] = role
        selected.append(choice)
    return pd.DataFrame(selected).reset_index(drop=True)


def write_tad_links(
    tads: pd.DataFrame,
    chrom: str,
    start: int,
    end: int,
    output: Path,
) -> int:
    subset = tads[
        (tads["chrom"] == chrom)
        & (tads["left"] >= start)
        & (tads["right_exclusive"] <= end)
    ]
    lines = []
    for row in subset.itertuples(index=False):
        # Use the exact final BED boundaries; no visualization-only snapping.
        lines.append(
            f"{chrom}\t{row.left}\t{row.left + 1}\t"
            f"{chrom}\t{row.right_boundary - 1}\t{row.right_boundary}\n"
        )
    output.write_text("".join(lines))
    return len(lines)


def write_chiapet_links(
    bedpe: Path,
    chrom: str,
    start: int,
    end: int,
    output: Path,
) -> int:
    lines = []
    with bedpe.open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 6 or fields[0] != chrom or fields[3] != chrom:
                continue
            left_start, left_end = int(fields[1]), int(fields[2])
            right_start, right_end = int(fields[4]), int(fields[5])
            if left_start >= start and right_end <= end:
                lines.append(
                    f"{chrom}\t{left_start}\t{left_end}\t"
                    f"{chrom}\t{right_start}\t{right_end}\n"
                )
    output.write_text("".join(lines))
    return len(lines)


def write_tracks_ini(
    output: Path,
    mcool: Path,
    tad_links: Path,
    chiapet_links: Path,
    ctcf: Path,
    h3k27: Path,
    h3k36: Path,
) -> None:
    output.write_text(
        f"""[x-axis]
where = top

[hic]
file = {mcool}::/resolutions/{RESOLUTION}
title = 4000M Hi-C
height = 6.0
file_type = hic_matrix
colormap = RdYlBu_r
transform = no
min_value = 0
max_value = 0.006
depth = 490000
rasterize = true

[tad_triangles]
file = {tad_links}
title =
file_type = links
links_type = triangles
color = black
line_width = 1.3
alpha = 1
overlay_previous = share-y

[chia_pet]
file = {chiapet_links}
title = CTCF ChIA-PET
height = 1.2
file_type = links
links_type = arcs
use_middle = true
orientation = inverted
color = blue
line_width = 0.5
alpha = 0.8
compact_arcs_level = 1

[ctcf_signal]
file = {ctcf}
title = CTCF
height = 1.3
file_type = bigwig
color = green
min_value = 0
max_value = 5
number_of_bins = 360
summary_method = mean
alpha = 0.92
show_data_range = true

[h3k27me3]
file = {h3k27}
title = H3K27me3
height = 1.3
file_type = bigwig
color = blue
min_value = 0
max_value = 5
number_of_bins = 450
summary_method = mean
alpha = 0.92
show_data_range = true

[h3k36me3]
file = {h3k36}
title = H3K36me3
height = 1.3
file_type = bigwig
color = red
min_value = 0
max_value = 5
number_of_bins = 450
summary_method = mean
alpha = 0.92
show_data_range = true
"""
    )


def render_region(
    row: pd.Series,
    tads: pd.DataFrame,
    args: argparse.Namespace,
) -> Path:
    chrom, start, end = str(row["chrom"]), int(row["start"]), int(row["end"])
    asset_dir = args.asset_dir / f"{chrom}_{start}_{end}"
    asset_dir.mkdir(parents=True, exist_ok=True)
    tad_links = asset_dir / "contexttad.links"
    chiapet_links = asset_dir / "chiapet.links"
    n_tads = write_tad_links(tads, chrom, start, end, tad_links)
    n_chiapet = write_chiapet_links(args.chiapet, chrom, start, end, chiapet_links)
    tracks_ini = asset_dir / "tracks.ini"
    write_tracks_ini(
        tracks_ini,
        args.mcool,
        tad_links,
        chiapet_links,
        args.ctcf,
        args.h3k27,
        args.h3k36,
    )
    prefix = asset_dir / f"{chrom}_{start}_{end}"
    region = f"{chrom}:{start}-{end}"
    base = [
        str(args.pygenometracks_bin),
        "--tracks",
        str(tracks_ini),
        "--region",
        region,
        "--width",
        str(args.width),
        "--dpi",
        str(args.dpi),
        "--trackLabelHAlign",
        "left",
        "--trackLabelFraction",
        "0.08",
    ]
    subprocess.run(base + ["--outFileName", str(prefix.with_suffix(".pdf"))], check=True)
    subprocess.run(base + ["--outFileName", str(prefix.with_suffix(".png"))], check=True)
    print(f"[render] {region}: {n_tads} TADs, {n_chiapet} ChIA-PET links", flush=True)
    return prefix.with_suffix(".png")


def combine_images(paths: list[Path], output_prefix: Path, dpi: int) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    gap = 34
    label_margin = 90
    height = sum(image.height for image in images) + gap * (len(images) - 1) + label_margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 52)
    except OSError:
        font = ImageFont.load_default()
    y = label_margin
    for index, image in enumerate(images):
        x = (width - image.width) // 2
        canvas.paste(image, (x, y))
        draw.text((20, y + 8), f"({chr(ord('a') + index)})", fill="black", font=font)
        y += image.height + gap
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_prefix.with_suffix(".png"), dpi=(dpi, dpi))
    canvas.save(output_prefix.with_suffix(".pdf"), resolution=dpi)


def append_selected_regions_to_summary(selected: pd.DataFrame, summary_path: Path) -> None:
    marker = "\nRepresentative regions (4000M):\n"
    current = summary_path.read_text() if summary_path.exists() else "ContextTAD Figure 6 summary\n"
    current = current.split(marker, maxsplit=1)[0].rstrip()
    lines = [current, marker.strip()]
    for row in selected.itertuples(index=False):
        lines.append(
            "\t".join(
                [
                    row.chrom,
                    f"{int(row.start)}-{int(row.end)}",
                    f"role={row.selection_role}",
                    f"n_tads={int(row.n_tads)}",
                    f"nested={int(row.nested_tads)}",
                    f"shared_boundaries={int(row.shared_boundary_positions)}",
                    f"complexity={float(row.complexity_score):.6f}",
                ]
            )
        )
    summary_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select and render three reproducible ContextTAD regions."
    )
    parser.add_argument("--final-bed", type=Path, required=True)
    parser.add_argument("--mcool", type=Path, required=True)
    parser.add_argument("--chiapet", type=Path, required=True)
    parser.add_argument("--ctcf", type=Path, required=True)
    parser.add_argument("--h3k27", type=Path, required=True)
    parser.add_argument("--h3k36", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--pygenometracks-bin", default="pyGenomeTracks")
    parser.add_argument("--width", type=float, default=45.0)
    parser.add_argument("--dpi", type=int, default=350)
    args = parser.parse_args()

    for path in (args.final_bed, args.mcool, args.chiapet, args.ctcf, args.h3k27, args.h3k36):
        if not path.exists():
            raise FileNotFoundError(path)
    resolved_bin = shutil.which(args.pygenometracks_bin)
    if resolved_bin is None and Path(args.pygenometracks_bin).is_file():
        resolved_bin = str(Path(args.pygenometracks_bin).resolve())
    if resolved_bin is None:
        raise FileNotFoundError(f"pyGenomeTracks executable not found: {args.pygenometracks_bin}")
    args.pygenometracks_bin = resolved_bin
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.asset_dir.mkdir(parents=True, exist_ok=True)

    tads = load_tads(args.final_bed)
    candidates = enumerate_candidates(tads, args.mcool)
    selected = select_regions(candidates)
    candidates.to_csv(args.output_dir / "representative_region_candidates.tsv", sep="\t", index=False)
    selected.to_csv(args.output_dir / "representative_regions.tsv", sep="\t", index=False)

    paths = [render_region(row, tads, args) for _, row in selected.iterrows()]
    combine_images(paths, args.output_dir / "figS_representative_regions", args.dpi)
    append_selected_regions_to_summary(selected, args.output_dir / "summary.txt")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
