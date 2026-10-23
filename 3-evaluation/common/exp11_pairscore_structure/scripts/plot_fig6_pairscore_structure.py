#!/usr/bin/env python3
"""Generate Figure 6: pair-score quartiles versus Hi-C structural evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cooler
import cooltools
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import map_coordinates
from scipy.stats import rankdata, spearmanr


RESOLUTION = 5_000
CHROMS = ("chr15", "chr16", "chr17")
QUARTILES = ("Q1", "Q2", "Q3", "Q4")
COLORS = {
    "Q1": "#9ecae1",
    "Q2": "#6baed6",
    "Q3": "#fdae6b",
    "Q4": "#d94801",
}
def load_final_scored_tads(final_bed: Path, raw_bed: Path) -> pd.DataFrame:
    score_map: dict[tuple[str, int, int], float] = {}
    with raw_bed.open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) >= 4 and fields[0] in CHROMS:
                score_map[(fields[0], int(fields[1]), int(fields[2]))] = float(fields[3])

    rows = []
    with final_bed.open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 3 or fields[0] not in CHROMS:
                continue
            chrom = fields[0]
            left = int(fields[1])
            right_boundary = int(fields[2])
            right_exclusive = right_boundary + RESOLUTION
            key = (chrom, left, right_exclusive)
            if key not in score_map:
                raise KeyError(f"Final TAD has no exact raw-score match: {key}")
            rows.append(
                {
                    "chrom": chrom,
                    "left": left,
                    "right_boundary": right_boundary,
                    "right_exclusive": right_exclusive,
                    "pair_score": score_map[key],
                    "length_bp": right_exclusive - left,
                    "length_bins": (right_exclusive - left) // RESOLUTION,
                }
            )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["pair_score", "chrom", "left", "right_boundary"],
        kind="mergesort",
    ).reset_index(drop=True)
    q_index = np.floor(np.arange(len(frame)) * 4 / len(frame)).astype(int)
    q_index = np.clip(q_index, 0, 3)
    frame["quartile"] = np.asarray(QUARTILES, dtype=object)[q_index]
    frame["quartile_rank"] = q_index + 1
    return frame


def _mcool_group(handle: h5py.File, resolution: int) -> h5py.Group:
    key = f"resolutions/{resolution}"
    if key not in handle:
        raise KeyError(f"Missing mcool group: {key}")
    return handle[key]


def build_oe_band(
    mcool: Path,
    chrom: str,
    cache_path: Path,
    max_diag: int,
    chunk_pixels: int = 4_000_000,
) -> dict[str, np.ndarray]:
    if cache_path.exists():
        cached = np.load(cache_path)
        if int(cached["max_diag"]) >= max_diag:
            return {key: cached[key] for key in cached.files}

    print(f"[O/E] building {chrom} band (0..{max_diag} bins)", flush=True)
    with h5py.File(mcool, "r") as handle:
        group = _mcool_group(handle, RESOLUTION)
        chrom_names = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in group["chroms/name"][:]
        ]
        chrom_id = chrom_names.index(chrom)
        bins_chrom = group["bins/chrom"][:]
        selected = np.flatnonzero(bins_chrom == chrom_id)
        lo = int(selected[0])
        hi = int(selected[-1]) + 1
        n_bins = hi - lo
        weights = group["bins/weight"][lo:hi].astype(np.float64)
        valid_bins = np.isfinite(weights)
        offsets = group["indexes/bin1_offset"]
        pixel_start = int(offsets[lo])
        pixel_stop = int(offsets[hi])

        sums = np.zeros(max_diag + 1, dtype=np.float64)
        # Cooler stores only non-zero pixels. Expected contact at each genomic
        # distance must include every valid bin pair, including implicit zeros.
        counts = np.asarray(
            [
                np.count_nonzero(valid_bins[: n_bins - distance] & valid_bins[distance:])
                for distance in range(max_diag + 1)
            ],
            dtype=np.int64,
        )
        for start in range(pixel_start, pixel_stop, chunk_pixels):
            stop = min(start + chunk_pixels, pixel_stop)
            b1 = group["pixels/bin1_id"][start:stop].astype(np.int64) - lo
            b2 = group["pixels/bin2_id"][start:stop].astype(np.int64) - lo
            obs = group["pixels/count"][start:stop].astype(np.float64)
            distance = b2 - b1
            keep = (b1 >= 0) & (b2 < n_bins) & (distance >= 0) & (distance <= max_diag)
            if not np.any(keep):
                continue
            b1, b2, obs, distance = b1[keep], b2[keep], obs[keep], distance[keep]
            balanced = obs * weights[b1] * weights[b2]
            finite = np.isfinite(balanced)
            sums += np.bincount(distance[finite], weights=balanced[finite], minlength=max_diag + 1)

        expected = np.divide(
            sums,
            counts,
            out=np.full_like(sums, np.nan),
            where=counts > 0,
        )
        band = np.zeros((n_bins, max_diag + 1), dtype=np.float32)
        for start in range(pixel_start, pixel_stop, chunk_pixels):
            stop = min(start + chunk_pixels, pixel_stop)
            b1 = group["pixels/bin1_id"][start:stop].astype(np.int64) - lo
            b2 = group["pixels/bin2_id"][start:stop].astype(np.int64) - lo
            obs = group["pixels/count"][start:stop].astype(np.float64)
            distance = b2 - b1
            keep = (b1 >= 0) & (b2 < n_bins) & (distance >= 0) & (distance <= max_diag)
            if not np.any(keep):
                continue
            b1, b2, obs, distance = b1[keep], b2[keep], obs[keep], distance[keep]
            balanced = obs * weights[b1] * weights[b2]
            denom = expected[distance]
            values = np.divide(
                balanced,
                denom,
                out=np.zeros_like(balanced),
                where=np.isfinite(balanced) & np.isfinite(denom) & (denom != 0),
            )
            band[b1, distance] = values.astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        band=band,
        valid_bins=valid_bins.astype(np.uint8),
        expected=expected.astype(np.float32),
        max_diag=np.asarray(max_diag, dtype=np.int32),
    )
    return {
        "band": band,
        "valid_bins": valid_bins.astype(np.uint8),
        "expected": expected.astype(np.float32),
        "max_diag": np.asarray(max_diag, dtype=np.int32),
    }


def band_square(cache: dict[str, np.ndarray], start: int, end: int) -> np.ndarray:
    band = cache["band"]
    valid_bins = cache["valid_bins"].astype(bool)
    indices = np.arange(start, end, dtype=np.int64)
    low = np.minimum.outer(indices, indices)
    distance = np.abs(np.subtract.outer(indices, indices))
    result = band[low, distance].astype(np.float32, copy=True)
    pair_valid = valid_bins[indices][:, None] & valid_bins[indices][None, :]
    result[~pair_valid] = np.nan
    return result


def resize_nan_square(matrix: np.ndarray, size: int) -> np.ndarray:
    axis = np.linspace(0, matrix.shape[0] - 1, size, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    coords = np.vstack([yy.ravel(), xx.ravel()])
    finite = np.isfinite(matrix)
    numerator = map_coordinates(
        np.nan_to_num(matrix, nan=0.0),
        coords,
        order=1,
        mode="nearest",
    ).reshape(size, size)
    denominator = map_coordinates(
        finite.astype(np.float32),
        coords,
        order=1,
        mode="nearest",
    ).reshape(size, size)
    return np.divide(
        numerator,
        denominator,
        out=np.full((size, size), np.nan, dtype=np.float32),
        where=denominator > 0.5,
    )


def upper_band_mean(
    cache: dict[str, np.ndarray],
    start: int,
    length: int,
    ignore_diags: int,
) -> tuple[float, float]:
    band = cache["band"]
    valid_bins = cache["valid_bins"].astype(bool)
    total = 0.0
    observed = 0
    possible = 0
    for distance in range(ignore_diags + 1, length):
        left = np.arange(start, start + length - distance, dtype=np.int64)
        right = left + distance
        valid = valid_bins[left] & valid_bins[right]
        values = band[left, distance]
        total += float(values[valid].sum(dtype=np.float64))
        observed += int(valid.sum())
        possible += len(left)
    if observed == 0:
        return np.nan, 0.0
    return total / observed, observed / max(possible, 1)


def compute_oe_metrics(
    tads: pd.DataFrame,
    caches: dict[str, dict[str, np.ndarray]],
    pileup_size: int,
    flank_fraction: float,
    ignore_diags: int,
    min_valid_fraction: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    records = []
    pileup_members: dict[str, list[np.ndarray]] = {q: [] for q in QUARTILES}

    for row in tads.itertuples(index=False):
        cache = caches[row.chrom]
        n_bins = cache["band"].shape[0]
        left_bin = row.left // RESOLUTION
        right_bin = row.right_exclusive // RESOLUTION
        length = right_bin - left_bin

        inside, inside_valid = upper_band_mean(cache, left_bin, length, ignore_diags)
        flank_values = []
        flank_valids = []
        if left_bin - length >= 0:
            value, valid = upper_band_mean(cache, left_bin - length, length, ignore_diags)
            if valid >= min_valid_fraction:
                flank_values.append(value)
                flank_valids.append(valid)
        if right_bin + length <= n_bins:
            value, valid = upper_band_mean(cache, right_bin, length, ignore_diags)
            if valid >= min_valid_fraction:
                flank_values.append(value)
                flank_valids.append(valid)
        background = float(np.nanmean(flank_values)) if flank_values else np.nan
        contrast = inside - background if np.isfinite(inside) and np.isfinite(background) else np.nan

        flank = max(1, int(round(length * flank_fraction)))
        ext_start = left_bin - flank
        ext_end = right_bin + flank
        pileup_ok = ext_start >= 0 and ext_end <= n_bins
        if pileup_ok:
            square = band_square(cache, ext_start, ext_end)
            valid_fraction = float(np.isfinite(square).mean())
            pileup_ok = valid_fraction >= min_valid_fraction
            if pileup_ok:
                pileup_members[row.quartile].append(resize_nan_square(square, pileup_size))
        else:
            valid_fraction = 0.0

        records.append(
            {
                "mean_oe_inside": inside,
                "mean_oe_flank": background,
                "structural_contrast": contrast,
                "inside_valid_fraction": inside_valid,
                "flank_valid_fraction": float(np.nanmean(flank_valids)) if flank_valids else 0.0,
                "pileup_valid_fraction": valid_fraction,
                "pileup_included": bool(pileup_ok),
            }
        )

    result = pd.concat([tads.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    aggregate = {}
    for quartile in QUARTILES:
        matrices = pileup_members[quartile]
        aggregate[quartile] = np.nanmean(np.stack(matrices), axis=0)
        result.loc[result["quartile"] == quartile, "pileup_group_n"] = len(matrices)
    return result, aggregate


def load_or_compute_insulation(
    mcool: Path,
    cache_path: Path,
    window_bp: int,
) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path, sep="\t")
    clr = cooler.Cooler(f"{mcool}::/resolutions/{RESOLUTION}")
    view = pd.DataFrame(
        {
            "chrom": list(CHROMS),
            "start": [0] * len(CHROMS),
            "end": [int(clr.chromsizes[c]) for c in CHROMS],
            "name": list(CHROMS),
        }
    )
    table = cooltools.insulation(
        clr,
        [window_bp],
        view_df=view,
        ignore_diags=2,
        min_frac_valid_pixels=0.66,
        append_raw_scores=True,
        nproc=1,
        verbose=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache_path, sep="\t", index=False)
    return table


def profile_ci(values: np.ndarray, seed: int = 42, n_boot: int = 500):
    mean = np.nanmean(values, axis=0)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, values.shape[1]), dtype=np.float32)
    for index in range(n_boot):
        sampled = rng.integers(0, values.shape[0], values.shape[0])
        boots[index] = np.nanmean(values[sampled], axis=0)
    low, high = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return mean, low, high


def compute_insulation_profiles(
    tads: pd.DataFrame,
    insulation: pd.DataFrame,
    window_bp: int,
    radius_bp: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_col = f"log2_insulation_score_{window_bp}"
    if score_col not in insulation.columns:
        raise KeyError(f"Missing insulation column {score_col}; found {list(insulation.columns)}")
    radius_bins = radius_bp // RESOLUTION
    offsets = np.arange(-radius_bins, radius_bins + 1, dtype=np.int32)
    tracks: dict[str, np.ndarray] = {}
    for chrom in CHROMS:
        subset = insulation.loc[insulation["chrom"] == chrom].sort_values("start")
        tracks[chrom] = subset[score_col].to_numpy(dtype=np.float32)

    left_profiles = []
    right_profiles = []
    profile_valid = []
    left_centers = []
    right_centers = []
    for row in tads.itertuples(index=False):
        track = tracks[row.chrom]
        left = row.left // RESOLUTION
        right = row.right_boundary // RESOLUTION
        left_idx = left + offsets
        right_idx = right + offsets
        ok = (
            left_idx.min() >= 0
            and left_idx.max() < len(track)
            and right_idx.min() >= 0
            and right_idx.max() < len(track)
        )
        if ok:
            left_profile = track[left_idx]
            right_profile = track[right_idx]
            ok = np.isfinite(left_profile).mean() >= 0.8 and np.isfinite(right_profile).mean() >= 0.8
        if ok:
            left_profiles.append(left_profile)
            right_profiles.append(right_profile)
            profile_valid.append(True)
            center = radius_bins
            left_centers.append(float(np.nanmean(left_profile[center - 2 : center + 3])))
            right_centers.append(float(np.nanmean(right_profile[center - 2 : center + 3])))
        else:
            left_profiles.append(np.full(len(offsets), np.nan, dtype=np.float32))
            right_profiles.append(np.full(len(offsets), np.nan, dtype=np.float32))
            profile_valid.append(False)
            left_centers.append(np.nan)
            right_centers.append(np.nan)

    output = tads.copy()
    output["insulation_profile_included"] = profile_valid
    output["left_insulation_center"] = left_centers
    output["right_insulation_center"] = right_centers
    left_array = np.stack(left_profiles)
    right_array = np.stack(right_profiles)

    rows = []
    for quartile in QUARTILES:
        select = (output["quartile"].to_numpy() == quartile) & output["insulation_profile_included"].to_numpy()
        for boundary, array in (("left", left_array), ("right", right_array)):
            mean, low, high = profile_ci(array[select], seed=42 + QUARTILES.index(quartile))
            for offset, avg, lo, hi in zip(offsets * RESOLUTION // 1000, mean, low, high):
                rows.append(
                    {
                        "quartile": quartile,
                        "boundary": boundary,
                        "offset_kb": int(offset),
                        "mean": float(avg),
                        "ci_low": float(lo),
                        "ci_high": float(hi),
                        "n": int(select.sum()),
                    }
                )
    return output, pd.DataFrame(rows)


def draw_figure(
    tads: pd.DataFrame,
    pileups: dict[str, np.ndarray],
    profiles: pd.DataFrame,
    output_prefix: Path,
    dpi: int,
) -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )
    fig = plt.figure(figsize=(15.2, 6.25), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        4,
        height_ratios=[1.0, 0.12, 0.85],
        hspace=0.10,
        wspace=0.22,
    )

    pooled = np.concatenate([matrix[np.isfinite(matrix) & (matrix > 0)] for matrix in pileups.values()])
    # Match the established coolpuppy visual language while retaining one
    # shared scale across quartiles for direct comparison.
    vmax = max(float(np.nanmax(pooled)), 1.0)
    vmin = 1.0 / vmax
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps["coolwarm"].copy()
    cmap.set_bad((0.98, 0.98, 0.98))

    heat_axes = []
    for index, quartile in enumerate(QUARTILES):
        ax = fig.add_subplot(grid[0, index])
        heat_axes.append(ax)
        ax.imshow(
            pileups[quartile],
            cmap=cmap,
            norm=norm,
            origin="upper",
            interpolation="nearest",
            rasterized=True,
        )
        subset = tads[tads["quartile"] == quartile]
        ax.set_title(f"{quartile}  (n={int(subset['pileup_included'].sum())})", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#333333")
    boundary_colors = {"left": "#008b8b", "right": "#ef6c45"}
    profile_axes = []
    y_low = float(profiles["ci_low"].min())
    y_high = float(profiles["ci_high"].max())
    y_pad = max((y_high - y_low) * 0.08, 0.01)
    for index, quartile in enumerate(QUARTILES):
        ax = fig.add_subplot(grid[2, index])
        profile_axes.append(ax)
        for boundary, label in (("left", "Left boundary"), ("right", "Right boundary")):
            values = profiles[
                (profiles["quartile"] == quartile) & (profiles["boundary"] == boundary)
            ].sort_values("offset_kb")
            x = values["offset_kb"].to_numpy()
            ax.fill_between(
                x,
                values["ci_low"],
                values["ci_high"],
                color=boundary_colors[boundary],
                alpha=0.10,
                linewidth=0,
            )
            ax.plot(
                x,
                values["mean"],
                color=boundary_colors[boundary],
                linewidth=2.0,
                label=label,
            )
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(quartile, pad=5, fontweight="bold")
        ax.set_xlabel("Relative position (kb)")
        ax.set_ylim(y_low - y_pad, y_high + y_pad)
        ax.grid(axis="y", linestyle="--", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    profile_axes[0].set_ylabel("Insulation score (log2)")
    for ax in profile_axes[1:]:
        ax.tick_params(labelleft=False)
    handles, labels = profile_axes[0].get_legend_handles_labels()
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.set_axis_off()
    legend_ax.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="center",
        bbox_to_anchor=(0.5, 0.50),
        handlelength=2.6,
        columnspacing=2.4,
    )

    fig.subplots_adjust(left=0.060, right=0.985, top=0.93, bottom=0.10)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_structural_contrast(tads: pd.DataFrame, output_prefix: Path, dpi: int) -> None:
    """Export structural contrast as a standalone supplementary figure."""
    sns.set_theme(style="white", context="paper")
    fig, ax_c = plt.subplots(figsize=(5.5, 4.4))
    valid = tads[np.isfinite(tads["structural_contrast"])].copy()
    sns.violinplot(
        data=valid,
        x="quartile",
        y="structural_contrast",
        hue="quartile",
        order=QUARTILES,
        palette=COLORS,
        legend=False,
        inner=None,
        cut=0,
        linewidth=0.7,
        density_norm="width",
        ax=ax_c,
    )
    sns.boxplot(
        data=valid,
        x="quartile",
        y="structural_contrast",
        order=QUARTILES,
        width=0.22,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.9},
        whiskerprops={"color": "black", "linewidth": 0.9},
        capprops={"color": "black", "linewidth": 0.9},
        medianprops={"color": "#b2182b", "linewidth": 1.7},
        ax=ax_c,
    )
    rho, pvalue = spearmanr(valid["pair_score"], valid["structural_contrast"])
    ax_c.text(
        0.04,
        0.96,
        f"Spearman ρ = {rho:.2f}\nP = {pvalue:.2e}",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    ax_c.set_xlabel("Pair-score quartile")
    ax_c.set_ylabel("Within-domain O/E − flank O/E")
    ax_c.grid(axis="y", linestyle="--", alpha=0.22)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_summary(tads: pd.DataFrame, output: Path) -> None:
    lines = [
        "ContextTAD Figure 6 summary",
        f"Total final 4000M TADs: {len(tads)}",
        "",
        "Quartile statistics:",
    ]
    for quartile in QUARTILES:
        subset = tads[tads["quartile"] == quartile]
        lines.append(
            "\t".join(
                [
                    quartile,
                    f"n={len(subset)}",
                    f"score_min={subset['pair_score'].min():.6f}",
                    f"score_max={subset['pair_score'].max():.6f}",
                    f"contrast_median={subset['structural_contrast'].median():.6f}",
                    f"pileup_n={int(subset['pileup_included'].sum())}",
                    f"insulation_n={int(subset['insulation_profile_included'].sum())}",
                ]
            )
        )
    valid = tads[np.isfinite(tads["structural_contrast"])]
    rho, pvalue = spearmanr(valid["pair_score"], valid["structural_contrast"])
    lines.extend(["", f"Spearman pair_score vs structural_contrast: rho={rho:.6f}, p={pvalue:.6e}"])

    log_length = np.log(valid["length_bp"].to_numpy(dtype=float))
    score = rankdata(valid["pair_score"].to_numpy(dtype=float))
    contrast = rankdata(valid["structural_contrast"].to_numpy(dtype=float))
    score_resid = score - np.polyval(np.polyfit(log_length, score, 1), log_length)
    contrast_resid = contrast - np.polyval(np.polyfit(log_length, contrast, 1), log_length)
    partial_rho, partial_p = spearmanr(score_resid, contrast_resid)
    lines.append(
        "Length-adjusted rank association: "
        f"rho={partial_rho:.6f}, p={partial_p:.6e}"
    )
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relate ContextTAD pair scores to 4000M Hi-C structural evidence."
    )
    parser.add_argument("--final-bed", type=Path, required=True)
    parser.add_argument("--raw-bed", type=Path, required=True)
    parser.add_argument("--mcool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--pileup-size", type=int, default=99)
    parser.add_argument("--flank-fraction", type=float, default=1.0)
    parser.add_argument("--max-diag", type=int, default=1100)
    parser.add_argument("--insulation-window", type=int, default=50_000)
    parser.add_argument("--profile-radius", type=int, default=200_000)
    parser.add_argument("--dpi", type=int, default=350)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.mcool, args.final_bed, args.raw_bed):
        if not path.exists():
            raise FileNotFoundError(path)

    tads = load_final_scored_tads(args.final_bed, args.raw_bed)
    caches = {
        chrom: build_oe_band(
            args.mcool,
            chrom,
            args.cache_dir / f"oe_band_4000M_{chrom}.npz",
            args.max_diag,
        )
        for chrom in CHROMS
    }
    tads, pileups = compute_oe_metrics(
        tads,
        caches,
        pileup_size=args.pileup_size,
        flank_fraction=args.flank_fraction,
        ignore_diags=2,
        min_valid_fraction=0.66,
    )
    np.savez(args.cache_dir / "quartile_pileups_4000M.npz", **pileups)

    insulation = load_or_compute_insulation(
        args.mcool,
        args.cache_dir / f"insulation_4000M_{args.insulation_window // 1000}kb.tsv.gz",
        args.insulation_window,
    )
    profile_metrics, profiles = compute_insulation_profiles(
        tads,
        insulation,
        args.insulation_window,
        args.profile_radius,
    )
    for column in (
        "insulation_profile_included",
        "left_insulation_center",
        "right_insulation_center",
    ):
        tads[column] = profile_metrics[column]

    tads.to_csv(args.output_dir / "pairscore_quartiles.tsv", sep="\t", index=False)
    profiles.to_csv(args.output_dir / "insulation_profiles.tsv", sep="\t", index=False)
    quartile_summary = (
        tads.groupby("quartile", observed=True)
        .agg(
            n=("pair_score", "size"),
            score_min=("pair_score", "min"),
            score_median=("pair_score", "median"),
            score_max=("pair_score", "max"),
            structural_contrast_median=("structural_contrast", "median"),
            pileup_n=("pileup_included", "sum"),
            insulation_n=("insulation_profile_included", "sum"),
        )
        .reindex(QUARTILES)
    )
    quartile_summary.to_csv(args.output_dir / "quartile_summary.tsv", sep="\t")

    draw_figure(
        tads,
        pileups,
        profiles,
        args.output_dir / "fig6_pairscore_structure",
        args.dpi,
    )
    draw_structural_contrast(
        tads,
        args.output_dir / "figS_pairscore_structural_contrast",
        args.dpi,
    )
    write_summary(tads, args.output_dir / "summary.txt")
    print(quartile_summary.to_string())


if __name__ == "__main__":
    main()
