#!/usr/bin/env python3
"""Infer target/decoy tracks and select final TADs at the requested FDR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import predict_structure
import decode
from train import normalize_matrix


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_path(root: Path, dataset: str, chrom: str, window: str) -> Path:
    for path in (root / dataset / chrom / f"{window}.npy",
                 root / dataset / dataset / chrom / f"{window}.npy"):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing window matrix: {dataset}/{chrom}/{window}")


def infer(args) -> None:
    torch.set_num_threads(args.threads)
    predict_structure.set_seed(args.seed)
    device = torch.device(args.device)
    model, model_args = predict_structure.load_model(str(args.checkpoint), device)
    if model_args.get("use_obs", False):
        raise ValueError("This decoder requires an O/E-input checkpoint")
    windows = json.loads((args.data_root / "window_list.json").read_text())["windows"]["test"]
    provenance_base = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "window_list_sha256": sha256(args.data_root / "window_list.json"),
        "inference_script_sha256": sha256(Path(__file__)),
        "decoder_script_sha256": sha256(Path(decode.__file__)),
        "decoy": "near3_far5", "seed": args.seed, "dataset": args.dataset,
        "window_count": len(windows),
    }
    for chrom in args.chroms:
        selected = sorted((w for w in windows if w.startswith(chrom + "_")),
                          key=lambda w: int(w.rsplit("_", 1)[1]))
        if not selected:
            raise ValueError(f"No test windows for {args.dataset}/{chrom}")
        output = args.output_dir / "cache/tracks" / args.dataset / f"{chrom}.npz"
        metadata = output.with_suffix(".json")
        provenance = {**provenance_base, "chrom": chrom}
        if output.exists() and metadata.exists() and json.loads(metadata.read_text()) == provenance:
            print(f"[cache hit] {output}", flush=True)
            continue
        if output.exists():
            raise RuntimeError(f"Incompatible existing cache: {output}")
        first = matrix_path(args.data_root, args.dataset, chrom, selected[0])
        window_bins = np.load(first, mmap_mode="r").shape[-1]
        size = max(int(w.rsplit("_", 1)[1]) for w in selected) + window_bins
        tracks = np.zeros((4, size), dtype=np.float32)
        counts = np.zeros(size, dtype=np.float32)
        for index, window in enumerate(selected, 1):
            offset = int(window.rsplit("_", 1)[1])
            matrix = np.load(matrix_path(args.data_root, args.dataset, chrom, window),
                             mmap_mode="r")[1]
            rng = np.random.default_rng(predict_structure.stable_seed(
                f"{args.dataset}:{chrom}:{window}", args.seed))
            decoy = decode.shift_distance_bands(matrix, rng)
            batch = np.stack((normalize_matrix(matrix), normalize_matrix(decoy)))
            tensor = torch.from_numpy(batch).unsqueeze(1).to(device)
            with torch.inference_mode(), torch.amp.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                prediction = model(tensor)
            stop = min(size, offset + len(matrix))
            span = stop - offset
            tracks[0, offset:stop] += prediction["left_boundary"][0, :span].float().cpu().numpy()
            tracks[1, offset:stop] += prediction["right_boundary"][0, :span].float().cpu().numpy()
            tracks[2, offset:stop] += prediction["left_boundary"][1, :span].float().cpu().numpy()
            tracks[3, offset:stop] += prediction["right_boundary"][1, :span].float().cpu().numpy()
            counts[offset:stop] += 1
            if index % 20 == 0 or index == len(selected):
                print(f"[infer] {args.dataset} {chrom}: {index}/{len(selected)}", flush=True)
        valid = counts > 0
        tracks[:, valid] /= counts[valid]
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, left_target=tracks[0], right_target=tracks[1],
                            left_decoy=tracks[2], right_decoy=tracks[3])
        temporary.replace(output)
        metadata.write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"[saved] {output}", flush=True)


def calibrate(args) -> None:
    target, null = {}, {}
    for chrom in args.chroms:
        source = args.output_dir / "cache/tracks" / args.dataset / f"{chrom}.npz"
        with np.load(source) as z:
            target[chrom] = decode.prepare(decode.assemble(
                chrom, z["left_target"], z["right_target"], args.peak_radius))
            null[chrom] = decode.prepare(decode.assemble(
                chrom, z["left_decoy"], z["right_decoy"], args.peak_radius))
        print(f"[candidates] {args.dataset} {chrom}: "
              f"{len(target[chrom].pairs)} target, {len(null[chrom].pairs)} decoy", flush=True)
    calls, decoys, details = set(), set(), []
    for chrom in args.chroms:
        cutoff, selected, selected_null = decode.calibrate(
            target[chrom], null[chrom], args.fdr_alpha)
        calls.update(selected)
        decoys.update(selected_null)
        details.append({"dataset": args.dataset, "fdr": args.fdr_alpha,
                        "chrom": chrom, "score_cutoff": cutoff,
                        "target_calls": len(selected), "decoy_calls": len(selected_null)})
    ratio = len(decoys) / len(calls) if calls else 0.0
    if ratio > args.fdr_alpha + 1e-12:
        raise AssertionError(f"FDR violated: {ratio} > {args.fdr_alpha}")
    beds = args.output_dir / "beds"
    beds.mkdir(parents=True, exist_ok=True)
    filename = (f"GM12878_{args.dataset}_contexttad.bed"
                if args.dataset not in ("K562", "IMR90") else f"{args.dataset}_contexttad.bed")
    with (beds / filename).open("w") as handle:
        for chrom, left, right in sorted(calls):
            handle.write(f"{chrom}\t{left}\t{right}\n")
    stats = args.output_dir / "postprocess_stats"
    stats.mkdir(parents=True, exist_ok=True)
    with (stats / f"{args.dataset}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    (stats / f"{args.dataset}.json").write_text(json.dumps({
        "dataset": args.dataset, "fdr": args.fdr_alpha, "calls": len(calls),
        "decoy_calls": len(decoys), "final_null_target_ratio": ratio,
    }, indent=2) + "\n")
    print(f"[result] {args.dataset}: {len(calls)} TADs, {len(decoys)} decoys, "
          f"ratio={ratio:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage", choices=("all", "infer", "calibrate"), default="all")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chroms", nargs="+", default=["chr15", "chr16", "chr17"])
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--peak-radius", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not 0 < args.fdr_alpha < 1 or args.peak_radius < 1 or args.threads < 1:
        parser.error("FDR must be in (0,1); peak radius and threads must be positive")
    if args.stage in ("all", "infer"):
        infer(args)
    if args.stage in ("all", "calibrate"):
        calibrate(args)


if __name__ == "__main__":
    main()
