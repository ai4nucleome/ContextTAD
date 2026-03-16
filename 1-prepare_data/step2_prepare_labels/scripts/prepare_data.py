#!/usr/bin/env python
"""
Build per-window training/inference arrays from multi-coverage GM12878 Hi-C data.

This script loads mcool matrices, computes obs/OE channels, aligns annotation tracks,
and writes .npy data plus label files under TAD_DATA_DIR.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --only-4000M
    python scripts/prepare_data.py --skip-4000M
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cooler
from scipy.sparse import coo_matrix
from tqdm import tqdm
from pathlib import Path
import time

# Local structural score implementation used during preprocessing.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from robustadScore import Delta


MCOOL_TEMPLATE = os.environ.get(
    "MCOOL_TEMPLATE",
    "/home/weicai/projectnvme/TADAnno_final/0-data/2_eval_tads_data/mcool_data/GM12878/Rao2014/4DNFIXP4QG5B_Rao2014_GM12878_frac{frac}.mcool",
)
RESOLUTION = 5000
OBS_SIZE = 400     # Window size in bins.
STEP_SIZE = 200    # Sliding step in bins.
MAX_TADS = 50      # Max number of TADs per window (with padding).

# Coverage tag and mcool fraction token used in MCOOL_TEMPLATE.
COVERAGES = [
    ("4000M", "1"),
    ("2000M", "0.5"),
    ("1000M", "0.25"),
    ("500M",  "0.125"),
    ("250M",  "0.0625"),
    ("125M",  "0.03125"),
    ("62_5M", "0.015625"),
]

CHROMS = [f"chr{i}" for i in range(1, 23)]

DATA_DIR = Path(os.environ.get("TAD_DATA_DIR", "/home/weicai/projectnvme/TADAnno_final/0-data/1_dp_train_infer_data"))
ANNO_DIR = DATA_DIR / "annotations"
LABELS_DIR = DATA_DIR / "labels"
OE_LARGE_DIR = DATA_DIR / "oeLarge"

# Optional path for reusing existing 4000M intermediate outputs.
EXISTING_4000M_DIR = Path(
    os.environ.get(
        "EXISTING_4000M_DIR",
        "/home/weicai/projectnvme/TADAnno_final/1-prepare_data/step2_process_data/outputs/lwc_gm12878",
    )
)

TAD_BED = ANNO_DIR / "4DNFIXP4QG5B_Rao2014_GM12878_frac1_TAD_hq_cleaned.bed"
CTCF_BED = ANNO_DIR / "gm12878_ctcf.bed"
ATAC_BED = ANNO_DIR / "gm12878_atac.bed"
EIGS_TSV = ANNO_DIR / "gm12878_eigenvector.fillnan.tsv"


#  ( process_data.py)

def upper_coo_to_symm(row, col, data, N):
    """"""
    shape = (N, N)
    sparse_matrix = coo_matrix((data, (row, col)), shape=shape)
    symm = sparse_matrix + sparse_matrix.T
    diag_val = symm.diagonal(0) / 2
    symm = symm.tocsr()
    symm.setdiag(diag_val)
    return symm


def process_cool_file(coolfile, chrom):
    """
     Hi-C 

    Returns:
        obsMat: ICE balanced  (CSR)
        oeMat: O/E  (CSR)
        N:  bin 
    """
    extent = coolfile.extent(chrom)
    N = extent[1] - extent[0]

    ccdata = coolfile.matrix(balance=True, sparse=True, as_pixels=True).fetch(chrom)
    ccdata['bin1_id'] -= extent[0]
    ccdata['bin2_id'] -= extent[0]

    #  O/E
    ccdata['distance'] = ccdata['bin2_id'] - ccdata['bin1_id']
    d_means = ccdata.groupby('distance')['balanced'].transform('mean')
    ccdata['oe'] = ccdata['balanced'] / d_means
    ccdata['oe'] = ccdata['oe'].fillna(0)
    ccdata['balanced'] = ccdata['balanced'].fillna(0)

    obsMat = upper_coo_to_symm(
        ccdata['bin1_id'].to_numpy(),
        ccdata['bin2_id'].to_numpy(),
        ccdata['balanced'].to_numpy(), N
    )
    oeMat = upper_coo_to_symm(
        ccdata['bin1_id'].to_numpy(),
        ccdata['bin2_id'].to_numpy(),
        ccdata['oe'].to_numpy(), N
    )

    return obsMat, oeMat, N


def extract_matrices(mat, N, center_start, center_end, obs_size, large_size):
    """
    

    Returns:
        small: [obs_size, obs_size] 
        large: [~large_size, ~large_size]  ( padding)
    """
    margin = (large_size - obs_size) // 2

    large_start = max(0, center_start - margin)
    large_end = min(N, center_end + margin)

    large_mat = mat[large_start:large_end, large_start:large_end].toarray()

    # Padding ()
    if center_start - margin < 0:
        pad = margin - center_start
        large_mat = np.pad(large_mat, ((pad, 0), (pad, 0)), mode='constant')
    elif center_end + margin > N:
        pad = center_end + margin - N
        large_mat = np.pad(large_mat, ((0, pad), (0, pad)), mode='constant')

    small_mat = large_mat[margin:margin + obs_size, margin:margin + obs_size]

    #  NaN
    large_mat = np.nan_to_num(large_mat, 0)
    small_mat = np.nan_to_num(small_mat, 0)

    return small_mat.astype(np.float32), large_mat.astype(np.float32)


def extract_small_matrix(mat, N, center_start, center_end):
    """ (, )"""
    small = mat[center_start:center_end, center_start:center_end].toarray()
    small = np.nan_to_num(small, 0)
    return small.astype(np.float32)



def read_tad_dp_file(tad_dp_path):
    """
     TAD_dp.txt,

    TAD_dp.txt : left right ( [left, right))
    : [(left, right), ...]  [left, right]
    """
    tads = []
    if not os.path.exists(tad_dp_path):
        return tads
    with open(tad_dp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                left = int(parts[0])
                right = int(parts[1]) - 1  #   
                if right > left:
                    tads.append((left, right))
    return tads


def compute_nesting_depth(tad, all_tads):
    """ TAD  ( TAD )"""
    left, right = tad
    depth = 0
    for other_left, other_right in all_tads:
        if other_left < left and other_right > right:
            depth += 1
        elif other_left <= left and other_right > right and (other_left, other_right) != (left, right):
            depth += 1
        elif other_left < left and other_right >= right and (other_left, other_right) != (left, right):
            depth += 1
    return depth


def compute_boundary_scores(oe_large, tads_half_open, margin):
    """
     RobusTAD Delta  TAD  score

    Args:
        oe_large: oeLarge  [~1210, ~1210]
        tads_half_open: TAD  (, )
        margin: oeLarge 

    Returns:
        tad_scores: {(left, right): score}  TAD  score
    """
    tad_scores = {}
    for left, right in tads_half_open:
        #  oeLarge 
        oe_left = margin + left
        oe_right = margin + right

        if oe_left < 0 or oe_right > oe_large.shape[0]:
            tad_scores[(left, right)] = 0.0
            continue

        try:
            score = Delta(
                data=oe_large,
                offset=0,
                left=oe_left,
                right=oe_right,
                minRatio=1.1,
                mask=None
            )
            tad_scores[(left, right)] = max(score, 0.0)  #  0
        except Exception:
            tad_scores[(left, right)] = 0.0

    return tad_scores


def generate_labels(tads_closed, tads_half_open, oe_large, margin, obs_size):
    """
    

    Args:
        tads_closed: [(left, right), ...] 
        tads_half_open: [(left, right), ...]  ( Delta )
        oe_large: oeLarge 
        margin: oeLarge margin
        obs_size:  (400)

    Returns:
        labels: [400, 6] float32
        tads_array: [MAX_TADS, 2] int32
    """
    n = obs_size

    #  RobusTAD Delta Score
    tad_scores = compute_boundary_scores(oe_large, tads_half_open, margin)

    boundary_score = np.zeros(n, dtype=np.float32)
    boundary_mask = np.zeros(n, dtype=np.float32)
    nesting_depth = np.zeros(n, dtype=np.float32)

    for tad_closed, tad_ho in zip(tads_closed, tads_half_open):
        left_c, right_c = tad_closed  # 
        depth = compute_nesting_depth(tad_closed, tads_closed)
        score = tad_scores.get(tad_ho, 0.0)

        if 0 <= left_c < n:
            boundary_mask[left_c] = 1.0
            boundary_score[left_c] = max(boundary_score[left_c], score)
            nesting_depth[left_c] = max(nesting_depth[left_c], depth)

        #  (, right )
        if 0 <= right_c < n:
            boundary_mask[right_c] = 1.0
            boundary_score[right_c] = max(boundary_score[right_c], score)
            nesting_depth[right_c] = max(nesting_depth[right_c], depth)

    #  labels [400, 6]
    # : boundary_score, boundary_mask, nesting_depth, ctcf, atac, eigenvector
    # CTCF/ATAC/eig  (0), 
    labels = np.zeros((n, 6), dtype=np.float32)
    labels[:, 0] = boundary_score
    labels[:, 1] = boundary_mask
    labels[:, 2] = nesting_depth

    # TAD  (, padding -1)
    tads_array = np.full((MAX_TADS, 2), -1, dtype=np.int32)
    for i, (left, right) in enumerate(tads_closed[:MAX_TADS]):
        tads_array[i] = [left, right]

    return labels, tads_array


# 1D 

def load_annotation_files():
    """ 1D """
    files = {}
    for name, path in [("CTCF", CTCF_BED), ("ATAC", ATAC_BED), ("eigs", EIGS_TSV)]:
        df = pd.read_csv(path, sep='\t', header=None, comment='#')
        files[name] = df
    return files


def map_annotations_to_window(files_dict, chrom, start_bin, obs_size, resol=RESOLUTION):
    """
     1D 

    Returns:
        ctcf: [obs_size] float32
        atac: [obs_size] float32
        eig: [obs_size] float32
    """
    ctcf = np.zeros(obs_size, dtype=np.float32)
    atac = np.zeros(obs_size, dtype=np.float32)
    eig = np.zeros(obs_size, dtype=np.float32)

    for name in ["CTCF", "ATAC"]:
        df = files_dict[name].copy()
        df[[1, 2]] = df[[1, 2]] // resol - start_bin
        df[3] = df[3] / 1000
        mask = ((df[0] == chrom) | (df[0] == chrom[3:])) & (df[1] >= 0) & (df[1] < obs_size)
        filtered = df[mask]
        arr = ctcf if name == "CTCF" else atac
        for idx, v in filtered[[1, 3]].values:
            arr[int(idx)] = v

    # Eigenvector
    df = files_dict["eigs"].copy()
    df[[1, 2]] = df[[1, 2]] // resol - start_bin
    mask = ((df[0] == chrom) | (df[0] == chrom[3:])) & (df[1] >= 0) & (df[1] < obs_size)
    filtered = df[mask]
    if len(filtered) > 0:
        # chr1, chr6, chr12  E2 (col 13),  E1 (col 12)
        col = 13 if chrom in ['chr1', 'chr6', 'chr12'] else 12
        if col < filtered.shape[1]:
            for _, row in filtered.iterrows():
                idx = int(row[1])
                if 0 <= idx < obs_size:
                    eig[idx] = row[col]

    return ctcf, atac, eig



def get_window_positions(N, obs_size=OBS_SIZE, step=STEP_SIZE):
    """ ( process_data.py )"""
    positions = []
    for start in range(0, N, step):
        if start + obs_size > N:
            start = N - obs_size
        if start not in [p for p in positions]:
            positions.append(start)
    return positions


def process_4000M(chroms=CHROMS):
    """
     4000M :  + oeLarge + 

    ,  RobusTAD Delta Score
    """
    print("=" * 60)
    print(" 4000M  ( + oeLarge)")
    print("=" * 60)

    coverage_name = "4000M"
    frac = "1"
    coolpath = MCOOL_TEMPLATE.format(frac=frac) + "::/resolutions/5000"

    print(f"Opening: {coolpath}")
    c = cooler.Cooler(coolpath)

    print("Loading annotation files...")
    anno_files = load_annotation_files()

    obs_large_size = OBS_SIZE * 3 + 10  # 1210
    margin = (obs_large_size - OBS_SIZE) // 2  # 405

    total_windows = 0
    total_tads = 0

    for chrom in chroms:
        print(f"\n--- {chrom} ---")
        obsMat, oeMat, N = process_cool_file(c, chrom)
        print(f"  N = {N} bins")

        (DATA_DIR / coverage_name / chrom).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / chrom).mkdir(parents=True, exist_ok=True)
        (OE_LARGE_DIR / chrom).mkdir(parents=True, exist_ok=True)

        positions = get_window_positions(N)
        print(f"  {len(positions)} windows")

        for start in tqdm(positions, desc=f"  {chrom}"):
            end = start + OBS_SIZE
            window_name = f"{chrom}_{start}"

            obs_small = extract_small_matrix(obsMat, N, start, end)
            oe_small, oe_large = extract_matrices(oeMat, N, start, end, OBS_SIZE, obs_large_size)

            #  matrices.npy: [2, 400, 400] (ch0=obs, ch1=oe)
            matrices = np.stack([obs_small, oe_small], axis=0)
            np.save(DATA_DIR / coverage_name / chrom / f"{window_name}.npy", matrices)

            #  oeLarge.npy
            np.save(OE_LARGE_DIR / chrom / f"{window_name}.npy", oe_large)

            #  TAD_dp.txt ()
            tad_dp_path = EXISTING_4000M_DIR / chrom / window_name / "TAD_dp.txt"
            tads_half_open = []
            if tad_dp_path.exists():
                with open(tad_dp_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 2:
                                tads_half_open.append((int(parts[0]), int(parts[1])))

            tads_closed = [(l, r - 1) for l, r in tads_half_open if r - 1 > l]

            #  ( RobusTAD Delta Score)
            labels, tads_array = generate_labels(
                tads_closed, tads_half_open, oe_large, margin, OBS_SIZE
            )

            #  1D  ( 3, 4, 5)
            ctcf, atac, eig = map_annotations_to_window(anno_files, chrom, start, OBS_SIZE)
            labels[:, 3] = ctcf
            labels[:, 4] = atac
            labels[:, 5] = eig

            np.save(LABELS_DIR / chrom / f"{window_name}_labels.npy", labels)
            np.save(LABELS_DIR / chrom / f"{window_name}_tads.npy", tads_array)

            total_windows += 1
            total_tads += len(tads_closed)

    print(f"\n4000M : {total_windows} windows, {total_tads} TADs")


def process_other_coverage(coverage_name, frac, chroms=CHROMS):
    """
     4000M :  obs + oe 
     4000M 
    """
    print(f"\n{'=' * 60}")
    print(f" {coverage_name} (frac={frac})")
    print("=" * 60)

    coolpath = MCOOL_TEMPLATE.format(frac=frac) + "::/resolutions/5000"
    print(f"Opening: {coolpath}")
    c = cooler.Cooler(coolpath)

    total_windows = 0

    for chrom in chroms:
        print(f"\n--- {chrom} ---")
        obsMat, oeMat, N = process_cool_file(c, chrom)

        (DATA_DIR / coverage_name / chrom).mkdir(parents=True, exist_ok=True)

        positions = get_window_positions(N)

        for start in tqdm(positions, desc=f"  {chrom}"):
            end = start + OBS_SIZE
            window_name = f"{chrom}_{start}"

            label_path = LABELS_DIR / chrom / f"{window_name}_labels.npy"
            if not label_path.exists():
                continue  # 

            obs_small = extract_small_matrix(obsMat, N, start, end)
            oe_small = extract_small_matrix(oeMat, N, start, end)

            #  [2, 400, 400]
            matrices = np.stack([obs_small, oe_small], axis=0)
            np.save(DATA_DIR / coverage_name / chrom / f"{window_name}.npy", matrices)

            total_windows += 1

    print(f"\n{coverage_name} : {total_windows} windows")



def verify_data():
    """"""
    print("\n" + "=" * 60)
    print("")
    print("=" * 60)

    for coverage_name, _ in COVERAGES:
        coverage_dir = DATA_DIR / coverage_name
        npy_count = len(list(coverage_dir.rglob("*.npy")))
        print(f"  {coverage_name:8s}: {npy_count:5d} ")

    label_count = len(list(LABELS_DIR.rglob("*_labels.npy")))
    tad_count = len(list(LABELS_DIR.rglob("*_tads.npy")))
    oelarge_count = len(list(OE_LARGE_DIR.rglob("*.npy")))
    print(f"  {'labels':8s}: {label_count:5d} ")
    print(f"  {'tads':8s}: {tad_count:5d} TAD ")
    print(f"  {'oeLarge':8s}: {oelarge_count:5d} oeLarge ")

    print("\n:")
    sample_path = DATA_DIR / "4000M" / "chr1" / "chr1_0.npy"
    if sample_path.exists():
        m = np.load(sample_path)
        print(f"   shape: {m.shape}, dtype: {m.dtype}")
        print(f"  obs range: [{m[0].min():.4f}, {m[0].max():.4f}]")
        print(f"  oe  range: [{m[1].min():.4f}, {m[1].max():.4f}]")

    sample_label = LABELS_DIR / "chr1" / "chr1_0_labels.npy"
    if sample_label.exists():
        l = np.load(sample_label)
        print(f"   shape: {l.shape}, dtype: {l.dtype}")
        print(f"  boundary_score range: [{l[:,0].min():.4f}, {l[:,0].max():.4f}]")
        print(f"  boundary_mask  sum  : {l[:,1].sum():.0f}")
        print(f"  nesting_depth  max  : {l[:,2].max():.0f}")

    sample_tad = LABELS_DIR / "chr1" / "chr1_0_tads.npy"
    if sample_tad.exists():
        t = np.load(sample_tad)
        valid = t[t[:, 0] >= 0]
        print(f"  TAD  shape: {t.shape},  TAD: {len(valid)}")
        if len(valid) > 0:
            print(f"   TAD (): {valid[:3].tolist()}")

    sample_oel = OE_LARGE_DIR / "chr1" / "chr1_0.npy"
    if sample_oel.exists():
        ol = np.load(sample_oel)
        print(f"  oeLarge shape: {ol.shape}, dtype: {ol.dtype}")

    print("\n:")
    for d in [DATA_DIR]:
        total = sum(f.stat().st_size for f in d.rglob("*.npy"))
        print(f"  {d.name}: {total / 1e9:.2f} GB")



def main():
    parser = argparse.ArgumentParser(description='TAD-Polaris ')
    parser.add_argument('--only-4000M', action='store_true', help=' 4000M')
    parser.add_argument('--skip-4000M', action='store_true', help=' 4000M')
    parser.add_argument('--chroms', type=str, default=None,
                       help=', ,  chr1,chr2')
    parser.add_argument('--verify-only', action='store_true', help='')
    args = parser.parse_args()

    if args.verify_only:
        verify_data()
        return

    chroms = args.chroms.split(',') if args.chroms else CHROMS

    start_time = time.time()

    # Step 1:  4000M ()
    if not args.skip_4000M:
        process_4000M(chroms)

    # Step 2: 
    if not args.only_4000M:
        for coverage_name, frac in COVERAGES:
            if coverage_name == "4000M":
                continue  # 
            process_other_coverage(coverage_name, frac, chroms)

    elapsed = time.time() - start_time
    print(f"\n: {elapsed / 60:.1f} ")

    verify_data()


if __name__ == '__main__':
    main()
