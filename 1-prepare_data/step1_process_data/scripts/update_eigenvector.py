"""
eigenvector,E1_v3E2_v3
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def update_eigenvector():
    original_file = Path(
        os.environ.get(
            "EIGEN_ORIGINAL_FILE",
            "/home/weicai/projectnvme/TADAnno_final/0-data/1_dp_train_infer_data/annotations/gm12878_eigenvector.fillnan.tsv",
        )
    )
    eigen_file = Path(
        os.environ.get(
            "EIGEN_TRACK_FILE",
            str(Path(__file__).resolve().parent.parent / "outputs" / "check_eigen" / "intermediate_results" / "eigenvectors.tsv"),
        )
    )
    output_dir = Path(
        os.environ.get(
            "EIGEN_OUTPUT_DIR",
            "/home/weicai/projectnvme/TADAnno_final/0-data/1_dp_train_infer_data/annotations",
        )
    )
    output_file = output_dir / original_file.name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("...")
    original_df = pd.read_csv(original_file, sep='\t')
    
    print("eigenvectors...")
    # eigenvectors
    eigen_df = pd.read_csv(eigen_file, sep='\t')
    
    print("...")
    chromosomes = eigen_df['chrom'].unique()
    
    # E1_v3E2_v3
    e1_v3_list = []
    e2_v3_list = []
    match_keys = []  #  (chrom, start, end)
    
    for chrom in chromosomes:
        print(f"   {chrom}...")
        
        # eigen
        chrom_eigs = eigen_df[eigen_df['chrom'] == chrom].copy()
        e1_data = chrom_eigs['E1'].values
        e2_data = chrom_eigs['E2'].values
        
        #  E1
        if not np.all(np.isnan(e1_data)):
            mean_e1 = np.nanmean(e1_data)
            e1_data_centered = e1_data - mean_e1
            e1_data_centered = np.nan_to_num(e1_data_centered, nan=0.0)
        else:
            e1_data_centered = np.nan_to_num(e1_data, nan=0.0)
        
        #  E2
        if not np.all(np.isnan(e2_data)):
            mean_e2 = np.nanmean(e2_data)
            e2_data_centered = e2_data - mean_e2
            e2_data_centered = np.nan_to_num(e2_data_centered, nan=0.0)
        else:
            e2_data_centered = np.nan_to_num(e2_data, nan=0.0)
        
        chrom_eigs['E1_v3'] = e1_data_centered
        chrom_eigs['E2_v3'] = e2_data_centered
        
        for idx in chrom_eigs.index:
            row = chrom_eigs.loc[idx]
            match_keys.append((row['chrom'], row['start'], row['end']))
            e1_v3_list.append(row['E1_v3'])
            e2_v3_list.append(row['E2_v3'])
    
    # DataFrame
    match_df = pd.DataFrame({
        'chrom': [k[0] for k in match_keys],
        'start': [k[1] for k in match_keys],
        'end': [k[2] for k in match_keys],
        'E1_v3': e1_v3_list,
        'E2_v3': e2_v3_list
    })
    
    print("...")
    # #chromchrom()
    original_df_work = original_df.copy()
    original_df_work['chrom'] = original_df_work['#chrom']
    
    merged_df = original_df_work.merge(
        match_df[['chrom', 'start', 'end', 'E1_v3', 'E2_v3']],
        on=['chrom', 'start', 'end'],
        how='left'
    )
    
    # chrom
    merged_df = merged_df.drop(columns=['chrom'])
    
    # (fillna)
    matched_count = merged_df['E1_v3'].notna().sum()
    
    # NaN0.0()
    merged_df['E1_v3'] = merged_df['E1_v3'].fillna(0.0)
    merged_df['E2_v3'] = merged_df['E2_v3'].fillna(0.0)
    
    # : + E1_v3 + E2_v3
    original_cols = list(original_df.columns)
    final_cols = original_cols + ['E1_v3', 'E2_v3']
    merged_df = merged_df[final_cols]
    
    print("...")
    merged_df.to_csv(output_file, sep='\t', index=False)
    
    print(f"!: {output_file}")
    print(f": E1_v3, E2_v3")
    print(f": {len(merged_df)}")
    print(f"eigenvectors.tsv: {matched_count}")

if __name__ == "__main__":
    update_eigenvector()
