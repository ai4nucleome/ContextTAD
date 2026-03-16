"""
(chr1-chr22) Hi-C , E1/E2 track(Plotly HTML)
 (Dynamic Thresholding) 
"""

import argparse
import os
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import cooler
import cooltools
import bioframe
from packaging import version
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#  cooltools 
if version.parse(cooltools.__version__) < version.parse('0.5.4'):
    raise AssertionError("Script relies on cooltools version 0.5.4 or higher")

def parse_args():
    parser = argparse.ArgumentParser(description=" Hi-C  + E1/E2 Tracks")
    parser.add_argument("--mcool", type=str, required=True, help="mcool ")
    parser.add_argument("--resolution", type=int, default=10000, help=" (bp)")
    parser.add_argument("--fasta", type=str, default="./hg38.fa", help=" fasta")
    parser.add_argument("--output-dir", type=str, default="./output_html", help="")
    parser.add_argument("--max_display_bins", type=int, default=3000, help="Bin")
    return parser.parse_args()

def get_gc_content(clr, fasta_path, cache_file):
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, sep='\t')
    if not os.path.isfile(fasta_path):
        subprocess.call('wget --progress=bar:force:noscroll https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.fa.gz', shell=True)
        subprocess.call('gunzip hg38.fa.gz', shell=True)
        fasta_path = './hg38.fa'
    bins = clr.bins()[:]
    hg38_genome = bioframe.load_fasta(fasta_path)
    gc_cov = bioframe.frac_gc(bins[['chrom', 'start', 'end']], hg38_genome)
    gc_cov.to_csv(cache_file, index=False, sep='\t')
    return gc_cov

def get_eigenvectors(clr, gc_cov, chromosomes, cache_file):
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, sep='\t')
    view_df = pd.DataFrame({'chrom': chromosomes, 'start': 0, 'end': [clr.chromsizes[c] for c in chromosomes], 'name': chromosomes})
    cis_eigs = cooltools.eigs_cis(clr, gc_cov, view_df=view_df, n_eigs=3)
    eig_df = cis_eigs[1]
    eig_df.to_csv(cache_file, index=False, sep='\t')
    return eig_df

# : zmin
def calculate_dynamic_zmin(matrix):
    """
    
    :
    1. NaN
    2.  Log10
    3.  1.5% 
    """
    # (0NaN)
    valid_mask = (matrix > 0) & (~np.isnan(matrix))
    valid_data = matrix[valid_mask]
    
    if len(valid_data) == 0:
        return -5.5 # Fallback
    
    log_data = np.log10(valid_data)
    
    # 1.5% ,(),
    #  -5.5  chr17 
    z_min_dynamic = np.percentile(log_data, 0.1)
    
    if z_min_dynamic < -7.0: 
        z_min_dynamic = -7.0
        
    return z_min_dynamic - 1.0

def plot_interactive_heatmap(chrom, matrix, e1, e2, output_file, resolution, dynamic_zmin):
    """
     dynamic_zmin 
    """
    x_coords = np.arange(len(e1))
    
    # 0 -> NaN,  log10(0)
    matrix_with_nan = np.where(matrix == 0, np.nan, matrix)
    log_matrix = np.log10(matrix_with_nan)

    # zmax  -1.0 ( 0.1),
    z_max_val = -1.0
    # zmin 
    z_min_val = dynamic_zmin

    # ---  (Fall) ---
    fall_colorscale = [
        [0.0, 'white'], 
        [0.05, '#fbfddf'], 
        [0.2, '#f6e496'], 
        [0.45, '#eeb464'], 
        [0.70, '#d14310'], 
        [0.90, '#800000'], 
        [1.0, 'black']
    ]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.005,
        row_heights=[0.08, 0.08, 0.84], 
        # subplot_titles=(f"{chrom} E1", f"E2", "")
    )

    # --- E1 ---
    e1_pos = np.where(e1 > 0, e1, 0)
    e1_neg = np.where(e1 < 0, e1, 0)
    fig.add_trace(go.Scatter(x=x_coords, y=e1_pos, fill='tozeroy', mode='lines', 
                             line=dict(color='#d73027', width=0), name='E1 A', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_coords, y=e1_neg, fill='tozeroy', mode='lines', 
                             line=dict(color='#4575b4', width=0), name='E1 B', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_coords, y=e1, mode='lines', 
                             line=dict(color='black', width=0.5), name='E1', showlegend=True), row=1, col=1)

    # --- E2 ---
    e2_pos = np.where(e2 > 0, e2, 0)
    e2_neg = np.where(e2 < 0, e2, 0)
    fig.add_trace(go.Scatter(x=x_coords, y=e2_pos, fill='tozeroy', mode='lines', 
                             line=dict(color='#fdae61', width=0), name='E2 (+)', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_coords, y=e2_neg, fill='tozeroy', mode='lines', 
                             line=dict(color='#abd9e9', width=0), name='E2 (-)', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_coords, y=e2, mode='lines', 
                             line=dict(color='grey', width=0.5), name='E2', showlegend=True), row=2, col=1)

    # --- Heatmap ---
    fig.add_trace(go.Heatmap(
        z=log_matrix,
        colorscale=fall_colorscale,
        zmin=z_min_val,
        zmax=z_max_val,
        colorbar=dict(
            title='log10(Freq)', 
            len=0.5, y=0.4, thickness=15,
            tickvals=[-4, -3, -2, -1],
            ticktext=['104', '103', '102', '0.1']
        ),
        name='Contact Map',
        hoverongaps=False 
    ), row=3, col=1)

    # --- Layout ---
    total_width = 1000
    total_height = 1200 

    fig.update_layout(
        title_text=f"{chrom} ({resolution//1000}kb) - Auto zmin={z_min_val:.2f}",
        width=total_width,
        height=total_height,
        template='plotly_white',
        plot_bgcolor='white',
        hovermode="x unified",
        margin=dict(l=50, r=50, t=80, b=50)
    )

    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, row=3, col=1, title=f"{chrom} bins", showgrid=False, zeroline=False)
    # fig.update_xaxes(showgrid=False, row=3, col=1, title="Genomic Bins")
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    
    e1_max = np.nanmax(np.abs(e1)) if not np.all(np.isnan(e1)) else 1
    fig.update_yaxes(range=[-e1_max, e1_max], row=1, col=1, title="E1", showgrid=True)
    e2_max = np.nanmax(np.abs(e2)) if not np.all(np.isnan(e2)) else 1
    fig.update_yaxes(range=[-e2_max, e2_max], row=2, col=1, title="E2", showgrid=True)

    fig.write_html(output_file)
    print(f"    : {output_file} ( zmin: {z_min_val:.3f})")

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "intermediate_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("Step 1:  mcool ...")
    filepath = f'{args.mcool}::resolutions/{args.resolution}'
    clr = cooler.Cooler(filepath)
    
    print("Step 2:  GC Content...")
    gc_cache_path = cache_dir / f'hg38_gc_cov_{args.resolution//1000}kb.tsv'
    gc_cov = get_gc_content(clr, args.fasta, gc_cache_path)
    
    print("Step 3: ...")
    chromosomes = [c for c in clr.chromnames if c.startswith('chr') and c[3:].isdigit() and int(c[3:]) <= 22]
    # chromosomes = ['chr17']  # debug
    
    print("Step 4 & 5: ...")
    eig_cache_path = cache_dir / f'eigenvectors.tsv'
    eig_df = get_eigenvectors(clr, gc_cov, chromosomes, eig_cache_path)

    print("Step 6: ...")
    for chrom in chromosomes:
        print(f"   {chrom}...")
        
        chrom_eigs = eig_df[eig_df['chrom'] == chrom]
        e1_data = chrom_eigs['E1'].values
        e2_data = chrom_eigs['E2'].values

        #  E1
        if not np.all(np.isnan(e1_data)):
            mean_e1 = np.nanmean(e1_data)
            e1_data = e1_data - mean_e1
            e1_data = np.nan_to_num(e1_data, nan=0.0)
        else:
            e1_data = np.nan_to_num(e1_data, nan=0.0)

        #  E2
        if not np.all(np.isnan(e2_data)):
            mean_e2 = np.nanmean(e2_data)
            e2_data = e2_data - mean_e2
            e2_data = np.nan_to_num(e2_data, nan=0.0)
        else:
            e2_data = np.nan_to_num(e2_data, nan=0.0)
        
        # , zmin,
        full_matrix = clr.matrix(balance=True).fetch(chrom)
        full_matrix = np.nan_to_num(full_matrix)
        
        auto_zmin = calculate_dynamic_zmin(full_matrix)
        
        n_bins = full_matrix.shape[0]
        
        if n_bins > args.max_display_bins:
            scale_factor = n_bins // args.max_display_bins
            plot_matrix = full_matrix[::scale_factor, ::scale_factor]
            plot_e1 = e1_data[::scale_factor]
            plot_e2 = e2_data[::scale_factor]
            print(f"    [Info] {chrom}  {scale_factor} (zmin={auto_zmin:.2f})")
        else:
            plot_matrix = full_matrix
            plot_e1 = e1_data
            plot_e2 = e2_data
            print(f"    [Info] {chrom}  (zmin={auto_zmin:.2f})")
            
        out_html = output_dir / f'{chrom}_E1_E2_heatmap.html'
        
        #  zmin
        plot_interactive_heatmap(chrom, plot_matrix, plot_e1, plot_e2, out_html, args.resolution, auto_zmin)

    print("\n!")

if __name__ == "__main__":
    main()