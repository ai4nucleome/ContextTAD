"""
Window-level preprocessing for Hi-C model inputs and initial TAD candidates.

This script reads a GM12878 mcool file and paired annotation BED/TSV files,
then generates per-window matrices and features used by DP refinement.

Coordinate notes:
- BED intervals are 0-based half-open: [start, end)
- TAD intervals are represented in bin space and exported with half-open semantics
- Outputs are aligned to RobusTAD-compatible indexing
"""

import os
import json
import argparse
from pathlib import Path
import cooler
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import coo_matrix
np.set_printoptions(suppress=True) 

# Load and process Hi-C data
def extract_matrices(obsMat, oeMat, N, center_start, center_end, obsSize, obsLargeSize):
    """
    RobusTADdomain score,3
    
    Args:
        obsMat:  Hi-C 
        oeMat:  O/E 
        N: 
        center_start: 
        center_end: 
        obsSize: 
        obsLargeSize: 
    """
    
    # --- 1.  ---
    # (obsSize),(padding), robusTAD domain score
    margin = (obsLargeSize-obsSize)//2  # 

    #  (temp),(0),0
    temp = center_start - margin
    if temp < 0:
        large_start = 0
    else:
        large_start = temp
    temp = center_end + margin

    # , N, N
    if temp > N:
        large_end = N
    else:
        large_end = temp

    # --- 2.  ---
    #  obsMat ,(.toarray())
    obsLarge = obsMat[large_start:large_end,large_start:large_end].toarray()
    
    # --- 3.  (Padding) ---
    # ,, 0 
    if center_start - margin < 0:
        obsLarge = np.pad(obsLarge, pad_width=((margin-center_start, 0), (margin-center_start, 0)), mode='constant', constant_values=0)
    elif center_end + margin > N:
        obsLarge = np.pad(obsLarge, pad_width=((0, margin+center_end-N), (0, margin+center_end-N)), mode='constant', constant_values=0)
    
    # --- 4.  ---
    obs = obsLarge[margin:-margin, margin:-margin]
    #  O/E 
    oe = oeMat[center_start:center_end, center_start:center_end].toarray()
    # (NaN)0,
    obs = np.nan_to_num(obs, 0)
    oe = np.nan_to_num(oe, 0)
    
    return obsLarge, obs, oe

# Process bed files
def process_bed_files(files_dict, chrom, start, end, obsSize, resol=5000):
    """
    "" CTCFTAD ,
    
    : 0-based half-open  [start, end), BED 
    
    Args:
        files_dict: , CTCFTADdf
        chrom: 
        start: ()
        end: (, half-open)
        obsSize:  (= end - start)
        resol: 
    Return:
        data: 
            "TAD": TAD  [start, end) (0-based half-open)
            "CTCF", "ATAC", "CAGE" : obsSize 
            "E1"  "E2": compartment ( bin ),
    """
    data = {}  # 
    
    for name, df_ in files_dict.items():
        df = df_.copy() # , . , load 

        # ""( 100259bp), 0-based half-open 
        # 1.  (//resol) -> (bin)
        # 2.  (-start) -> 
        # :BED 0-based half-open,
        df[[1,2]] = df[[1,2]] // resol - start 
        df[3] = df[3] / 1000  # TODO, , score,signalValue
        if name == 'TAD':
            #  TAD
            #  half-open :TAD  end  obsSize( end )
            mask = ((df[0] == chrom) | (df[0] == chrom[3:])) & (df[1] >= 0) & (df[2] <= (end-start))
            filtered_df = df[mask]
            
            # print(f"{name}: {len(filtered_df)}")
            # print(filtered_df.head())
            # print(filtered_df.tail())
            
            #  TAD (half-open:[start, end))
            filtered_df = filtered_df.sort_values(by=[1,2])
            data[name] = filtered_df[[1, 2]].values  # start, end,  N*2 
        else:
            #  CTCF, ATAC 
            # ,df[1] >= 0  df[1] < xx, start
            # TODO: 
            mask = ((df[0] == chrom) | (df[0] == chrom[3:])) & (df[1] >= 0) & (df[1] < (end-start))
            filtered_df = df[mask]
            
            # print(f"{name}: {len(filtered_df)}")
            # print(filtered_df.head())
            # print(filtered_df.tail())
            
            if name == 'eigs':  #  Compartment 
                #  E1  E2
                # chr1, chr6, chr12  filtered_df[13].values, filtered_df[12].values
                if chrom in ['chr1', 'chr6', 'chr12']:
                    data["E2"] = filtered_df[13].values
                else:
                    data["E1"] = filtered_df[12].values
            else:  # CTCF, RAD21, etc.
                data[name] = np.zeros(obsSize)
                for idx, v in filtered_df[[1, 3]].values:  # start, peak value
                    data[name][int(idx)] = v
    return data

def generate_data(obsMat, oeMat, N, chrom, center_start, saveDir, obsSize, files_dict):
    """
    , 
    Args:
        obsMat:  Hi-C 
        oeMat:  O/E 
        N: 
        chrom: 
        center_start: 
        saveDir: 
        obsSize: 
        files_dict: , CTCFTADdf
    """
    obsLargeSize = obsSize * 3 + 10  # 3RobusTADdomain score, 10

    os.makedirs(saveDir, exist_ok=True)
    
    center_end = center_start + obsSize  # 
    
    # Extract matrices
    obsLarge, obs, oe = extract_matrices(obsMat, oeMat, N, center_start, center_end, obsSize, obsLargeSize)

    # Save matrices
    np.savetxt(f'{saveDir}/obsLarge.txt', obsLarge)
    np.savetxt(f'{saveDir}/obs.txt', obs)
    np.savetxt(f'{saveDir}/oe.txt', oe)

    # Process bed files
    #  CTCF,ATAC,E1/E2,TAD 
    data = process_bed_files(files_dict, chrom, center_start, center_end, obsSize)
    
    #  E1  E2( process_bed_files  chrom )
    # chr1, chr6, chr12  E1, E2
    eigen_key = 'E2' if chrom in ['chr1', 'chr6', 'chr12'] else 'E1'
    
    # print(len(data['CTCF'].tolist()),len(data['RAD21'].tolist()),len(data['SMC3'].tolist()),len(data['H3K27ac'].tolist()),len(data['H3K27me3'].tolist()),len(data['E1'].tolist()),len(data['E2'].tolist()),len(data['E3'].tolist()))
    # Combine all data into a DataFrame
    # combined_data = pd.DataFrame({
    #     '1': data['CTCF'].tolist(),
    #     '2': data['RAD21'].tolist(),
    #     '3': data['SMC3'].tolist(),
    #     '4': data['H3K27ac'].tolist(),
    #     '5': data['H3K27me3'].tolist(),
    #     '6': data['E1'].tolist(),
    #     '7': data['E2'].tolist(),
    #     '8': data['E3'].tolist(),
    # comment_row = pd.DataFrame({
    #     '1': ['l'],
    #     '2': ['l'],
    #     '3': ['l'],
    #     '4': ['l'],
    #     '5': ['l'],
    #     '6': ['fl'],
    #     '7': ['fl'],
    #     '8': ['fl'],
    #  DataFrame(linearAnno)
    # : 1  CTCF
    #  2  ATAC
    #  3  E1  E2()
    #  bin
    # :, CAGE 
    combined_data = pd.DataFrame({
        'CTCF': data['CTCF'].tolist(),
        'ATAC': data['ATAC'].tolist(),
        eigen_key: data[eigen_key].tolist(),
    })
    comment_row = pd.DataFrame({
        'CTCF': ['l'],
        'ATAC': ['l'],
        eigen_key: ['fl'],
    })
    combined_data = pd.concat([comment_row, combined_data]).reset_index(drop=True)
    combined_data.head()
    # Save combined data
    combined_data.to_csv(f'{saveDir}/linearAnno.csv', index=False, sep=',')
    #  TAD  [start,end]  TAD.txt,
    np.savetxt(f'{saveDir}/TAD.txt', data['TAD'], fmt='%d', delimiter=' ')
    # print(f'{saveDir.split("/")[-1]} done...')

def upperCoo2symm(row,col,data,N):
    """
     Hi-C , NxN ()
    Args:
        row: 
        col: 
        data: 
        N: 
    """
    # row = np.array([0,1])
    # col = np.array([1,2])
    # data = np.array([5,8])
    # M = coo_matrix((data,(row,col)), shape=(3,3)).toarray()
    # # M:
    # # [[0,5,0],
    # #  [0,0,8],
    # #  [0,0,0]]
    shape=(N,N)
    sparse_matrix = coo_matrix((data, (row, col)), shape=shape)

    # Hi-C :M[i,j] = M[j,i]
    # sparse_matrix, 0,
    symm = sparse_matrix + sparse_matrix.T
    # 2,
    diagVal = symm.diagonal(0)/2
    #  CSR ,
    symm = symm.tocsr()
    symm.setdiag(diagVal)
    return symm

def processcoolfile(coolfile, cchrom):
    """
     Hi-C ,  cool  obsMat  oeMat
    Args:
        coolfile: cooler 
        cchrom: 
    """
    extent = coolfile.extent(cchrom) 
    N = extent[1] - extent[0]
    #  cool  bin  index
    #  extent = (100000, 120000),  bin  100000  120000
    # N = 120000 - 100000 = 20000  bin
    
    ccdata = coolfile.matrix(balance=True, sparse=True, as_pixels=True).fetch(cchrom)
    #  cool 
    # balance=True: ICE normalization
    # sparse=True: 
    # as_pixels=True:  pixels ,
    #  table,,
    # ccdata['balanced'] = ccdata['balanced'].fillna(0)

    ccdata['bin1_id'] -= extent[0]
    ccdata['bin2_id'] -= extent[0]
    #  bin1_id, bin2_id , 0  N-1 

    # ---  Observed / Expected (O/E) ---
    ccdata['distance'] = ccdata['bin2_id'] - ccdata['bin1_id']

    #  (Expected)
    # transform('mean'):  pandas ,
    # df = pd.DataFrame({
    #     'distance':[1,1,2,2],
    #     'balanced':[10,30,20,40]
    # m = df.groupby('distance')['balanced'].transform('mean')
    # # distance=1 -> mean=20; distance=2 -> mean=30
    # # m = [20,20,30,30]
    d_means = ccdata.groupby('distance')['balanced'].transform('mean')

    # !!!! O/E (observed / expected),expected 
    ccdata['oe'] = ccdata['balanced'] / d_means
    #  NAN  0
    ccdata['oe'] = ccdata['oe'].fillna(0)
    #  balanced  NaN  0
    ccdata['balanced'] = ccdata['balanced'].fillna(0)

    # obsMat: NxN  ICE normalized,  pixel  Hic 
    # oeMat: NxN,  pixel  Hic-O/E 
    obsMat = upperCoo2symm(ccdata['bin1_id'].to_numpy(), ccdata['bin2_id'].to_numpy(), ccdata['balanced'].to_numpy(), N)
    oeMat = upperCoo2symm(ccdata['bin1_id'].to_numpy(), ccdata['bin2_id'].to_numpy(), ccdata['oe'].to_numpy(), N)
    
    return obsMat, oeMat, N

def main():
    default_coolpath = os.environ.get(
        "COOLPATH",
        "/home/weicai/projectnvme/TADAnno_final/0-data/2_eval_tads_data/mcool_data/GM12878/Rao2014/4DNFIXP4QG5B_Rao2014_GM12878_frac1.mcool::/resolutions/5000",
    )
    default_savedir = os.environ.get(
        "SAVEDIR",
        str(Path(__file__).resolve().parents[1] / "outputs" / "lwc_gm12878"),
    )
    anno_root = Path(
        os.environ.get(
            "PREPROCESS_EIGEN_ROOT",
            "/home/weicai/projectnvme/TADAnno_final/0-data/1_dp_train_infer_data/annotations",
        )
    )
    #  argparse 
    parser = argparse.ArgumentParser(description='Process Hi-C contact map data')
    parser.add_argument('--coolpath', type=str, 
                        default=default_coolpath,
                        help='Path to the cool file')
    parser.add_argument('--chrom', type=str, default=None, help='chromosome name')
    parser.add_argument('--size', type=int, default=400, help='heatmap size')
    parser.add_argument('--step', type=int, default=200, help='step size')
    parser.add_argument('--savedir', type=str, 
                        default=default_savedir,
                        help='output directory')
    parser.add_argument('--resol', type=int, default=5000, help='resolution')
    args = parser.parse_args()
    
    # coolpath = args.coolpath
    # chrom = args.chrom
    # size = args.size
    # step = args.step
    # savedir = args.savedir
    # resol = args.resol
    assert args.step <= args.size, print('step must be less than size')

    # data_dic ={
    #     1 : 'CTCF',
    #     2 : 'RAD21',
    #     3 : 'SMC3',
    #     4 : 'H3K27ac',
    #     5 : 'H3K27me3',
    #     6 : 'A/BE1',
    #     7 : 'A/BE2',
    #     8 : 'A/BE3',
    #  data_dic(:CTCF, ATAC, E1/E2)
    # data_dic = {
    #     'CTCF': 'CTCF',
    #     'ATAC': 'ATAC',
    #     'E1': 'E1',  # A/B compartment ( chr1, chr6, chr12)
    #     'E2': 'E2',  # A/B compartment ()
    
    os.makedirs(args.savedir, exist_ok=True)
    # with open(f'{args.savedir}/epi_num2str.json', 'w') as f:
    #     json.dump(data_dic, f, indent=4)

    #  files_dict, TAD_epis_file_path.json
    files_dict = {
        "TAD": str(anno_root / "4DNFIXP4QG5B_Rao2014_GM12878_frac1_TAD_hq_cleaned.bed"),
        "CTCF": str(anno_root / "gm12878_ctcf.bed"),
        "ATAC": str(anno_root / "gm12878_atac.bed"),
        "eigs": str(anno_root / "gm12878_eigenvector.fillnan.tsv"),
    }

    c = cooler.Cooler(args.coolpath)  # cooler mcool 

    if args.chrom is None:
        chrom = c.chromnames
    else:
        chrom = args.chrom.split(',')
        for i in range(len(chrom)):
            if 'chr' not in chrom[i]:
                chrom[i] = f'chr{chrom[i]}'
    for rmchr in ['chrMT','MT','chrM','M','Y','chrY','X','chrX']:
        if rmchr in chrom:
            chrom.remove(rmchr)  
    
    for name, file_path in files_dict.items():
        #  bed : chrom, start, end, score
        df = pd.read_csv(file_path, sep='\t', header=None, comment='#')
        files_dict[name] = df
    print(f"Load traces data done...")
    
    for cc in chrom:
        #  chroms
        obsMat, oeMat, N = processcoolfile(c, cc)
        print(f"Load {cc} obsMat & oeMat data done...")
        
        # : root_dir/chr{xx}/
        chrom_dir = f'{args.savedir}/{cc}'
        os.makedirs(chrom_dir, exist_ok=True)
        
        starts = tqdm(range(0, N, args.step))
        for start in starts:
            starts.set_description(f'Processing {cc}: {start}')
            if start + args.size > N:
                start = N - args.size  
            #  root_dir/chr{xx}/chr{xx}_{start}
            generate_data(obsMat, oeMat, N, cc, start, f'{chrom_dir}/{cc}_{start}', args.size, files_dict)

    print('All Things Done!')

if __name__ == '__main__':
    main()
