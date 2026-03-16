import numpy as np
import pandas as pd
import argparse
import sys


def eval(resol, tadfile, output):
    tadfile = pd.read_csv(tadfile, sep='\t', header=None)
    # tadfile[[1,2]]//=resol
    results = {'chrom':[],'start':[],'end':[],'score':[],'level':[]}

    chr = list(set(tadfile[0]))

    results=[]
    for _chr in chr:
        chrdata = tadfile[tadfile[0]==_chr][[0,1,2]].reset_index(drop=True)
        tads = []
        for i in range(len(chrdata)):
            tads.append((chrdata[1][i],chrdata[2][i]))
        deltas = []
        for tad in tads:
            nestTADs=[]
            for tad2 in tads:
                if tad2[0]>=tad[0] and tad2[1]<=tad[1]:
                    if tad2[0]==tad[0] and tad2[1] == tad[1]:
                        pass
                    else:
                        nestTADs.append(tad2)

            if len(nestTADs) > 0:
                _delta = 1
            else:
                _delta = 0
            deltas.append(_delta)
        results.append(np.concatenate([chrdata.to_numpy(),np.asarray(deltas)[:,None]],axis=1))
    results=np.concatenate(results,axis=0)
    results=results[results[:,-1]==1,:]
    # print(results)
    results = pd.DataFrame.from_dict(results)

    results[[0,1,2]].to_csv(output, index=False, header=False, float_format='%.4f', sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' L1+  TAD')
    parser.add_argument('--resol', type=int, default=10000, help=' [: 10000]')
    parser.add_argument('--tadfile', type=str, required=True, help=' TAD ')
    parser.add_argument('--output', type=str, required=True, help='')
    
    args = parser.parse_args()
    eval(args.resol, args.tadfile, args.output)
