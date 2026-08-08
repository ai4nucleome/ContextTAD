"""
RobusTAD-style structural scoring utilities.

This module provides:
- Delta / DeltaNB core score functions
- getTADScores helper for local TAD score evaluation

Interval convention:
- 0-based half-open [left, right)

Implementation notes:
- Numba JIT acceleration is used for inner loops.
- Complexity is quadratic in interval span.
"""

import numpy as np
from numba import jit
import functools

eps = np.finfo(float).eps

cached = {}
counts = {'hitcache': 0, 'misscache': 0}


def getTADScores(data, offset, left, right, minRatio=1.1):
    """
    TAD
    
    :
        data: Hi-C
        offset: 
        left: TAD (0-based)
        right: TAD (half-open,right)
        minRatio: win/loss
    
    :
        scores: ([-5, +5])
    """
    scores = []
    for i in range(-5, 6):
        _left = left + i
        _right = right + i
        if _left - offset < 0 or _right - offset > data.shape[0]:
            continue
        score = Delta(data=data, offset=offset, left=_left, right=_right, 
                      minRatio=minRatio, mask=None)
        scores.append(score)
    return scores


@jit(nopython=True, cache=True)
def count_wins_losses_fast(within_sorted, cross_sorted, minRatio):
    """
    winloss
    
    RobusTAD:
    - win: ratio >= minRatio (within/cross >= minRatio => cross <= within/minRatio)
    - loss: ratio <= 1/minRatio (within/cross <= 1/minRatio => cross >= within*minRatio)
    
    :
        within_sorted: TAD
        cross_sorted: 
        minRatio: win/loss
    
    :
        (win, loss): 
    """
    n_within = len(within_sorted)
    n_cross = len(cross_sorted)
    
    win = 0
    loss = 0
    j_win = 0
    j_loss = 0
    
    for i in range(n_within):
        w = within_sorted[i]
        # win: within/cross >= minRatio => cross <= within/minRatio
        threshold_win = w / minRatio
        # loss: within/cross <= 1/minRatio => cross >= within*minRatio
        threshold_loss = w * minRatio
        
        # win:cross <= threshold_win
        while j_win < n_cross and cross_sorted[j_win] <= threshold_win:
            j_win += 1
        win += j_win
        
        # loss:cross >= threshold_loss
        while j_loss < n_cross and cross_sorted[j_loss] < threshold_loss:
            j_loss += 1
        loss += (n_cross - j_loss)
    
    return win, loss


@jit(nopython=True, cache=True)
def DeltaNB(diags, left, right, w, minRatio=1.1):
    """
    TADDelta(numba)
    
    0-based half-open [left, right)
    RobusTAD
    
    :
        diags: ,diags[i,j]ij
        left: TAD()
        right: TAD(,half-open)
        w: TAD (= right - left)
        minRatio: "win"
    
    :
        Tuple[np.ndarray, int]:
            - scores: 
            - N: 
    """
    scores = np.zeros(w) * np.nan
    N = 0
    
    for diag in range(1, w):
        # :RobusTAD,
        
        # crossIF1: 
        start1 = max(0, left - diag)
        crossIF1 = diags[start1:left, diag]
        
        # crossIF2: 
        start2 = right - diag
        crossIF2 = diags[start2:right, diag]
        
        crossIF = np.concatenate((crossIF1, crossIF2))
        
        # NaNeps
        cross_valid = np.empty(len(crossIF), dtype=np.float64)
        cross_count = 0
        for v in crossIF:
            if not np.isnan(v):
                cross_valid[cross_count] = v + eps
                cross_count += 1
        
        if cross_count == 0:
            continue
        
        # TAD
        withinIF = diags[left:right - diag, diag]
        
        within_valid = np.empty(len(withinIF), dtype=np.float64)
        within_count = 0
        for v in withinIF:
            if not np.isnan(v):
                within_valid[within_count] = v + eps
                within_count += 1
        
        if within_count == 0:
            continue
        
        # win/loss
        within_sorted = np.sort(within_valid[:within_count])
        cross_sorted = np.sort(cross_valid[:cross_count])
        win, loss = count_wins_losses_fast(within_sorted, cross_sorted, minRatio)
        
        n1 = within_count
        n2 = cross_count
        scale = n1 * n2
        
        if scale > 0:
            score = (win - loss) / scale * (n1 + n2)
            N += (n1 + n2)
            scores[diag] = score
    
    return scores[1:], N


def cacheDelta(func):
    """,TAD"""
    @functools.wraps(func)
    def lazzyDelta(**kwargs):
        tad = (kwargs['left'], kwargs['right'])
        if kwargs['mask'] is not None:
            mask = tuple(dict.fromkeys(kwargs['mask']))
        else:
            mask = ()
        if tad in cached and mask in cached[tad]:
            counts['hitcache'] += 1
            return cached[tad][mask]
        val = func(**kwargs)
        if tad not in cached:
            counts['misscache'] += 1
            cached[tad] = {}
        cached[tad][mask] = val
        return val
    return lazzyDelta


def Delta(data, offset, left, right, minRatio=1.1, mask=None):
    """
    TADDelta
    
    0-based half-open [left, right)
    
    :
        data: Hi-C(obsLarge)
        offset: ()
        left: TAD (0-based)
        right: TAD (half-open)
        minRatio: "win"
        mask: 
    
    :
        float: TADDelta,TAD
    """
    data = data.copy()
    s = data.shape[0]
    
    # mask:NaN
    if mask is not None:
        for i in range(len(mask)):
            l, r = mask[i]
            # :maskhalf-open
            local_l = max(0, l - offset)
            local_r = r - offset
            data[local_l:local_r, local_l:local_r] = np.nan
            corner_l = max(0, l - offset - 4)
            corner_r = min(s, r - offset + 4)
            data[corner_l:l - offset + 4, corner_l:corner_r] = np.nan
    
    left = left - offset
    right = right - offset
    
    # TAD(half-open, = right - left)
    w = right - left
    
    if w <= 0:
        return 0.0
    
    diags = np.zeros((s, w + 2)) * np.nan
    
    for j in range(min(w + 2, s)):
        diagj = np.diagonal(data, j)
        diags[:len(diagj), j] = diagj
    
    # DeltaNB
    scores, N = DeltaNB(diags, left, right, w, minRatio)
    
    return np.nansum(np.asarray(scores)) / (N + eps)

