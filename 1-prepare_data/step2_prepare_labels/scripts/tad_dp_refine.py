#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DP refinement for raw TAD candidates at the window level.

The refinement integrates:
- Hi-C structural consistency
- boundary support from 1D signals (CTCF/ATAC/eigenvector)
- local conflict resolution across overlapping candidate groups

Main outputs:
- refined TAD intervals per window
- metadata required by later label-building steps
"""

import numpy as np
import os
import sys
from typing import List, Tuple, Dict, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
import argparse
import logging
from itertools import product
from numba import jit, prange

warnings.filterwarnings('ignore')
# Keep NumPy output compact and deterministic.
np.set_printoptions(suppress=True)

# Coordinate conventions:
# - Input TAD.txt uses half-open intervals [left, right)
# - Internal closed-right form is [left, right_closed] with right_closed = right - 1
# - Output TAD_dp.txt is written back in half-open form

# Delta score follows RobusTAD half-open semantics [left, right).

eps = np.finfo(float).eps

_diags_cache = {}

@jit(nopython=True, cache=True)
def _binary_search_le(arr, val):
    """:arr <= val """
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= val:
            lo = mid + 1
        else:
            hi = mid
    return lo

@jit(nopython=True, cache=True)
def _binary_search_ge(arr, val):
    """:arr >= val """
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < val:
            lo = mid + 1
        else:
            hi = mid
    return len(arr) - lo

@jit(nopython=True, cache=True)
def _count_win_loss_sorted(within_sorted, cross_sorted, minRatio):
    """
    win/loss
    
    , O(n*m)  O(n*log(m))
    """
    n1 = len(within_sorted)
    n2 = len(cross_sorted)
    if n1 == 0 or n2 == 0:
        return 0, 0, n1, n2
    
    win = 0
    loss = 0
    
    # withinw:
    # - win: cross c  w/c >= minRatio,  c <= w/minRatio
    # - loss: cross c  w/c <= 1/minRatio,  c >= w*minRatio
    
    for i in range(n1):
        w = within_sorted[i]
        threshold_win = w / minRatio
        threshold_loss = w * minRatio
        
        # win: cross <= threshold_win 
        win += _binary_search_le(cross_sorted, threshold_win)
        
        # loss: cross >= threshold_loss 
        loss += _binary_search_ge(cross_sorted, threshold_loss)
    
    return win, loss, n1, n2

@jit(nopython=True, cache=True)
def _deltaNB_numba(diags, left, right, w, minRatio=1.1):
    """
    NumbaDeltaNB
    +win/loss
    """
    eps_val = 1e-16
    scores = np.empty(w - 1)
    scores[:] = np.nan
    N = 0
    
    for diag in range(1, w):
        start1 = max(0, left - diag)
        end1 = left
        start2 = right - diag
        end2 = right
        
        # cross
        cross_count = 0
        for i in range(start1, end1):
            if not np.isnan(diags[i, diag]):
                cross_count += 1
        for i in range(start2, end2):
            if not np.isnan(diags[i, diag]):
                cross_count += 1
        
        # within
        within_count = 0
        for i in range(left, right - diag):
            if not np.isnan(diags[i, diag]):
                within_count += 1
        
        n1 = within_count
        n2 = cross_count
        
        if n1 > 0 and n2 > 0:
            within_arr = np.empty(n1)
            cross_arr = np.empty(n2)
            
            # within
            idx = 0
            for i in range(left, right - diag):
                val = diags[i, diag]
                if not np.isnan(val):
                    within_arr[idx] = val + eps_val
                    idx += 1
            
            # cross
            idx = 0
            for i in range(start1, end1):
                val = diags[i, diag]
                if not np.isnan(val):
                    cross_arr[idx] = val + eps_val
                    idx += 1
            for i in range(start2, end2):
                val = diags[i, diag]
                if not np.isnan(val):
                    cross_arr[idx] = val + eps_val
                    idx += 1
            
            within_sorted = np.sort(within_arr)
            cross_sorted = np.sort(cross_arr)
            
            # win/loss
            win, loss, _, _ = _count_win_loss_sorted(within_sorted, cross_sorted, minRatio)
            
            scale = n1 * n2
            score = (win - loss) / scale * (n1 + n2)
            N += (n1 + n2)
            scores[diag - 1] = score
    
    return scores, N

def DeltaNB(diags, left, right, w, minRatio=1.1):
    """
    TADDelta() - Numba
    """
    return _deltaNB_numba(diags, left, right, w, minRatio)

_precomputed_diags = None
_precomputed_diags_info = None  # (data_id, offset)

def precompute_diags_for_matrix(data, max_diag=None):
    """
    
    
    :,Delta
    
    :
        data: np.ndarray - Hi-C
        max_diag: int - ()
    
    :
        np.ndarray - ,diags[i,j]  i  i+j 
    """
    s = data.shape[0]
    if max_diag is None:
        max_diag = s
    
    diags = np.full((s, max_diag), np.nan)
    
    for j in range(min(max_diag, s)):
        diagj = np.diagonal(data, j)
        diags[:len(diagj), j] = diagj
    
    return diags

def Delta(data, offset, left, right, minRatio=1.1, mask=None):
    """
    TADDelta() - 
    
    RobusTAD,TAD
    :0-based half-open [left, right)
    
    :,
    
    :
        data: np.ndarray - Hi-C(obsLarge)
        offset: int - ()
        left: int - TAD (0-based)
        right: int - TAD (half-open,right)
        minRatio: float - "win"(1.1)
        mask: Optional[List[Tuple[int,int]]] - (TAD)
    
    :
        float: TADDelta,TAD
    """
    global _precomputed_diags, _precomputed_diags_info
    
    s = data.shape[0]
    
    local_left = left - offset
    local_right = right - offset
    
    # TAD
    w = local_right - local_left
    
    if w <= 0:
        return 0.0
    
    # mask,()
    if mask is not None:
        data = data.copy()
        for i in range(len(mask)):
            l, r = mask[i]
            mask_l = max(0, l - offset)
            mask_r = r - offset
            data[mask_l:mask_r, mask_l:mask_r] = np.nan
            corner_l = max(0, l - offset - 4)
            corner_r = min(s, r - offset + 4)
            data[corner_l:l - offset + 4, corner_l:corner_r] = np.nan
        
        # mask
        diags = np.full((s, w + 2), np.nan)
        for j in range(min(w + 2, s)):
            diagj = np.diagonal(data, j)
            diags[:len(diagj), j] = diagj
    else:
        data_id = id(data)
        if _precomputed_diags is None or _precomputed_diags_info != (data_id, offset):
            _precomputed_diags = precompute_diags_for_matrix(data)
            _precomputed_diags_info = (data_id, offset)
        
        max_diag_needed = min(w + 2, _precomputed_diags.shape[1])
        diags = _precomputed_diags[:, :max_diag_needed]
    
    # DeltaNB
    scores, N = DeltaNB(diags, local_left, local_right, w, minRatio)
    
    return np.nansum(np.asarray(scores)) / (N + eps)

def clear_diags_cache():
    """"""
    global _precomputed_diags, _precomputed_diags_info
    _precomputed_diags = None
    _precomputed_diags_info = None


# -1e9float('-inf')
NEG_INF = -1e9 

# :SEARCH_WINDOW
SEARCH_WINDOW = 5  

# TAD
NESTED_TOLERANCE = 3 

MAX_COMBINATIONS = 100000


# CTCF - 
# TADCTCF
BONUS_CTCF_DOUBLE = 100.0

# CTCF + Eigenvector
# CTCF,Eigenvector
BONUS_CTCF_ZERO = 80.0

# CTCF + ATAC
# CTCF,ATAC
BONUS_CTCF_ATAC = 50.0

# ATAC
# TADATAC
BONUS_ATAC_DOUBLE = 30.0

# CTCF
# CTCF
BONUS_SINGLE_CTCF = 20.0

# ATAC
# ATAC
BONUS_SINGLE_ATAC = 10.0

# Corner
# TADO/E3x3
BONUS_CORNER_CENTER = 30.0

# Corner
# TADO/E3x3()
BONUS_CORNER_SIDE = 15.0

# TAD score
# TAD score
BONUS_SCORE_OPTIMAL = 50.0

# TAD score
# TAD score
PENALTY_SCORE_SUBOPTIMAL = -20.0

BONUS_ORIGINAL_POS = 5.0

# TAD:
_tad_score_cache = {}

def setup_logger(debug: bool = False) -> logging.Logger:
    """
    
    
    :
        debug: bool - 
                      True: DEBUG
                      False: INFO
    
    :
        logging.Logger: 
    """
    logger = logging.getLogger('TAD_DP_Refine')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # handler
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


@dataclass
class UniqueBoundary:
    """
    
    
    TAD,TAD(TAD)
    "",TAD
    
    :
        boundary_id: int - 
        canonical_coord: int - ()
        original_coords: List[int] - 
        candidates: List[int] - ()
        optimal_coord: Optional[int] - 
    
    :
        share_count: int - TAD()
    """
    boundary_id: int                              # ID
    canonical_coord: int                          # ()
    original_coords: List[int] = field(default_factory=list)  # 
    candidates: List[int] = field(default_factory=list)       # 
    optimal_coord: Optional[int] = None           # 
    
    @property
    def share_count(self) -> int:
        """TAD"""
        return len(self.original_coords)

@dataclass
class BoundaryMap:
    """
    
    
    ,
    TAD(NESTED_TOLERANCE),
    
    
    :
        boundaries: Dict[int, UniqueBoundary] - IDUniqueBoundary
        coord_to_boundary_id: Dict[int, int] - ID
        next_id: int - ID
    
    :
        get_or_create_boundary: ,
    """
    boundaries: Dict[int, UniqueBoundary] = field(default_factory=dict)
    coord_to_boundary_id: Dict[int, int] = field(default_factory=dict)
    next_id: int = 0
    
    def get_or_create_boundary(self, coord: int, tolerance: int = 3) -> int:
        """
        ID
        
        (<=tolerance),;
        
        
        :
            coord: int - 
            tolerance: int - (3bin)
        
        :
            int: ID
        """
        for bid, boundary in self.boundaries.items():
            if abs(coord - boundary.canonical_coord) <= tolerance:
                boundary.original_coords.append(coord)
                self.coord_to_boundary_id[coord] = bid
                return bid
        
        bid = self.next_id
        self.next_id += 1
        self.boundaries[bid] = UniqueBoundary(bid, coord, [coord])
        self.coord_to_boundary_id[coord] = bid
        return bid

@dataclass
class TADNode:
    """
    TAD
    
    TAD,TAD
    ID
    
    :
        tad_id: int - TAD
        left: int - ()
        right: int - ()
        original_left: int - ()
        original_right: int - ()
        left_boundary_id: int - ID(-1)
        right_boundary_id: int - ID(-1)
        candidates_left: List[int] - 
        candidates_right: List[int] - 
    """
    tad_id: int                                   # TAD ID
    left: int                                     # 
    right: int                                    # 
    original_left: int                            # 
    original_right: int                           # 
    left_boundary_id: int = -1                    # ID
    right_boundary_id: int = -1                   # ID
    candidates_left: List[int] = field(default_factory=list)   # 
    candidates_right: List[int] = field(default_factory=list)  # 

@dataclass
class LinearSignals:
    """
    
    
    
    linearAnno.csv
    
    :
        ctcf: np.ndarray - CTCF(CTCF)
        atac: np.ndarray - ATAC(ATAC/)
        eigenvector: np.ndarray - Eigenvector(A/B compartment)
    
    :
        length: int - (bin)
    """
    ctcf: np.ndarray        # CTCF
    atac: np.ndarray        # ATAC
    eigenvector: np.ndarray # Eigenvector
    
    @property
    def length(self) -> int:
        """"""
        return len(self.ctcf)

# Union-Find () 

class UnionFind:
    """
    
    
    TADTAD,
    ,
    
    
    
    :
        parent: List[int] - ,parent[i]i
        rank: List[int] - ,
    """
    
    def __init__(self, n: int):
        """
        
        
        :
            n: int - (TAD)
        """
        self.parent = list(range(n))
        # 0
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        """
        x
        
        :
        
        :
            x: int - 
        
        :
            int: x()
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> None:
        """
        
        
        :
        
        :
            x: int - 
            y: int - 
        """
        px, py = self.find(x), self.find(y)
        if px == py: 
            return  # 
        
        if self.rank[px] < self.rank[py]: 
            px, py = py, px
        self.parent[py] = px
        
        # ,+1
        if self.rank[px] == self.rank[py]: 
            self.rank[px] += 1
    
    def get_components(self) -> Dict[int, List[int]]:
        """
        
        
        :
            Dict[int, List[int]]: 
        """
        components = defaultdict(list)
        for i in range(len(self.parent)):
            components[self.find(i)].append(i)
        return dict(components)


def load_data(data_dir: str, logger: logging.Logger):
    """
    
    
    :
    - linearAnno.csv: CTCFATACEigenvector
    - obs.txt: Hi-C
    - obsLarge.txt: Hi-C()
    - oe.txt: O/E(Observed/Expected)(Corner)
    - TAD.txt: TAD(half-open)
    
    :
    - TAD.txt  half-open  [left, right)
    -  [left, right_closed], right_closed = right - 1
    -  bin 
    
    :
        data_dir: str - 
        logger: logging.Logger - 
    
    :
        Tuple: (signals, obs, obs_large, oe, tads)
            - signals: LinearSignals - 
            - obs: np.ndarray - 
            - obs_large: np.ndarray - 
            - oe: np.ndarray - O/E
            - tads: np.ndarray - TAD,(N, 2),[left, right]()
    """
    logger.info(f" {data_dir} ...")
    
    linear_anno_path = os.path.join(data_dir, 'linearAnno.csv')
    linear_anno_data = np.loadtxt(linear_anno_path, delimiter=',', skiprows=2)
    if linear_anno_data.ndim == 1: 
        linear_anno_data = linear_anno_data.reshape(1, -1)
    
    ctcf = linear_anno_data[:, 0]           # 1:CTCF
    atac = linear_anno_data[:, 1]           # 2:ATAC
    # 3:Eigenvector()
    eigenvector = linear_anno_data[:, 2] if linear_anno_data.shape[1] > 2 else np.zeros(len(ctcf))
    signals = LinearSignals(ctcf=ctcf, atac=atac, eigenvector=eigenvector)
    
    # Hi-C
    obs = np.loadtxt(os.path.join(data_dir, 'obs.txt'))           # 
    obs_large = np.loadtxt(os.path.join(data_dir, 'obsLarge.txt')) # 
    oe = np.loadtxt(os.path.join(data_dir, 'oe.txt'))             # O/E
    
    # TAD
    # :TAD.txt  half-open  [left, right)
    #  [left, right_closed], right_closed = right - 1
    tad_path = os.path.join(data_dir, 'TAD.txt')
    tads = np.loadtxt(tad_path)
    if np.size(tads) == 0: 
        tads = np.empty((0, 2), dtype=int)
    else:
        if tads.ndim == 1: 
            tads = tads.reshape(1, -1)
    tads = tads.astype(int)
    
    #  half-open  right (right - 1)
    if len(tads) > 0:
        tads[:, 1] = tads[:, 1] - 1
    
    return signals, obs, obs_large, oe, tads

def create_boundary_map(tads: np.ndarray, logger: logging.Logger) -> BoundaryMap:
    """
    ,
    
    ID:
    1. TAD
    2. 
    3. (<=NESTED_TOLERANCE)
    
    TAD,
    
    :
        tads: np.ndarray - TAD,(N, 2)
        logger: logging.Logger - 
    
    :
        BoundaryMap: 
    """
    boundary_map = BoundaryMap()
    
    # all_coords: [(, TAD_ID, ), ...]
    all_coords = []
    for i, (start, end) in enumerate(tads):
        all_coords.append((start, i, 'left'))   # 
        all_coords.append((end, i, 'right'))    # 
    
    all_coords.sort(key=lambda x: x[0])
    
    for coord, _, _ in all_coords:
        boundary_map.get_or_create_boundary(coord, NESTED_TOLERANCE)
    
    logger.info(f"  : {len(all_coords)} -> {len(boundary_map.boundaries)} ")
    return boundary_map

def create_tad_nodes(tads: np.ndarray, boundary_map: BoundaryMap) -> Dict[int, TADNode]:
    """
    TAD
    
    TADTADNode,ID
    
    :
        tads: np.ndarray - TAD,(N, 2)
        boundary_map: BoundaryMap - 
    
    :
        Dict[int, TADNode]: TAD_IDTADNode
    """
    node_map = {}
    
    for i, (start, end) in enumerate(tads):
        # ID
        left_bid = boundary_map.coord_to_boundary_id.get(start, -1)
        right_bid = boundary_map.coord_to_boundary_id.get(end, -1)
        
        c_left = boundary_map.boundaries[left_bid].canonical_coord if left_bid >= 0 else start
        c_right = boundary_map.boundaries[right_bid].canonical_coord if right_bid >= 0 else end
        
        # TAD
        node = TADNode(i, c_left, c_right, start, end, left_bid, right_bid)
        node_map[i] = node
    
    return node_map

def generate_candidates(boundary: int, signals: LinearSignals, window: int = SEARCH_WINDOW) -> List[int]:
    """
    
    
    window
    
    :
        boundary: int - 
        signals: LinearSignals - ()
        window: int - (5)
    
    :
        List[int]: ()
    """
    candidates = set()
    
    start = max(0, boundary - window)
    end = min(signals.length, boundary + window + 1)
    
    for i in range(start, end):
        candidates.add(i)
    
    return sorted(list(candidates))

def prepare_candidates(node_map: Dict[int, TADNode], boundary_map: BoundaryMap, signals: LinearSignals):
    """
    TAD
    
    :
    1. 
    2. TAD
    
    :
        node_map: Dict[int, TADNode] - TAD
        boundary_map: BoundaryMap - 
        signals: LinearSignals - 
    """
    for bid, b in boundary_map.boundaries.items():
        b.candidates = generate_candidates(b.canonical_coord, signals)
    
    # TAD
    for node in node_map.values():
        if node.left_boundary_id >= 0:
            node.candidates_left = boundary_map.boundaries[node.left_boundary_id].candidates
        else:
            node.candidates_left = generate_candidates(node.left, signals)
        
        if node.right_boundary_id >= 0:
            node.candidates_right = boundary_map.boundaries[node.right_boundary_id].candidates
        else:
            node.candidates_right = generate_candidates(node.right, signals)

# TAD

def identify_tad_components(node_map: Dict[int, TADNode], boundary_map: BoundaryMap, 
                            logger: logging.Logger) -> List[List[int]]:
    """
    TAD
    
    Union-FindTAD
    
    :
    - ****TAD
    - TAD -> 
    - TAD -> 
    - TAD1=TAD2() -> 
    
    :
    1. TAD()
    2. TAD
    
    :
        node_map: Dict[int, TADNode] - TAD
        boundary_map: BoundaryMap - 
        logger: logging.Logger - 
    
    :
        List[List[int]]: ,TAD ID
    """
    n_tads = len(node_map)
    if n_tads == 0: 
        return []
    
    uf = UnionFind(n_tads)
    
    # TAD
    left_boundary_to_tads: Dict[int, List[int]] = defaultdict(list)   # ID -> TAD
    right_boundary_to_tads: Dict[int, List[int]] = defaultdict(list)  # ID -> TAD
    
    # TAD/
    for tad_id, node in node_map.items():
        if node.left_boundary_id >= 0:
            left_boundary_to_tads[node.left_boundary_id].append(tad_id)
        if node.right_boundary_id >= 0:
            right_boundary_to_tads[node.right_boundary_id].append(tad_id)
    
    # TAD
    # TAD,
    for bid, tad_list in left_boundary_to_tads.items():
        if len(tad_list) > 1:
            first = tad_list[0]
            for other in tad_list[1:]:
                uf.union(first, other)
    
    # TAD
    # TAD,
    for bid, tad_list in right_boundary_to_tads.items():
        if len(tad_list) > 1:
            first = tad_list[0]
            for other in tad_list[1:]:
                uf.union(first, other)
    
    raw_components = uf.get_components()
    components = list(raw_components.values())
    
    single_count = sum(1 for c in components if len(c) == 1)  # TAD
    multi_count = sum(1 for c in components if len(c) > 1)    # 
    logger.info(f"  TAD: {len(components)} ({single_count} , {multi_count} )")
    
    return components


def get_alignment_score(left_pos: int, right_pos: int, signals: LinearSignals) -> float:
    """
    
    
    TADCTCFATACEigenvector
    :
    1. CTCF (+100) - 
    2. CTCF + Eigenvector (+60) - 
    3. CTCF + ATAC (+50) - 
    4. CTCF (+20) - CTCF
    5. ATAC (+30) - 
    6. ATAC (+10) - ATAC
    7.  (0) - 
    
    :
        left_pos: int - 
        right_pos: int - 
        signals: LinearSignals - 
    
    :
        float: 
    """
    if left_pos < 0 or left_pos >= signals.length or right_pos < 0 or right_pos >= signals.length:
        return 0.0
    
    l_ctcf = signals.ctcf[left_pos] > 0     # CTCF
    r_ctcf = signals.ctcf[right_pos] > 0    # CTCF
    l_atac = signals.atac[left_pos] > 0     # ATAC
    r_atac = signals.atac[right_pos] > 0    # ATAC
    
    # Eigenvector
    # A/B compartment
    l_zero = False
    r_zero = False
    
    if 0 < left_pos < signals.length - 1:
        # 1:()
        if signals.eigenvector[left_pos-1] * signals.eigenvector[left_pos] < 0:
            l_zero = True
        # 2:(<0.05)
        elif abs(signals.eigenvector[left_pos]) < 0.05:
            l_zero = True
    
    if 0 < right_pos < signals.length - 1:
        if signals.eigenvector[right_pos-1] * signals.eigenvector[right_pos] < 0:
            r_zero = True
        elif abs(signals.eigenvector[right_pos]) < 0.05:
            r_zero = True
    
    score = 0.0
    
    
    # 1:CTCF - 
    if l_ctcf and r_ctcf:
        return BONUS_CTCF_DOUBLE
    
    # 2:CTCF + Eigenvector
    if (l_ctcf and r_zero) or (l_zero and r_ctcf):
        return BONUS_CTCF_ZERO
    
    # 3:CTCF + ATAC
    if (l_ctcf and r_atac) or (l_atac and r_ctcf):
        return BONUS_CTCF_ATAC
    
    # 4:CTCF
    if l_ctcf or r_ctcf:
        score += BONUS_SINGLE_CTCF
    
    # 5:ATAC
    if l_atac and r_atac:
        return max(score, BONUS_ATAC_DOUBLE)
    
    # 6:ATAC
    if l_atac or r_atac:
        score += BONUS_SINGLE_ATAC
    
    return score

def get_corner_bonus(oe: np.ndarray, left: int, right: int) -> float:
    """
    Corner()
    
    TAD()Hi-C
    O/ETAD3x3
    
    Corner:
    -  (+30):oe[left, right]3x3
    -  (+15):
    - Corner (0):
    
    :
        oe: np.ndarray - O/E
        left: int - 
        right: int - 
    
    :
        float: Corner
    """
    N = oe.shape[0]
    
    # :3x3
    if left < 1 or right < 1 or left >= N - 1 or right >= N - 1 or left >= right: 
        return 0.0
    
    center = oe[left, right]
    if np.isnan(center) or center <= 0: 
        return 0.0
    
    # 3x3
    max_val = center
    max_pos = (0, 0)  # 
    
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            ni, nj = left + di, right + dj
            if 0 <= ni < N and 0 <= nj < N:
                v = oe[ni, nj]
                if not np.isnan(v) and v > max_val:
                    max_val = v
                    max_pos = (di, dj)
    
    if max_pos == (0, 0): 
        #  - Corner
        return BONUS_CORNER_CENTER
    
    if max_pos in {(1, -1), (1, 0), (0, -1), (1, 1)}: 
        #  - Corner
        # (0, -1)  (0, 0)
        # (1, -1)  (1, 0)  (1, 1)
        return BONUS_CORNER_SIDE
    
    #  - Corner
    return 0.0

def calculate_tad_base_score(obs_large, obs, left, right) -> float:
    """
    TAD()
    
    DeltaTAD
    ,
    
    :0NEG_INF,
    
    
    : left, right  [left, right]
     Delta  half-open [left, right+1)
    
    :
        obs_large: np.ndarray - Hi-C
        obs: np.ndarray - Hi-C
        left: int - ()
        right: int - ()
    
    :
        float: TAD()
    """
    global _tad_score_cache
    
    key = (left, right)
    if key in _tad_score_cache: 
        return _tad_score_cache[key]
    
    # obs_largeobs
    offset = -(obs_large.shape[0] - obs.shape[0]) // 2
    N = obs_large.shape[0]
    
    #  half-open  Delta 
    right_half_open = right + 1
    
    # ( half-open  right)
    local_l, local_r = left - offset, right_half_open - offset
    
    if local_l < 0 or local_r > N or local_l >= local_r:
        _tad_score_cache[key] = 0.0  # :0NEG_INF
        return 0.0
    
    try:
        # Delta( half-open )
        score = Delta(obs_large, offset, left, right_half_open, 1.1)
        
        if np.isnan(score) or np.isinf(score): 
            score = 0.0
        
        _tad_score_cache[key] = score
        return score
    except:
        # 0()
        _tad_score_cache[key] = 0.0
        return 0.0

def clear_tad_score_cache():
    """
    TAD
    
    ,
    """
    global _tad_score_cache
    _tad_score_cache = {}
    clear_diags_cache()

def get_score_optimality_bonus(obs_large, obs, left, right, base_score) -> float:
    """
    TAD,/
    
    TAD,
    
    
    ():
    - >=95% -> +20
    - >=85% -> +10
    - >=75% -> 0
    - <75% -> -10
    
    :
        obs_large: np.ndarray - Hi-C
        obs: np.ndarray - Hi-C
        left: int - 
        right: int - 
        base_score: float - TAD
    
    :
        float: ()()
    """
    if base_score <= 0:
        return 0.0
    
    max_s = base_score
    for di in range(-SEARCH_WINDOW, SEARCH_WINDOW+1):
        if di == 0: 
            continue
        # (TAD)
        nl, nr = left + di, right + di
        s = calculate_tad_base_score(obs_large, obs, nl, nr)
        if s > max_s: 
            max_s = s
    
    if max_s <= 0:
        return BONUS_SCORE_OPTIMAL
    
    ratio = base_score / max_s
    
    if ratio >= 0.95:
        return BONUS_SCORE_OPTIMAL
    elif ratio >= 0.85:
        return BONUS_SCORE_OPTIMAL * 0.3
    elif ratio >= 0.80:
        return 0.0
    else:
        return PENALTY_SCORE_SUBOPTIMAL

def compute_tad_score(left: int, right: int, signals: LinearSignals, 
                      oe: np.ndarray, obs_large: np.ndarray, obs: np.ndarray,
                      node: TADNode, boundary_map: BoundaryMap) -> float:
    """
    TAD - 
    
    ,TAD:
    1. TAD:Hi-CDelta
    2. :CTCF/ATAC/Eigenvector
    3. Corner:TADO/E
    4. Score:
    5. :
    6. :
    
     - :
    left>=right,/,
    "",
    
    
    : [left, right], TAD  bin left  bin right
    
    :
        left: int - ()
        right: int - ()
        signals: LinearSignals - 
        oe: np.ndarray - O/E
        obs_large: np.ndarray - Hi-C
        obs: np.ndarray - Hi-C
        node: TADNode - TAD()
        boundary_map: BoundaryMap - ()
    
    :
        float: ()left>=rightNEG_INF
    """
    if left >= right: 
        return NEG_INF  # 
    
    # ===== 1. TAD =====
    # RobusTADDelta,TAD
    base = calculate_tad_base_score(obs_large, obs, left, right)
    
    # ===== 2.  =====
    # CTCF/ATAC/Eigenvector
    align = get_alignment_score(left, right, signals)
    
    # ===== 3. Corner =====
    # TADO/E
    corner = get_corner_bonus(oe, left, right)
    
    # ===== 4. TAD score/ =====
    score_bonus = get_score_optimality_bonus(obs_large, obs, left, right, base)
    
    # ===== 5.  =====
    original_bonus = 0.0
    if left == node.original_left:
        original_bonus += BONUS_ORIGINAL_POS * 0.5
    if right == node.original_right:
        original_bonus += BONUS_ORIGINAL_POS * 0.5
    
    # ===== 6.  =====
    # ,TAD
    l_share = boundary_map.boundaries[node.left_boundary_id].share_count if node.left_boundary_id >= 0 else 1
    r_share = boundary_map.boundaries[node.right_boundary_id].share_count if node.right_boundary_id >= 0 else 1
    share_weight = max(1.0, (l_share + r_share) / 2.0)
    
    total = base + (align * share_weight) + corner + score_bonus + original_bonus
    
    return total

# DP:

def optimize_component(component: List[int], node_map: Dict[int, TADNode], 
                       boundary_map: BoundaryMap, signals: LinearSignals,
                       oe: np.ndarray, obs_large: np.ndarray, obs: np.ndarray,
                       logger: logging.Logger) -> None:
    """
    TAD
    
    ,
    :
    
    1. TAD:TAD
    2. TAD:DP
    3. TAD:
    
    :
        component: List[int] - TAD ID
        node_map: Dict[int, TADNode] - TAD
        boundary_map: BoundaryMap - 
        signals: LinearSignals - 
        oe: np.ndarray - O/E
        obs_large: np.ndarray - Hi-C
        obs: np.ndarray - Hi-C
        logger: logging.Logger - 
    """
    # TAD:
    if len(component) == 1:
        _optimize_single_tad(component[0], node_map, boundary_map, signals, oe, obs_large, obs, logger)
        return
    
    # ID
    boundary_ids = set()
    for tad_id in component:
        node = node_map[tad_id]
        if node.left_boundary_id >= 0:
            boundary_ids.add(node.left_boundary_id)
        if node.right_boundary_id >= 0:
            boundary_ids.add(node.right_boundary_id)
    
    boundary_ids = sorted(boundary_ids)
    n_boundaries = len(boundary_ids)
    
    if n_boundaries == 0:
        return
    
    total_combinations = 1
    for bid in boundary_ids:
        n_cands = len(boundary_map.boundaries[bid].candidates)
        total_combinations *= n_cands
        if total_combinations > MAX_COMBINATIONS:
            break
    
    logger.debug(f"  Component: {len(component)} TADs, {n_boundaries} boundaries, ~{total_combinations} combinations")
    
    if total_combinations <= MAX_COMBINATIONS:
        # :DP
        _optimize_component_dp(component, boundary_ids, node_map, boundary_map, 
                               signals, oe, obs_large, obs, logger)
    else:
        logger.debug(f"    Using iterative optimization")
        _optimize_component_iterative(component, boundary_ids, node_map, boundary_map,
                                      signals, oe, obs_large, obs, logger)


def _optimize_single_tad(tad_id: int, node_map: Dict[int, TADNode], boundary_map: BoundaryMap,
                         signals: LinearSignals, oe: np.ndarray, 
                         obs_large: np.ndarray, obs: np.ndarray, logger: logging.Logger) -> None:
    """
    TAD
    
    TADTAD,,
    
    
    :
        tad_id: int - TAD ID
        node_map: Dict[int, TADNode] - TAD
        boundary_map: BoundaryMap - 
        signals: LinearSignals - 
        oe: np.ndarray - O/E
        obs_large: np.ndarray - Hi-C
        obs: np.ndarray - Hi-C
        logger: logging.Logger - 
    """
    node = node_map[tad_id]
    
    best_score = NEG_INF
    best_left = node.original_left
    best_right = node.original_right
    
    for l_coord in node.candidates_left:
        for r_coord in node.candidates_right:
            if l_coord >= r_coord:
                continue
            
            score = compute_tad_score(l_coord, r_coord, signals, oe, obs_large, obs, node, boundary_map)
            
            if score > best_score:
                best_score = score
                best_left = l_coord
                best_right = r_coord
    
    node.left = best_left
    node.right = best_right
    
    if node.left_boundary_id >= 0:
        boundary_map.boundaries[node.left_boundary_id].optimal_coord = best_left
    if node.right_boundary_id >= 0:
        boundary_map.boundaries[node.right_boundary_id].optimal_coord = best_right


def _optimize_component_dp(component: List[int], boundary_ids: List[int],
                           node_map: Dict[int, TADNode], boundary_map: BoundaryMap,
                           signals: LinearSignals, oe: np.ndarray,
                           obs_large: np.ndarray, obs: np.ndarray, 
                           logger: logging.Logger) -> None:
    """
    DP
    
    :,
    TAD,
    
    :,TAD
    ,(),
    
    
    :
        component: List[int] - TAD ID
        boundary_ids: List[int] - ID
        node_map: Dict[int, TADNode] - TAD
        boundary_map: BoundaryMap - 
        signals: LinearSignals - 
        oe: np.ndarray - O/E
        obs_large: np.ndarray - Hi-C
        obs: np.ndarray - Hi-C
        logger: logging.Logger - 
    """
    # :ID
    candidate_lists = []
    for bid in boundary_ids:
        candidate_lists.append(boundary_map.boundaries[bid].candidates)
    
    best_combination = {bid: boundary_map.boundaries[bid].canonical_coord for bid in boundary_ids}
    best_score = NEG_INF
    
    # product(*candidate_lists) 
    for combo in product(*candidate_lists):
        # ID
        coord_map = {bid: coord for bid, coord in zip(boundary_ids, combo)}
        
        # TAD
        total_score = 0.0
        valid = True
        
        for tad_id in component:
            node = node_map[tad_id]
            
            # TAD
            left_coord = coord_map.get(node.left_boundary_id, node.original_left)
            right_coord = coord_map.get(node.right_boundary_id, node.original_right)
            
            # :left < right
            if left_coord >= right_coord:
                valid = False
                break
            
            # TAD
            score = compute_tad_score(left_coord, right_coord, signals, oe, obs_large, obs, 
                                      node, boundary_map)
            
            # (NEG_INF)
            if score == NEG_INF:
                valid = False
                break
            
            total_score += score
        
        if valid and total_score > best_score:
            best_score = total_score
            best_combination = coord_map.copy()
    
    _apply_combination(component, boundary_ids, best_combination, node_map, boundary_map, logger)


def _optimize_component_iterative(component: List[int], boundary_ids: List[int],
                                  node_map: Dict[int, TADNode], boundary_map: BoundaryMap,
                                  signals: LinearSignals, oe: np.ndarray,
                                  obs_large: np.ndarray, obs: np.ndarray,
                                  logger: logging.Logger, max_iterations: int = 30) -> None:
    """
    
    
    ,:
    1. 
    2. ,()
    3. ()
    
    ,,
    
    :
        component: List[int] - TAD ID
        boundary_ids: List[int] - ID
        node_map: Dict[int, TADNode] - TAD
        boundary_map: BoundaryMap - 
        signals: LinearSignals - 
        oe: np.ndarray - O/E
        obs_large: np.ndarray - Hi-C
        obs: np.ndarray - Hi-C
        logger: logging.Logger - 
        max_iterations: int - (30)
    """
    current_combo = {bid: boundary_map.boundaries[bid].canonical_coord for bid in boundary_ids}
    
    for iteration in range(max_iterations):
        improved = False  # 
        
        for bid in boundary_ids:
            boundary = boundary_map.boundaries[bid]
            old_coord = current_combo[bid]
            best_coord = old_coord
            best_score = NEG_INF
            
            for candidate in boundary.candidates:
                current_combo[bid] = candidate
                
                # TAD
                total_score = 0.0
                valid = True
                
                for tad_id in component:
                    node = node_map[tad_id]
                    left_coord = current_combo.get(node.left_boundary_id, node.original_left)
                    right_coord = current_combo.get(node.right_boundary_id, node.original_right)
                    
                    if left_coord >= right_coord:
                        valid = False
                        break
                    
                    score = compute_tad_score(left_coord, right_coord, signals, oe, obs_large, obs,
                                              node, boundary_map)
                    if score == NEG_INF:
                        valid = False
                        break
                    
                    total_score += score
                
                if valid and total_score > best_score:
                    best_score = total_score
                    best_coord = candidate
            
            if best_coord != old_coord:
                improved = True
            current_combo[bid] = best_coord
        
        if not improved:
            break
    
    logger.debug(f"    Converged in {iteration + 1} iterations")
    
    _apply_combination(component, boundary_ids, current_combo, node_map, boundary_map, logger)


def _apply_combination(component: List[int], boundary_ids: List[int], 
                       coord_map: Dict[int, int], node_map: Dict[int, TADNode],
                       boundary_map: BoundaryMap, logger: logging.Logger) -> None:
    """
    TAD
    
    TAD
    
    :
        component: List[int] - TAD ID
        boundary_ids: List[int] - ID
        coord_map: Dict[int, int] - ID
        node_map: Dict[int, TADNode] - TAD
        boundary_map: BoundaryMap - 
        logger: logging.Logger - 
    """
    for bid, coord in coord_map.items():
        boundary_map.boundaries[bid].optimal_coord = coord
    
    # TAD
    for tad_id in component:
        node = node_map[tad_id]
        if node.left_boundary_id >= 0:
            node.left = coord_map.get(node.left_boundary_id, node.original_left)
        if node.right_boundary_id >= 0:
            node.right = coord_map.get(node.right_boundary_id, node.original_right)


def refine_tads(data_dir: str, output_path: str = None, debug: bool = False):
    """
    TAD
    
    TAD:
    1. 
    2. ()
    3. TAD
    4. 
    5. TAD
    6. 
    7. TAD
    
    :
        data_dir: str - ,:
                        linearAnno.csv, obs.txt, obsLarge.txt, oe.txt, TAD.txt
        output_path: str - (data_dir/TAD_dp.txt)
        debug: bool - 
    """
    logger = setup_logger(debug)
    logger.info("="*60)
    logger.info("TAD DP Refine (V7: DP-based Joint Optimization)")
    logger.info("="*60)
    
    clear_tad_score_cache()
    
    # Step 0: 
    signals, obs, obs_large, oe, tads = load_data(data_dir, logger)
    
    # TAD
    if len(tads) == 0:
        logger.info("TAD")
        if not output_path: 
            output_path = os.path.join(data_dir, 'TAD_dp.txt')
        np.savetxt(output_path, np.empty((0, 2)), delimiter=' ', fmt='%d')
        return
    
    # Step 1: 
    # ID
    logger.info("Step 1: ...")
    boundary_map = create_boundary_map(tads, logger)
    
    # Step 2: TAD
    # TAD,ID
    logger.info("Step 2: TAD...")
    node_map = create_tad_nodes(tads, boundary_map)
    
    # Step 3: 
    logger.info("Step 3: ...")
    prepare_candidates(node_map, boundary_map, signals)
    
    # Step 4: TAD
    # Union-FindTAD
    logger.info("Step 4: TAD...")
    components = identify_tad_components(node_map, boundary_map, logger)
    
    # Step 5: ()
    logger.info("Step 5: ...")
    max_tad_width = max(node.right - node.left for node in node_map.values()) + SEARCH_WINDOW * 2 + 10
    global _precomputed_diags, _precomputed_diags_info
    _precomputed_diags = precompute_diags_for_matrix(obs_large, max_tad_width)
    offset = -(obs_large.shape[0] - obs.shape[0]) // 2
    _precomputed_diags_info = (id(obs_large), offset)
    
    # Step 6: DP
    logger.info("Step 6: DP...")
    for i, component in enumerate(components):
        logger.debug(f"  Processing component {i+1}/{len(components)}: {len(component)} TADs")
        optimize_component(component, node_map, boundary_map, signals, oe, obs_large, obs, logger)
    
    # : [left, right], half-open [left, right+1)
    results = []
    opt_cnt = 0  # TAD
    for i in range(len(tads)):
        node = node_map[i]
        if node.left != node.original_left or node.right != node.original_right:
            opt_cnt += 1
        #  right  half-open(right + 1)
        results.append([node.left, node.right + 1])
    
    if not output_path: 
        output_path = os.path.join(data_dir, 'TAD_dp.txt')
    np.savetxt(output_path, np.array(results), delimiter=' ', fmt='%d')
    
    opt_rate = opt_cnt / len(tads) * 100 if len(tads) > 0 else 0
    logger.info(f"! {opt_cnt}/{len(tads)} TAD ({opt_rate:.1f}%)")
    logger.info(f" {output_path}")


def _is_valid_data_dir(p):
    """
    
    
    :
    - linearAnno.csv: 
    - obs.txt: Hi-C
    - obsLarge.txt: Hi-C
    - oe.txt: O/E
    - TAD.txt: TAD
    
    :
        p: str - 
    
    :
        bool: 
    """
    req = ['linearAnno.csv', 'obs.txt', 'obsLarge.txt', 'oe.txt', 'TAD.txt']
    return os.path.isdir(p) and all(os.path.exists(os.path.join(p, f)) for f in req)

def _collect_data_dirs(root):
    """
    
    
    :
        root: str - 
    
    :
        List[str]: ()
    """
    dirs = []
    for cur, _, _ in os.walk(root):
        if _is_valid_data_dir(cur): 
            dirs.append(cur)
    return sorted(dirs)

def main():
    """
    
    
    :
    1. :--data_dir 
    2. :--root_dir ,
    
    :
    - --output: ()
    - --debug: (,)
    """
    parser = argparse.ArgumentParser(
        description='TAD (V7: DP-based Joint Optimization)'
    )
    
    # :--data_dir--root_dir
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--data_dir', type=str, help='')
    g.add_argument('--root_dir', type=str, help='')
    
    parser.add_argument('--output', type=str, help='')
    parser.add_argument('--debug', action='store_true', help='')
    
    args = parser.parse_args()
    
    if args.data_dir:
        if _is_valid_data_dir(args.data_dir):
            refine_tads(args.data_dir, args.output, args.debug)
        else:
            print(f": {args.data_dir}")
    else:
        dirs = _collect_data_dirs(args.root_dir)
        if not dirs:
            print(f": {args.root_dir} ")
            return
        
        if args.debug: 
            dirs = dirs[:1]
        
        print(f" {len(dirs)} ")
        
        total_tads = 0
        total_opt = 0
        
        for i, d in enumerate(dirs, 1):
            print(f"[{i}/{len(dirs)}] : {d}")
            try: 
                refine_tads(d, None, args.debug)
            except Exception as e: 
                print(f": {e}")
        
        print(f"\n!")

if __name__ == '__main__':
    main()
