#!/usr/bin/env python
"""
Plot pileup from txt file
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import copy
from matplotlib import cm
import os
import sys

def get_min_max(pups, vmin=None, vmax=None, sym=True, scale="log"):
    """Automatically determine minimal and maximal colour intensity for pileups

    Parameters
    ----------
    pups : np.array
        Numpy array of numpy arrays containing pileups.
    vmin : float, optional
        Force certain minimal colour. The default is None.
    vmax : float, optional
        Force certain maximal colour. The default is None.
    sym : bool, optional
        Whether the output should be symmetrical around 0. The default is True.
    scale : str, optional
        Scale type ('log' or 'linear'). The default is 'log'.

    Returns
    -------
    vmin : float
        Selected minimal colour.
    vmax : float
        Selected maximal colour.

    """
    if vmin is not None and vmax is not None:
        if sym:
            print("Can't set both vmin and vmax and get symmetrical scale. Plotting non-symmetrical")
        return vmin, vmax
    else:
        # Handle both single array and list of arrays
        # If pups is a numpy array, use it directly; if list/tuple, concatenate
        if isinstance(pups, np.ndarray):
            comb = pups.ravel()
        elif isinstance(pups, (list, tuple)):
            # Check if it's a list of arrays or a single array wrapped in list
            if len(pups) == 1 and isinstance(pups[0], np.ndarray):
                comb = pups[0].ravel()
            else:
                comb = np.concatenate([np.array(pup).ravel() for pup in pups])
        else:
            comb = np.array(pups).ravel()
        comb = comb[comb != -np.inf]
        comb = comb[comb != np.inf]
        comb = comb[comb != 0]
        if np.isnan(comb).all():
            raise ValueError("Data only contains NaNs or zeros")
    if vmin is None and vmax is None:
        vmax = np.nanmax(comb)
        vmin = np.nanmin(comb)
    elif vmin is not None:
        vmax = np.nanmax(comb)
    elif vmax is not None:
        vmin = np.nanmin(comb)
    if sym:
        if scale == "linear":
            print("Can't use symmetrical scale with linear. Plotting non-symmetrical")
            pass
        else:
            vmax = np.max(np.abs([vmin, vmax]))
            if vmax >= 1:
                vmin = 2 ** -np.log2(vmax)
            else:
                raise ValueError(
                    "Maximum value is less than 1.0, can't plot using symmetrical scale"
                )
    return vmin, vmax


def plot_pileup(txt_file, output_file=None, title=None):
    """
    Plot pileup from txt file
    
    Parameters
    ----------
    txt_file : str
        Path to input txt file containing pileup data
    output_file : str, optional
        Path to output figure file. If None, will be generated from input filename.
    title : str, optional
        Title for the plot. If None, will use filename.
    """
    # Load data
    scaledPU = np.loadtxt(txt_file)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Setup colormap
    cmap_emptypixel = (0.98, 0.98, 0.98)
    try:
        # Try new matplotlib API (>=3.7)
        pucmap = copy.copy(plt.colormaps['coolwarm'])
    except (AttributeError, KeyError):
        # Fallback to old API
        pucmap = copy.copy(cm.get_cmap('coolwarm'))
    pucmap.set_bad(cmap_emptypixel)
    
    # Get min/max values
    vmin, vmax = get_min_max([scaledPU], None, None, sym=True, scale='log')
    print(f"vmin: {vmin:.4f}, vmax: {vmax:.4f}")
    
    # Plot
    im = ax.imshow(scaledPU, cmap=pucmap, norm=LogNorm(vmin, vmax))
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set title
    if title is None:
        title = os.path.basename(txt_file)
    ax.set_title(title)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Log scale')
    
    # Save figure
    if output_file is None:
        base_name = os.path.splitext(txt_file)[0]
        output_file = f"{base_name}_plot.pdf"
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {output_file}")
    
    return fig, ax


def plot_all_tools(data_dir="data", output_file="fig3_15toolsPU.pdf"):
    """
    Plot pileups for all 15 tools in a 3x5 grid
    
    Parameters
    ----------
    data_dir : str
        Directory containing tool subdirectories with txt files
    output_file : str
        Output PDF filename (can be relative or absolute path)
    """
    # Tool configuration matching fig3.ipynb
    # Format: (display_name, tool_dir_name, bed_pattern_for_matching)
    tools_config = [
        ('RobusTAD', 'RobusTAD', 'LMCC_TAD_ratio1_delta0.2_hq'),
        ('Arrowhead', 'Arrowhead', 'Arrowhead'),
        ('EAST', 'EAST', 'EAST2'),
        ('HiTAD', 'HiTAD', 'hitad'),
        ('IC-Finder', 'IC-Finder', 'icfinder'),
        ('OnTAD', 'OnTAD', 'OnTAD'),
        ('RefHiC', 'RefHiC', 'refhic'),
        ('GMAP', 'GMAP', 'rGMAP'),
        ('TopDom', 'TopDom', 'TopDom'),
        ('Armatus', 'Armatus', 'Armatus'),
        ('Domaincall', 'Domaincall', 'DI'),
        ('CaTCH', 'CaTCH', 'CaTCH'),
        ('deDoc', 'deDoc', 'deDoc'),
        ('Grinch', 'Grinch', 'grinch'),
        ('HiCSeg', 'HiCSeg', 'HiCSeg'),
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, data_dir)
    
    # If output_file is relative, make it relative to script_dir
    if not os.path.isabs(output_file):
        output_path = os.path.join(script_dir, output_file)
    else:
        output_path = output_file
    
    # Find all txt files
    txt_files = []
    tool_names = []
    
    for display_name, tool_dir, bed_pattern in tools_config:
        tool_data_dir = os.path.join(data_path, tool_dir)
        if not os.path.exists(tool_data_dir):
            print(f"Warning: Directory not found: {tool_data_dir}")
            continue
        
        # Find txt file in this directory
        txt_file = None
        for f in os.listdir(tool_data_dir):
            if f.endswith('_10-shifts_local_rescaled.txt'):
                # Check if it matches the expected pattern
                if bed_pattern in f or tool_dir in f:
                    txt_file = os.path.join(tool_data_dir, f)
                    break
        # If not found with pattern matching, use any matching file
        if txt_file is None:
            for f in os.listdir(tool_data_dir):
                if f.endswith('_10-shifts_local_rescaled.txt'):
                    txt_file = os.path.join(tool_data_dir, f)
                    break
        
        if txt_file and os.path.exists(txt_file):
            txt_files.append(txt_file)
            tool_names.append(display_name)
        else:
            print(f"Warning: txt file not found for {display_name} in {tool_data_dir}")
    
    if len(txt_files) == 0:
        print("Error: No txt files found!")
        return None
    
    print(f"Found {len(txt_files)} tool pileup files")
    
    # Create 3x5 subplot grid
    fig, axs = plt.subplots(3, 5, figsize=(20, 12))
    
    # Setup colormap
    cmap_emptypixel = (0.98, 0.98, 0.98)
    try:
        pucmap = copy.copy(plt.colormaps['coolwarm'])
    except (AttributeError, KeyError):
        pucmap = copy.copy(cm.get_cmap('coolwarm'))
    pucmap.set_bad(cmap_emptypixel)
    
    # Plot each tool
    for i, (txt_file, name) in enumerate(zip(txt_files, tool_names)):
        row = i // 5
        col = i % 5
        
        axs[row, col].clear()
        scaledPU = np.loadtxt(txt_file)
        # get_min_max expects array directly (as in fig3.ipynb)
        vmin, vmax = get_min_max(scaledPU, None, None, sym=True, scale='log')
        print(f"{name}: vmin={vmin:.4f}, vmax={vmax:.4f}")
        
        axs[row, col].imshow(scaledPU, cmap=pucmap, norm=LogNorm(vmin, vmax))
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])
        axs[row, col].set_title(name)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nFigure saved to: {output_path}")
    
    return fig, axs


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        # Plot all tools
        data_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
        output_file = sys.argv[3] if len(sys.argv) > 3 else "fig3_15toolsPU.pdf"
        plot_all_tools(data_dir, output_file)
    elif len(sys.argv) < 2:
        print(f"Usage:")
        print(f"  Single tool: {sys.argv[0]} <input.txt> [output.pdf] [title]")
        print(f"  All tools:   {sys.argv[0]} --all [data_dir] [output.pdf]")
        print(f"Examples:")
        print(f"  {sys.argv[0]} pileup.txt output.pdf 'Armatus'")
        print(f"  {sys.argv[0]} --all data fig3_15toolsPU.pdf")
        sys.exit(1)
    else:
        # Single tool plot
        txt_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        title = sys.argv[3] if len(sys.argv) > 3 else None
        
        if not os.path.exists(txt_file):
            print(f"Error: File not found: {txt_file}")
            sys.exit(1)
        
        plot_pileup(txt_file, output_file, title)
    # plt.show()
