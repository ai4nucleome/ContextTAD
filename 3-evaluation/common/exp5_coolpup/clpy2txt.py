import h5sparse,tables
import pandas as pd
import numpy as np
import os
import re
import logging
logger = logging.getLogger(__name__)
__version__ = "1.1.0"
def load_pileup_df(filename, quaich=False, skipstripes=False):
    """
    Loads a dataframe saved using `save_pileup_df`

    Parameters
    ----------
    filename : str
        File to load from.
    quaich : bool, optional
        Whether to assume standard quaich file naming to extract sample name and bedname.
        The default is False.

    Returns
    -------
    annotation : pd.DataFrame
        Pileups are in the "data" column, all metadata in other columns

    """
    with h5sparse.File(filename, "r", libver="latest") as f:
        metadata = dict(zip(f["attrs"].attrs.keys(), f["attrs"].attrs.values()))
        dstore = f["data"]
        data = []
        for chunk in dstore.iter_chunks():
            chunk = dstore[chunk]
            data.append(chunk)
        annotation = pd.read_hdf(filename, "annotation")
        annotation["data"] = data
        vertical_stripe = []
        horizontal_stripe = []
        coordinates = []
        if not skipstripes:
            try:
                for i in range(len(data)):
                    vstripe = "vertical_stripe_" + str(i)
                    hstripe = "horizontal_stripe_" + str(i)
                    coords = "coordinates_" + str(i)
                    vertical_stripe.append(f[vstripe][:].toarray())
                    horizontal_stripe.append(f[hstripe][:].toarray())
                    coordinates.append(f[coords][:].astype("U13"))
                annotation["vertical_stripe"] = vertical_stripe
                annotation["horizontal_stripe"] = horizontal_stripe
                annotation["coordinates"] = coordinates
            except KeyError:
                pass
    for key, val in metadata.items():
        if key != "version":
            annotation[key] = val
        elif val != __version__:
            logger.debug(
                f"pileup generated with v{val}. Current version is v{__version__}"
            )
    if quaich:
        basename = os.path.basename(filename)
        sample, bedname = re.search(
            "^(.*)-(?:[0-9]+)_over_(.*)_(?:[0-9]+-shifts|expected).*\.clpy", basename
        ).groups()
        annotation["sample"] = sample
        annotation["bedname"] = bedname
    return annotation

def get_min_max(pups, vmin=None, vmax=None, sym=True, scale="log"):
    """Automatically determine minimal and maximal colour intensity for pileups

    Parameters
    ----------
    pups : np.array
        Numpy array of numpy arrays conaining pileups.
    vmin : float, optional
        Force certain minimal colour. The default is None.
    vmax : float, optional
        Force certain maximal colour. The default is None.
    sym : bool, optional
        Whether the output should be cymmetrical around 0. The default is True.

    Returns
    -------
    vmin : float
        Selected minimal colour.
    vmax : float
        Selected maximal colour.

    """
    if vmin is not None and vmax is not None:
        if sym:
            logger.info(
                "Can't set both vmin and vmax and get symmetrical scale. Plotting non-symmetrical"
            )
        return vmin, vmax
    else:
        comb = np.concatenate([pup.ravel() for pup in pups.ravel()])
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
            logger.info(
                "Can't use symmetrical scale with linear. Plotting non-symmetrical"
            )
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


import sys
if len(sys.argv)!=3:
    print(sys.argv[0],'input.clpy','output.txt')
else:
    pu=load_pileup_df(sys.argv[1])
    np.savetxt(sys.argv[2],pu['data'][0])