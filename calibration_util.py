import json
import numpy as np
from typing import Tuple


def get_mtx_dist() -> Tuple[np.ndarray, np.ndarray]:
    """Get MTX and DIST
        Load and return the mtx and dist np ndarrays

    :return: Tuple(np.ndarry, np.ndarray)
    """
    with open("./data/mtx.json", "r") as fp:
        mtx = np.asarray(json.load(fp))

    with open("./data/dist.json", "r") as fp:
        dist = np.asarray(json.load(fp))

    return mtx, dist
