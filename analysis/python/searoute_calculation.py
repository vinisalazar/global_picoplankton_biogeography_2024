#!/usr/bin/env python

"""
Calculate searoute distance between sampling stations.
"""

import sys

from functools import lru_cache
from itertools import combinations

from tqdm import tqdm
import pandas as pd
import searoute as sr

sys.path.insert(0, "/local/path/to/scripts/")

@lru_cache(maxsize=None)
def get_searoute_distance(p0, p1):
    # Check and modify p0 and p1, required if there are 0.0 values
    p0 = tuple(val if val != 0 else val + 1e-9 for val in p0)
    p1 = tuple(val if val != 0 else val + 1e-9 for val in p1)
    
    return sr.searoute(origin=p0, destination=p1, restrictions=None)


def calculate_searoute_pairwise_distances(smd):
    coords = smd[["station_fmt", "coords"]].copy()
    coords["coords"] = coords["coords"].apply(eval)
    D = pd.DataFrame(data=None, index=coords.index, columns=coords.index)
    combinations_ = list(combinations(coords.index, 2))
    distances = {}
    for i, j in tqdm(combinations_):
        p0, p1 = coords.loc[i, "coords"], coords.loc[j, "coords"]
        p0, p1 = p0[::-1], p1[::-1]
        try:
            distances[(i, j)] = get_searoute_distance(p0, p1)
            distances[(j, i)] = get_searoute_distance(p1, p0)
        except:
            breakpoint()
        D0 = distances[(i, j)]["properties"]["length"]
        D1 = distances[(j, i)]["properties"]["length"]
        D_mean = (D0 + D1) / 2
        D.loc[i, j] = D_mean
        D.loc[j, i] = D_mean

    for ix in D.index:
        D.loc[ix, ix] = 0

    D_max = D.max().max()
    D_norm = D.map(lambda x: x / D_max)

    D.to_csv("/data/gpfs/projects/punim1989/biogo-hub/provinces_final/data/distances/searoute_dist_2132_km.csv")
    D_norm.to_csv("/data/gpfs/projects/punim1989/biogo-hub/provinces_final/data/distances/searoute_dist_2132_norm.csv")

    return D, D_norm, distances


if __name__ == "__main__":
    smd = pd.read_csv("/data/gpfs/projects/punim1989/biogo-hub/provinces_final/data/metadata.csv")
    D, D_norm, distances = calculate_searoute_pairwise_distances(smd)
