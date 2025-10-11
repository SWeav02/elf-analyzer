# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 15:22:05 2025

@author: Sam
"""

from elf_analyzer.core import ElfAnalyzer
import numpy as np
from numba import njit
import time

# ----------------------
# Union–Find utilities
# ----------------------

@njit(inline='always')
def find_root(parents, x):
    while parents[x] != x:
        parents[x] = parents[parents[x]] # path compression
        x = parents[x]
    return x


@njit(inline='always')
def union(parents, spans, a, b):
    ra = find_root(parents, a)
    rb = find_root(parents, b)
    if ra != rb:
        # Merge span info
        for d in range(3):
            spans[ra, d] = spans[ra, d] or spans[rb, d]
        parents[rb] = ra

@njit(fastmath=True, cache=True, inline="always")
def flat_to_coords(idx, nx, ny, nz):
    i = idx // (ny * nz)
    j = (idx % (ny * nz)) // nz
    k = idx % nz
    return i, j, k


@njit(fastmath=True, cache=True, inline="always")
def coords_to_flat(i, j, k, nx, ny, nz):
    return i * (ny * nz) + j * nz + k

# get test graph
analyzer = ElfAnalyzer.from_vasp(reference_filename="ELFCAR")
grid = analyzer.reference_grid
data = grid.total
basin_labels = analyzer.basin_labels

graph = analyzer._get_bifurcation_graph()

reducible_keys = []
feature_basins = []
appear_at = []
disappear_at = []
for node in graph:
    if node.reducible:
        reducible_keys.append(node.key)
        feature_basins.append(node.basins)
        appear_at.append(node.min_elf)
        disappear_at.append(node.max_elf)
reducible_keys = np.array(reducible_keys, dtype=np.int64)
appear_at = np.array(appear_at, dtype=np.float64)
disappear_at = np.array(disappear_at, dtype=np.float64)

# get flat basins and data
flat_data = data.ravel()
flat_basin_labels = basin_labels.ravel()

# get unique elf values to iterate over
unique_elfs = np.unique(appear_at)

t0=time.time()
# get the indices in each range of elf values
bin_indices = np.digitize(flat_data, unique_elfs, right=True)
coords_per_value = [np.argwhere(bin_indices==i) for i in range(1,len(unique_elfs))]
t1=time.time()

_, basin_index_map = np.unique(flat_basin_labels, return_index=True)
basin_index_labels = basin_index_map[flat_basin_labels]
t2=time.time()
print(t1-t0)
print(t2-t1)

# create initial labels
initial_labels = np.arange(len(basin_index_labels))

# create tracker for which feature a basin belongs to
basin_features = np.empty(len(analyzer.basin_maxima_frac), dtype=np.int64)

nx, ny, nz = data.shape
neighbor_transforms, _ = grid.point_neighbor_transforms
for elf_idx, elf_value in enumerate(unique_elfs[1:]):
    # 1. Get new features at this value
    # 2. Update hole connections, noting which basins they border (how?)
    # 3. Loop over new features. temporarily connect basins outside this feature
    # NOTE: This would require basins are labeled in a non-periodic way or using
    # a supercell I think.
    # 4. Check which holes are periodic in at least one direction.
    # 5. Check which atoms sit in non-periodic holes. These are surrounded by the
    # feature
    # 6. If a feature doesn't contain any atoms, its children won't either. I
    # should track that somehow.
    
    new_coords = coords_per_value[elf_idx]
    
    # get the new features at this value
    current_features = []
    for features, feature_elf in zip(feature_basins, appear_at):
        if feature_elf
    features = feature_basins[elf_idx]
    # reset basin feature map
    basin_features[:] = -1
    for feature_idx, feature in enumerate(features):
        for basin_idx in feature:
            basin_features[basin_idx] = feature_idx
    # iterate over each newly available coord and find its unions
    for flat_idx in new_coords:
        # get 3D coord
        coord = flat_to_coords(flat_idx, nx, ny, nz)



