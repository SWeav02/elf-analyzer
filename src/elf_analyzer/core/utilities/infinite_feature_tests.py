# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 15:55:01 2025

@author: sammw
"""
from numba import njit, prange
import numpy as np

from baderkit.core.methods.shared_numba import wrap_point
        
from elf_analyzer.core import ElfAnalyzer

@njit(parallel=True)
def flood_fill_round(
    feature_mask,
    flood_mask,
    edge_mask,
    edge_indices,
    neighbor_transforms,
        ):
    # get supercell shape and single cell shape
    nx, ny, nz = feature_mask.shape
    
    for edge_idx in prange(len(edge_indices)):
        i,j,k = edge_indices[edge_idx]
        for si, sj, sk in neighbor_transforms:
            ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
            in_bound = feature_mask[ni,nj,nk]
            # if this isn't part of our feature, skip
            if not in_bound:
                continue
            # if this point is not already flooded, flood it and add to our edge
            in_flood = flood_mask[ni,nj,nk]
            if not in_flood:
                flood_mask[ni,nj,nk] = True
                edge_mask[ni,nj,nk] = True
        # remove this point from our edge so we don't search it again
        edge_mask[i,j,k] = False
    return flood_mask, edge_mask

def flood_fill(
    feature_mask,
    seed_points,
    neighbor_transforms,
        ):
    # create our flood mask
    flood_mask = np.zeros_like(feature_mask, dtype=np.bool_)
    # add our seeds
    for i, j, k in seed_points:
        flood_mask[i,j,k] = True
    edge_mask = flood_mask.copy()
        
    # start filling
    while True:
        edge_indices = np.argwhere(edge_mask)
        print(len(edge_indices))
        if len(edge_indices) == 0:
            break
        flood_mask, edge_mask = flood_fill_round(
            feature_mask,
            flood_mask,
            edge_mask,
            edge_indices,
            neighbor_transforms,
            )
    
    return flood_mask

def check_infinite(
    flood_mask,
    grid_shape,
    seed_coord,
        ):
    nx, ny, nz = grid_shape
    transforms = [
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [1,1,0],
        [1,0,1],
        [0,1,1],
        [1,1,1]
        ]
    
    ox, oy, oz = seed_coord
    infinite = False
    for i,j,k in transforms:
        # shift coords
        x = ox + i*nx
        y = oy + j*ny
        z = oz + k*nz
        # check if transformed coord is in the flood
        if flood_mask[x,y,z]:
            infinite = True
            break

    return infinite

def check_infinite_all(
    flood_mask,
    grid_shape,
    seed_points,
        ):
    # create an array to track which points are infinite
    infinite_features = np.zeros(len(seed_points), dtype=np.bool_)
    for seed_idx in range(len(seed_points)):
        seed_coord = seed_points[seed_idx]
        infinite_features[seed_idx] = check_infinite(
            flood_mask,
            grid_shape,
            seed_coord
            )
    return infinite_features
        

analyzer = ElfAnalyzer.from_vasp() 
graph = analyzer.bifurcation_graph
basin_maxima_frac = analyzer.basin_maxima_frac
num_basins = len(basin_maxima_frac) 
grid = analyzer.reference_grid
data = grid.total
# NOTE: Should I use just shared faces instead?
neighbor_transforms, _ = grid.point_neighbor_transforms
basin_labels = analyzer.basin_labels

reducible_keys = []
basins = []
max_elfs = []
for node in graph:
    if node.reducible:
        reducible_keys.append(node.key)
        basins.append(node.basins)
        max_elfs.append(node.max_elf)
reducible_keys = np.array(reducible_keys, dtype=np.int64)
max_elfs = np.array(max_elfs, dtype=np.float64)

# get basin maxima as grid coords
basin_maxima_grid = np.round(grid.frac_to_grid(basin_maxima_frac)).astype(np.int64) % grid.shape

# create an array to store which features are infinite
infinite_features = np.zeros(max(reducible_keys)+1, dtype=np.bool_)

unique_elf_values = np.unique(max_elfs)
for value in unique_elf_values:
    # get the features at this point
    current_features = []
    current_basins = []
    seed_points = []
    for key, max_elf, basin_group in zip(reducible_keys, max_elfs, basins):
        if max_elf != value:
            continue
        # This feature disappears at this value, meaning this is the last value
        # where all basins in this feature are connected
        current_features.append(key)
        current_basins.extend(basin_group)
        # get a single point to use as a seed
        seed_points.append(basin_maxima_grid[basin_group[0]])

        
    current_basins = np.array(current_basins, dtype=np.int64)
    # create a mask for these features
    # >= should be correct here as the value should be == to the highest value
    # where all basins in this feature are connected.
    feature_mask = np.isin(basin_labels, current_basins) & (data >= value)
    feature_mask = np.tile(feature_mask, (2,2,2))
    if 4 in current_features:
        breakpoint()
    # flood
    flood_mask = flood_fill(feature_mask, seed_points, neighbor_transforms)
    # Now check which points flooded to at least one neighboring unit cell
    current_infinite_features = check_infinite_all(flood_mask, data.shape, seed_points)
    # update our infinite features array
    for key, is_infinite in zip(current_features, current_infinite_features):
        infinite_features[key] = is_infinite

