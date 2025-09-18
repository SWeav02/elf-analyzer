# -*- coding: utf-8 -*-

import math

import numpy as np
from numba import njit, prange, types
from numpy.typing import NDArray

from baderkit.core.methods.shared_numba import wrap_point

@njit(cache=True)
def find_connections(
        labeled_array: NDArray[np.int64],
        data: NDArray[np.float64],
        edge_voxels,
        basin_num: np.int64,
        neighbor_transforms: NDArray[np.int64],
        ):
    nx, ny, nz = labeled_array.shape
    # create a 2D array for tracking connections.
    # TODO: We could potentially reduce this by using lists as basins will never
    # include connections to basins with lower indices
    connection_array = np.zeros((basin_num, basin_num), dtype=np.float64)
    # loop over each edge voxel
    for i, j, k in edge_voxels:
        # get this points elf value and basin label
        elf_value = data[i,j,k]
        label = labeled_array[i,j,k]
        # loop over the neighbors
        for si, sj, sk in neighbor_transforms:
            # wrap points
            ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
            # get the label at this point
            neigh_label = labeled_array[ii,jj,kk]
            # skip if this is part of the same basin
            if neigh_label == label:
                continue
            # otherwise, get the value at this neighbor
            neigh_elf_value = data[ii,jj,kk]
            # the value at which these two points 'connect' when visualizing the
            # isosurface is the lower value.
            lower_elf = min(elf_value, neigh_elf_value)
            # we want to find the highest connection point for this pair of basins,
            # which we store in our connection array. The highest value for each
            # pair is located at index n, m where n is the lower label value
            lower_label = min(label, neigh_label)
            higher_label = max(label, neigh_label)
            # compare our values and if this is higher, update it
            if lower_elf > connection_array[lower_label, higher_label]:
                connection_array[lower_label, higher_label] = lower_elf
    
    # return connection array
    return connection_array