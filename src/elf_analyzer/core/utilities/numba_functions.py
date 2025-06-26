# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange, types
from numpy.typing import NDArray

@njit(parallel=True, cache=True)
def find_connections(
        labeled_array: NDArray[np.int64],
        data: NDArray[np.float64],
        basin_num: np.int64,
        neighbor_transforms: NDArray[np.int64],
        ):
    nx, ny, nz = labeled_array.shape
    # create a 2D array for tracking connections
    connection_array = np.zeros((basin_num, basin_num), dtype=np.float64)
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # Get this voxels label and elf value
                label = labeled_array[i, j, k]
                elf_value = data[i,j,k]
                # iterate over the neighboring voxels
                for shift_index, shift in enumerate(neighbor_transforms):
                    ii = (i + shift[0]) % nx  # Loop around box
                    jj = (j + shift[1]) % ny
                    kk = (k + shift[2]) % nz
                    # get neighbors label
                    neigh_label = labeled_array[ii, jj, kk]
                    # if any label is different, the current voxel is an edge.
                    # Note this in our edge array and break
                    if neigh_label != label:
                        # This voxel is an edge, and this neighbor belongs to a
                        # different basin
                        # NOTE: If we imagine scanning the ELF from high to low,
                        # the first value at which these voxels connect is the
                        # lower of the two.
                        neigh_elf = data[ii,jj,kk]
                        if neigh_elf < elf_value and neigh_elf > connection_array[label, neigh_label]:
                            connection_array[label, neigh_label] = lower_elf
    return connection_array