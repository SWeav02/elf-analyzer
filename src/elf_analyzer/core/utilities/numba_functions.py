# -*- coding: utf-8 -*-


import numpy as np
from numba import njit, prange
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
    """
    Finds the maximum ELF value at which each basin pair connects
    """
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
        if label == -1:
            # This shouldn't happen if bader is working properly, but if it does
            # we don't want to use this label
            continue
        # loop over the neighbors
        for si, sj, sk in neighbor_transforms:
            # wrap points
            ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
            # get the label at this point
            neigh_label = labeled_array[ii,jj,kk]
            # skip if this is part of the same basin
            if neigh_label == label or neigh_label == -1:
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

# @njit(cache=True)
def check_covalent(
    feature_frac_coord,
    atom_frac_coords,
    atom_cart_coords,
    atom_radii,
    frac2cart,
    min_covalent_angle,
        ):
    
    # first we find the two closest neighbors to this point
    # create arrays to store distances and vectors
    atom_dists = np.full(len(atom_cart_coords), 1e6, dtype=np.float64)
    atom_vecs = np.empty((len(atom_cart_coords), 3), dtype=np.float64)
    
    # transform the coord to each neighboring unit cell (and the current cell)
    fi, fj, fk = feature_frac_coord
    for si in (-1, 0, 1):
        for sj in (-1, 0, 1):
            for sk in (-1, 0, 1):
                ti = fi + si
                tj = fj + sj
                tk = fk + sk
                # convert to cartesian
                ci = ti * frac2cart[0][0] + tj * frac2cart[1][0] + tk * frac2cart[2][0]
                cj = ti * frac2cart[0][1] + tj * frac2cart[1][1] + tk * frac2cart[2][1]
                ck = ti * frac2cart[0][2] + tj * frac2cart[1][2] + tk * frac2cart[2][2]
                # calculate distance to each atom
                for i, (ai, aj, ak) in enumerate(atom_cart_coords):
                    di = ai-ci
                    dj = aj-cj
                    dk = ak-ck
                    dist = ((di)**2 + (dj)**2 + (dk)**2) ** 0.5
                    # if its lower than previous calculated distances, update
                    # our entry
                    if dist < atom_dists[i]:
                        atom_dists[i] = dist
                        atom_vecs[i] = (di, dj, dk)
    
    # Get the nearest and second nearest atoms
    sorted_atoms = np.argsort(atom_dists)
    nearest_atom = sorted_atoms[0]
    neighbor_atom = sorted_atoms[1]
    
    # First we check that we are reasonably close to being along this bond. We
    # do this by checking the angle between the neighboring atoms and our basin.
    # This is corresponds to:
        # θ = arccos((A ⋅ B) / (|A|*|B|))
    # where A and B are the vectors from the feature to each neighboring atom
    A = atom_vecs[nearest_atom]
    B = atom_vecs[neighbor_atom]

    cos_theta = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    # make sure our theta is within the bounds of arcos
    cos_theta = max(-1.0, min(1.0, cos_theta))
    # get theta
    theta = np.arccos(cos_theta)
    # If our angle is not above our tolerance, we return as not a covalent bond
    if theta < min_covalent_angle:
        return False
    else:
        return True
    
    # Next we need to distinguish between covalent bonds and polarized shells/donated electrons.
    # This is much more arbitrary as heterogenous bonds will always sit somewhere
    # on the spectrum of covalent <-> ionic. Generally, more ionic bonds will
    # pull the bonding electrons closer to the more electronegative atom. "closer"
    # here is the challenge, as we also need to take into account the size of
    # the atoms to begin with. 
    # As a rough approximation, we calculate the ratio of the bond belonging
    # to the nearest atom and compare to the ratio if we were to use each atoms
    # atomic radius

    nearest_dist = atom_dists[nearest_atom]
    neighbor_dist = atom_dists[neighbor_atom]
    
    feature_ratio = nearest_dist / (nearest_dist + neighbor_dist)
    
    # get the atomic radii of each
    nearest_radius = atom_radii[nearest_atom]
    neighbor_radius = atom_radii[neighbor_atom]
    
    radii_ratio = nearest_radius / (nearest_radius + neighbor_radius)
    
    # Now we want a measure of ionic/covalent character. Imagine 
    
    breakpoint()
    
    # Now check if the point is within a reasonable angle. First, get our points
    # in cartesian coordinates. 


# @njit(parallel=True, cache=True)
def check_all_covalent(
    feature_frac_coords,
    atom_frac_coords,
    atom_cart_coords,
    atom_radii,
    frac2cart,
    min_covalent_angle,
        ):
    # create an array to store if each feature is covalent
    covalent_features = np.zeros(len(feature_frac_coords), dtype=np.bool_)
    for i in prange(len(feature_frac_coords)):
        feature_frac_coord = feature_frac_coords[i]
        covalent_features[i] = check_covalent(
            feature_frac_coord,
            atom_frac_coords,
            atom_cart_coords,
            atom_radii,
            frac2cart,
            min_covalent_angle,
            )
    return covalent_features
    
    