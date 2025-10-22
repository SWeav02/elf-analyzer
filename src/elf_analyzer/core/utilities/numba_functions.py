# -*- coding: utf-8 -*-


import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.methods.shared_numba import wrap_point

@njit(cache=True)
def check_covalent(
    feature_frac_coord,
    atom_frac_coords,
    atom_cart_coords,
    frac2cart,
    min_covalent_angle,
        ):
    num_atoms = len(atom_cart_coords)
    # first we find the two closest neighbors to this point
    # create arrays to store distances and vectors
    # BUGFIX: Its possible for the closest and second closest neighbor to be
    # the same atom. We need to store the closest and second closest image of
    # each atom (e.g. single atom systems)
    atom_dists = np.full(num_atoms*2, 1e6, dtype=np.float64)
    atom_vecs = np.empty((num_atoms*2, 3), dtype=np.float64)
    
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
                    if dist <= atom_dists[i]:
                        # set the previous value to be the second nearest
                        atom_dists[i+num_atoms] = atom_dists[i]
                        atom_vecs[i+num_atoms] = atom_vecs[i]
                        # update the nearest value
                        atom_dists[i] = dist
                        atom_vecs[i] = (di, dj, dk)

    # sort atoms
    sorted_atoms = np.argsort(atom_dists)
    
    # Get the nearest and second nearest atoms
    nearest_atom = sorted_atoms[0]
    neighbor_atom = sorted_atoms[1]
    
    # Bug Fix: We must only have 2 nearest neighbors or this is more like a
    # non-nuclear attractor. We make sure we don't have a third neighbor
    neigh_dist = atom_dists[1]
    next_neigh_dist = atom_dists[2]
    # check if next neighbor is within 1% of the distance
    if (next_neigh_dist - neigh_dist) / neigh_dist < 0.01:
        return False, nearest_atom, neighbor_atom
    
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
    
    # wrap atoms in case they both belong to the same atom
    nearest_atom = nearest_atom % num_atoms
    neighbor_atom = neighbor_atom % num_atoms
    # If our angle is not above our tolerance, we return as not a covalent bond
    if theta < min_covalent_angle:
        return False, nearest_atom, neighbor_atom
    else:
        return True, nearest_atom, neighbor_atom

@njit(parallel=True, cache=True)
def check_all_covalent(
    feature_frac_coords,
    atom_frac_coords,
    atom_cart_coords,
    frac2cart,
    min_covalent_angle,
        ):
    # create an array to store if each feature is covalent
    covalent_features = np.zeros(len(feature_frac_coords), dtype=np.bool_)
    atom_neighs = np.empty((len(feature_frac_coords), 2), dtype=np.uint16)
    for i in prange(len(feature_frac_coords)):
        feature_frac_coord = feature_frac_coords[i]
        in_tolerance, atom0, atom1 = check_covalent(
            feature_frac_coord,
            atom_frac_coords,
            atom_cart_coords,
            frac2cart,
            min_covalent_angle,
            )
        covalent_features[i] = in_tolerance
        atom_neighs[i] = (atom0, atom1)
    return covalent_features, atom_neighs
    

@njit(parallel=True, cache=True)
def get_feature_edges(
    labeled_array: NDArray[np.int64],
    feature_map: NDArray[np.int64],
    neighbor_transforms: NDArray[np.int64],
    vacuum_mask: NDArray[np.bool_],
):
    """
    In a 3D array of labeled voxels, finds the voxels that neighbor at
    least one voxel with a different label.

    Parameters
    ----------
    labeled_array : NDArray[np.int64]
        A 3D array where each entry represents the basin label of the point.
    feature_map : NDArray[np.int64]
        A 1D array mapping basin labels to feature labels
    neighbor_transforms : NDArray[np.int64]
        The transformations from each voxel to its neighbors.
    vacuum_mask : NDArray[np.bool_]
        A 3D array representing the location of the vacuum

    Returns
    -------
    edges : NDArray[np.bool_]
        A mask with the same shape as the input grid that is True at points
        on basin edges.

    """
    nx, ny, nz = labeled_array.shape
    # create 3D array to store edges
    edges = np.zeros((nx, ny, nz), dtype=np.bool_)
    # loop over each voxel in parallel
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # if this voxel is part of the vacuum, continue
                if vacuum_mask[i, j, k]:
                    continue
                # get this voxels feature
                basin = labeled_array[i, j, k]
                feature_label = feature_map[basin]
                # iterate over the neighboring voxels
                for si, sj, sk in neighbor_transforms:
                    # wrap points
                    ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                    # get neighbors feature label
                    neigh_basin = labeled_array[ii, jj, kk]
                    neigh_feature_label = feature_map[neigh_basin]
                    # if any label is different, the current voxel is an edge.
                    # Note this in our edge array and break
                    # NOTE: we also check that the neighbor is not part of the
                    # vacuum
                    if neigh_feature_label != feature_label and not vacuum_mask[ii, jj, kk]:
                        edges[i, j, k] = True
                        break
    return edges



@njit(cache=True, fastmath=True)
def get_min_avg_feat_surface_dists(
    labels,
    feature_map,
    frac_coords,
    edge_mask,
    matrix,
    max_value,
):
    nx, ny, nz = labels.shape
    # create array to store best dists, sums, and counts
    dists = np.full(len(frac_coords), max_value, dtype=np.float64)
    dist_sums = np.zeros(len(frac_coords), dtype=np.float64)
    edge_totals = np.zeros(len(frac_coords), dtype=np.uint32)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # skip outside edges
                if not edge_mask[i,j,k]:
                    continue
                # get feature label at edge
                feature_label = feature_map[labels[i, j, k]]
                # add to our count
                edge_totals[feature_label] += 1
                # convert from voxel indices to frac
                fi = i / nx
                fj = j / ny
                fk = k / nz
                # calculate the distance to the appropriate frac coord
                ni, nj, nk = frac_coords[feature_label]
                # get differences between each index
                di = ni - fi
                dj = nj - fj
                dk = nk - fk
                # wrap at edges to be as close as possible
                di -= round(di)
                dj -= round(dj)
                dk -= round(dk)
                # convert to cartesian coordinates
                ci = di * matrix[0, 0] + dj * matrix[1, 0] + dk * matrix[2, 0]
                cj = di * matrix[0, 1] + dj * matrix[1, 1] + dk * matrix[2, 1]
                ck = di * matrix[0, 2] + dj * matrix[1, 2] + dk * matrix[2, 2]
                # calculate distance
                dist = np.linalg.norm(np.array((ci, cj, ck), dtype=np.float64))
                # add to our total
                dist_sums[feature_label] += dist
                # if this is the lowest distance, update radius
                if dist < dists[feature_label]:
                    dists[feature_label] = dist
    # get average dists
    average_dists = dist_sums / edge_totals
    return dists, average_dists
    