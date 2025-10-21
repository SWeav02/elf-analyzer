# -*- coding: utf-8 -*-


import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


# @njit(cache=True)
def check_covalent(
    feature_frac_coord,
    atom_frac_coords,
    atom_cart_coords,
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
        return False, nearest_atom, neighbor_atom
    else:
        return True, nearest_atom, neighbor_atom


# @njit(parallel=True, cache=True)
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
    
    