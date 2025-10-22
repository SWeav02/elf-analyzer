# -*- coding: utf-8 -*-

import math

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

from baderkit.core.toolkit.grid_numba import interp_nearest, interp_spline

@njit(parallel=True, cache=True)
def get_nearest_neighbors(
        atom_frac_coords,
        atom_cart_coords,
        frac2cart,
        ):
    
    # create arrays to store results
    best_dists = np.full(len(atom_frac_coords), 100.0, dtype=np.float64)
    best_neighs = np.empty(len(atom_frac_coords), dtype=np.int64)
    best_images = np.empty((len(atom_frac_coords), 3), dtype=np.int64)

    # loop over each fractional coordinate. Transform it to neighboring
    # cells. Check distance to each neighbor.
    for i in prange(len(atom_frac_coords)):
        fi, fj, fk = atom_frac_coords[i]
        # loop over transformations
        for si in (-1, 0, 1):
            for sj in (-1, 0, 1):
                for sk in (-1, 0, 1):
                    # transform frac coord
                    ti = fi + si
                    tj = fj + sj
                    tk = fk + sk
                    # convert to cartestian coord
                    ci = ti * frac2cart[0][0] + tj * frac2cart[1][0] + tk * frac2cart[2][0]
                    cj = ti * frac2cart[0][1] + tj * frac2cart[1][1] + tk * frac2cart[2][1]
                    ck = ti * frac2cart[0][2] + tj * frac2cart[1][2] + tk * frac2cart[2][2]
                    # compare distance to each neighbor
                    for j, (nci, ncj, nck) in enumerate(atom_cart_coords):
                        # skip if this is the current coord
                        if j == i and si==0 and sj==0 and sk==0:
                            continue
                        # otherwise, calculate the distance
                        dist = ((nci-ci)**2 + (ncj-cj)**2 + (nck-ck)**2) ** 0.5
                        # if its lower than previous calculated distances, update
                        # our entry
                        if dist < best_dists[i]:
                            best_dists[i] = dist
                            best_neighs[i] = j
                            best_images[i] = (-si, -sj, -sk)
    return best_neighs, best_dists, best_images

@njit(parallel=True, cache=True)
def get_ionic_radius(
    data,
    feature_labels,
    atom_idx,
    atom_coords,
    neigh_coords,
    covalent_labels,
    bond_dist,
    line_res: int = 20,
        ):
    # get the number of points to interpolate
    num_points = int(round(bond_dist*line_res))
    # I want this to always be odd because it is common for the exact midpoint
    # to be the correct fraction. This isn't required, but results in clean
    # values in these cases
    if num_points % 2 == 1:
        num_points += 1
    
    # get the vector pointing from each point along the line to the next
    step_vec = (neigh_coords - atom_coords) / (num_points - 1)

    # create arrays to store the values and labels along the bond
    values = np.empty(num_points, dtype=np.float64)
    labels = np.empty(num_points, dtype=np.float64)
    # calculate the positions, values, and labels along the line in parallel
    for point_idx in prange(num_points):
        x, y, z = atom_coords + float(point_idx) * step_vec
        values[point_idx] = interp_spline(x, y, z, data)
        labels[point_idx] = interp_nearest(x, y, z, feature_labels)
        
    # get the unique labels
    unique_labels = np.unique(labels)

    # SITUATION 1:
        # The atom's nearest neighbor is a translation of itself. The radius
        # must always be halfway between the two.
    if len(unique_labels) == 1:
        return 0.5 * bond_dist
        
    # SITUATION 2:
        # The atom is covalently bonded to its nearest neighbor. The radius is
        # the closest local maximum to the center of the bond
    covalent = False
    for label in unique_labels:
        if label in covalent_labels:
            covalent = True
            break
    if covalent:
        use_maximum = True
        # create placeholders for best maxima
        radius_index = -1
        maxima_dist = 1.0e6
        # create a tracker for the last point that belongs to this site
        last_idx = 0
        # get local maxima that are covalent
        midpoint = (len(values) - 1) / 2
        for i, (value, label) in enumerate(zip(values, labels)):
            # skip points that aren't part of the covalent bond
            if label not in covalent_labels:
                continue
            # if this point is assigned to the current atom, update our idx
            if label == atom_idx:
                last_idx = i
            # check if the point is a maximum
            if ((i == 0) or (values[i - 1] <= value)) and ((i == len(values) - 1) or (value > values[i + 1])):
                # if the maximum is closer to the midpoint than previous points,
                # update our best distance
                dist = abs(i-midpoint)
                if dist < maxima_dist:
                    maxima_dist = dist
                    radius_index = i
        # make sure we found a maximum. If not, default to the last point that
        # belongs to the current atom
        if radius_index == -1:
            radius_index = last_idx
            use_maximum = False
    
    # SITUATION 3:
        # The atom is ionically bonded to its nearest neighbor. The radius is
        # the closest local minimum to the center of the bond
    # NOTE: There is also a situation where there is a metallic bond between
    # the atoms. At the moment, I don't mark bare electrons as metallic/electride
    # until after this stage so I can't take that into account. Is this fixable?
    else:
        use_maximum = False
        radius_index = -1
        # find the first point that doesn't belong to the current atom
        for i, label in enumerate(labels):
            if label != atom_idx:
                radius_index = i
                break
        # make sure we found an assignment
        assert radius_index != -1
    
    # Now we want to refine the radius. First, we get the coordinate of the
    # current radius
    current_coord = atom_coords + radius_index*step_vec
    ci, cj, ck = current_coord
    current_value = interp_spline(ci, cj, ck, data)
    # This point must be within one index of the true radius. We iteratively
    # move a step closer to the true value, dividing the step by two after each
    # iteration
    # calculate number of steps needed to reach required resolution
    # res = 1/line_res * 0.5^n
    resolution = 1e-8 # angstroms
    n = round(math.log(resolution * line_res) / math.log(1/2))
    step_mult = 1.0
    for i in range(n):
        step_mult /= 2.0
        step = step_vec * step_mult
        # get value above and below
        ucoord = current_coord + step
        ui, uj, uk = ucoord
        dcoord = current_coord - step
        di, dj, dk = dcoord
        up_val = interp_spline(ui, uj, uk, data)
        down_val = interp_spline(di, dj, dk, data)
        if use_maximum:
            # check which value is the highest and adjust our coord
            if up_val > current_value and up_val >= down_val:
                current_value = up_val
                # update coord
                current_coord = ucoord
                # update index
                radius_index += step_mult
            elif down_val > current_value:
                current_value = down_val
                current_coord = dcoord
                radius_index -= step_mult
        else:
            # check which value is the lowest and adjust our coord
            if up_val < current_value and up_val <= down_val:
                current_value = up_val
                # update coord
                current_coord = ucoord
                # update index
                radius_index += step_mult
            elif down_val >= current_value:
                current_value = down_val
                current_coord = dcoord
                radius_index -= step_mult
    
    # We now have a refined radius. Calculate the actual bond distance
    bond_frac = radius_index / (num_points - 1)
    return bond_frac * bond_dist
    
@njit(cache=True)
def get_ionic_radii(
    equivalent_atoms,
    data,
    feature_labels,
    atom_frac_coords,
    neighbor_indices,
    neighbor_dists,
    neighbor_images,
    covalent_labels: NDArray,
        ):
    # get the unique atoms we need to calculate radii for
    unique_atoms = np.unique(equivalent_atoms)
    
    # create array to store radii
    atomic_radii = np.empty(len(atom_frac_coords), dtype=np.float64)    
    
    # get the radius for each atom. NOTE: We don't do this in parallel because
    # we want the interpolation to be done in parallel instead
    for atom_idx in unique_atoms:
        atom_coords = atom_frac_coords[atom_idx]
        neigh_idx = neighbor_indices[atom_idx]
        bond_dist = neighbor_dists[atom_idx]
        neigh_image = neighbor_images[atom_idx]
        
        # get the neighbors frac coords
        neigh_coords = atom_frac_coords[neigh_idx] + neigh_image
        
        # get the radius for this atom
        radius = get_ionic_radius(
            data,
            feature_labels,
            atom_idx,
            atom_coords,
            neigh_coords,
            covalent_labels,
            bond_dist,
                )
        atomic_radii[atom_idx] = radius
        
    # update values for equivalent atoms
    for atom_idx in range(len(atomic_radii)):
        equiv_atom = equivalent_atoms[atom_idx]
        atomic_radii[atom_idx] = atomic_radii[equiv_atom]
        
    return atomic_radii
    
    