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

@njit(cache=True)
def check_covalent(
    feature_frac_coord,
    atom_frac_coords,
    atom_cart_coords,
    frac2cart,
    min_covalent_bond_ratio,
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

    # check if the ratio is within our tolerance
    nearest_dist = atom_dists[nearest_atom]
    neighbor_dist = atom_dists[neighbor_atom]
    covalent_bond_ratio = nearest_dist / neighbor_dist # always 0-1
    if covalent_bond_ratio < min_covalent_bond_ratio:
        return False
    
    # Now check if the point is within a reasonable angle. First, get our points
    # in cartesian coordinates. This is corresponds to:
        # θ = arccos((A ⋅ B) / (|A|*|B|))
    # where A and B are the vectors from the feature to each neighboring atom
    A = atom_vecs[nearest_atom]
    B = atom_vecs[neighbor_atom]

    cos_theta = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    # make sure our theta is within the bounds of arcos
    cos_theta = max(-1.0, min(1.0, cos_theta))
    # get theta
    theta = np.arccos(cos_theta)
    # check if our angle is above our tolerance
    if theta > min_covalent_angle:
        return True
    else:
        return False

@njit(parallel=True, cache=True)
def check_all_covalent(
    feature_frac_coords,
    atom_frac_coords,
    atom_cart_coords,
    frac2cart,
    min_covalent_bond_ratio,
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
            frac2cart,
            min_covalent_bond_ratio,
            min_covalent_angle,
            )
    return covalent_features
    

###############################################################################
# Checking if Features Surround Atoms
###############################################################################

@njit(cache=True)
def check_surrounded(
    flood,
    grid_shape,
    atom_coord,
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
    
    ox, oy, oz = atom_coord
    surrounded = True
    for i,j,k in transforms:
        # shift coords
        x = ox + i*nx
        y = oy + j*ny
        z = oz + k*nz
        flood_label = flood[x,y,z]
        # if this site wasn't fully surrounded, it will have been flooded and
        # will have a value other than -1
        if flood_label == -1:
            continue
        surrounded = False
        break
    return surrounded

@njit(parallel=True, cache=True)
def check_surrounded_all(
    flood,
    grid_shape,
    atom_coords,
        ):
    # create a boolean array to store which atoms are surrounded
    surrounded_atoms = np.zeros(len(atom_coords), dtype=np.bool_)
    for atom_idx in prange(len(atom_coords)):
        atom_coord = atom_coords[atom_idx]
        surrounded_atoms[atom_idx] = check_surrounded(
            flood,
            grid_shape,
            atom_coord,
                )
    return surrounded_atoms

@njit(parallel=True, cache=True)
def flood_fill_round(
    flood,
    flood_edge,
    flood_frontier,
    edge_indices,
    data,
    max_value,
    neighbor_transforms,
    atom_connections,
    basin_flood_mask,
    basin_labels,
        ):
    # get supercell shape and single cell shape
    snx, sny, snz = flood.shape
    nx, ny, nz = data.shape
    
    for edge_idx in prange(len(edge_indices)):
        i,j,k = edge_indices[edge_idx]
        atom_label = atom_connections[flood[i,j,k]]
        is_edge = False
        for si, sj, sk in neighbor_transforms:
            ni = i + si
            nj = j + sj
            nk = k + sk
            # wrap around small cell
            nis = ni % nx
            njs = nj % ny
            nks = nk % nz
            
            value = data[nis, njs, nks]
            basin_label = basin_labels[nis, njs, nks]
            # if this point is below our limit, fill it.
            # <= is correct here because the point that == max_value is part
            # of the old set of features.
            if value <= max_value or basin_flood_mask[basin_label]:
                # wrap around large cell
                nil, njl, nkl = wrap_point(ni, nj, nk, snx, sny, snz)
                # check if we've already flooded this point, and if not, flood
                # to it and add it as a new edge
                new_label = flood[nil, njl, nkl]
                if new_label == -1:
                    flood[nil, njl, nkl] = atom_label
                    flood_frontier[nil, njl, nkl] = True
                # otherwise, check if the new label belongs to a different set of
                # atoms.
                # if so, we mark this point as an edge so we can find that connection
                # at the end of this flood fill
                else:
                    new_atom_label = atom_connections[new_label]
                    if atom_label != new_atom_label:
                        is_edge = True
                

                        
            else:
                is_edge = True
        # remove this point from our frontier so we don't search it again
        flood_frontier[i,j,k] = False
        # if this is an edge, add it to our flood edge
        flood_edge[i,j,k] = is_edge
    return flood, flood_frontier, flood_edge

@njit(cache=True)
def flood_fill(
    flood,
    flood_edge,
    data,
    max_value,
    neighbor_transforms,
    atom_connections,
    basin_flood_mask,
    basin_labels,
        ):
    # create our initial values to flood from (our frontier)
    flood_frontier = flood_edge.copy()
    while True:
        edge_indices = np.argwhere(flood_frontier)
        if len(edge_indices) == 0:
            break
        flood, flood_frontier, flood_edge = flood_fill_round(
            flood, 
            flood_edge, 
            flood_frontier, 
            edge_indices, 
            data, 
            max_value, 
            neighbor_transforms,
            atom_connections,
            basin_flood_mask,
            basin_labels,
            )
    return flood, flood_edge

@njit(cache=True)
def get_connected_atoms(
    flood,
    flood_edge,
    atom_connections,
    neighbor_transforms,
        ):
    nx,ny,nz = flood.shape
    edge_indices = np.argwhere(flood_edge)
    
    # create an array to store a single edge point for each atom group. This will
    # be used later to check what feature surrounds that atom group
    atom_edge_points = np.empty((len(atom_connections), 3), dtype=np.int64)
    atom_point_selected = np.zeros_like(atom_connections, dtype=np.bool_)
    
    for i,j,k in edge_indices:
        # get this sites atom label
        atom_label = atom_connections[flood[i,j,k]]
        # note if we've already given it an edge
        has_point = atom_point_selected[atom_label]
        # check neighbors
        is_edge = False
        for si, sj, sk in neighbor_transforms:
            # get neighbor and wrap
            ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
            # get neighbors label
            neigh_label = flood[ni,nj,nk]
            if neigh_label == -1:
                # note that this is a true edge
                is_edge = True
                # if we haven't assigned an edge point yet, do so
                if not has_point:
                    atom_edge_points[atom_label] = (ni,nj,nk)
                    atom_point_selected[atom_label] = True
                    has_point = True
                continue
            else:
                neigh_atom_label = atom_connections[neigh_label]
                if neigh_atom_label != atom_label:
                    atom_connections[atom_label] = min(atom_label, neigh_atom_label)
        if not is_edge:
            flood_edge[i,j,k] = False
        
    # Now reduce atom connections. This works in one go because our atom indices
    # are sorted from lowest to highest
    for i, connection in enumerate(atom_connections):
        atom_connections[i] = atom_connections[atom_connections[i]]
    return flood_edge, atom_connections, atom_edge_points

@njit(parallel=True, cache=True)
def get_features_surrounding_atoms(
    atom_edge_points,
    flood,
    basin_labels,
    basin_map,
    surrounded_atoms,
    atom_connections,
    neighbor_transforms,
    basin_flood_mask,
        ):
    nx, ny, nz = basin_labels.shape
    # create an array to store what feature surrounds an atom
    atom_features = np.full_like(surrounded_atoms, -1, dtype=np.int64)
    # loop over atoms and if they are surrounded, check by what feature
    for atom_idx in prange(len(surrounded_atoms)):
        surrounded=surrounded_atoms[atom_idx]
        atom_group=atom_connections[atom_idx]
        # skip this atom if its not surrounded
        if not surrounded:
            continue
        # check if this atom group already has an assignment
        if atom_features[atom_group] != -1:
            atom_features[atom_idx] = atom_features[atom_group]
            continue
        # if we've made it here, we need to get the surrounding features
        i, j, k = atom_edge_points[atom_group]
        for si, sj, sk in neighbor_transforms:
            # get neighbor and wrap
            ni = (i+si) % nx
            nj = (j+sj) % ny
            nk = (k+sk) % nz
            basin_label = basin_labels[ni,nj,nk]
            # if this point is part of our flood, skip it
            if flood[ni,nj,nk] != -1:
                continue
            # if this point is not part of one of the new features at this elf
            # value, continue
            if basin_flood_mask[basin_label]:
                continue
            feature = basin_map[basin_label]
            atom_features[atom_group] = feature
            break # It shouldn't be possible to have a different feature after this
    return atom_features
        
@njit(parallel=True, cache=True)
def get_surrounded_atoms(
    reducible_keys,
    basins,
    appear_at,
    disappear_at,
    basin_labels,
    num_basins,
    data,
    atom_grid_coords,
    neighbor_transforms,
        ):
    shape = np.array(data.shape, dtype=np.int64)
    nx, ny, nz = shape
    unique_elf_values = np.unique(appear_at)

    # create supercell array to represent flood fill
    flood = np.full((nx*2, ny*2, nz*2), -1, dtype=np.int64)
    flood_edge = np.zeros_like(flood, dtype=np.bool_)
    # initialize atom coords in flood
    atom_idx = 0
    for ai, aj, ak in atom_grid_coords:
        flood[ai, aj, ak] = atom_idx
        atom_idx += 1
        flood_edge[ai, aj, ak] = True

    # create an array to track which atoms have connected to each other
    atom_connections = np.arange(len(atom_grid_coords))

    # create a list to track which atoms each feature surrounds
    feature_atoms = []
    feature_keys  = []
    # create array to represent which features have already been checked
    checked_features = np.zeros(max(reducible_keys)+1, dtype=np.bool_)
    checked_features_idx = np.full_like(checked_features, -1, dtype=np.int64)

    # create an array to map basins to the feature they belong to at each elf
    # value
    basin_map = np.full(num_basins, -1, dtype=np.int64)
    num_features = 0

    for max_value in unique_elf_values:
        # create a list to store the current features
        current_features = []

        for key, appears, disappears, basin_list in zip(reducible_keys, appear_at, disappear_at, basins):
            if disappears == max_value:
                # this feature just disappeared. We want to remove it from the
                # basin map. This must be done before any child reducible features
                # are added. Luckily, the features are already ordered by when they
                # appear
                basin_map[basin_list] = -1
                num_features -= 1
            elif appears == max_value:
                # This feature just appeared. We want to add it to our basin map.
                basin_map[basin_list] = key
                num_features += 1
                current_features.append(key)
        
        # create an array to indicate labels that should be flooded regardless of
        # their elf value. This will include any reducible basins
        basin_flood_mask = np.where(basin_map == -1, True, False)
        
        # flood fill
        flood, flood_edge = flood_fill(
            flood,
            flood_edge,
            data,
            max_value,
            neighbor_transforms,
            atom_connections,
            basin_flood_mask,
            basin_labels,
            )
        # check for atoms that flooded into one another
        flood_edge, atom_connections, atom_edge_points = get_connected_atoms(
            flood,
            flood_edge,
            atom_connections,
            neighbor_transforms,
            )
        # check which atoms are surrounded
        surrounded_atoms = check_surrounded_all(
            flood, shape, atom_grid_coords)
        # if none are surrounded, we are finished
        if np.all(~surrounded_atoms):
            break
        # Now we want to get which features are fully surrounded. However, it is
        # possible for multiple features to fully surround the same atoms (e.g. multiple shells).
        # Therefore, we will need to check which features surround the atom, flood
        # fill those, and check again iteratively. However, we still want to use the
        # current flood in our next loop, so we copy it here.
        flood_temp = flood.copy()
        flood_edge_temp = flood_edge.copy()
        atom_connections_temp = atom_connections.copy()
        
        
        while True:
            # get the features surrounding each atom
            feature_assignments = get_features_surrounding_atoms(
                atom_edge_points,
                flood_temp,
                basin_labels,
                basin_map,
                surrounded_atoms,
                atom_connections_temp,
                neighbor_transforms,
                basin_flood_mask,
                    )
            # add each features surrounded atoms to our list. Also track which feature
            # we have a result for
            features_w_results = []
            feature_atom_labels = []
            for atom_idx, feature_key in enumerate(feature_assignments):
                if feature_key == -1:
                    continue
                # note this feature surrounds some atoms
                if not feature_key in features_w_results:
                    features_w_results.append(feature_key)
                    feature_atom_labels.append(atom_connections_temp[atom_idx])
                    
                # We may have already found that this feature surrounds an atom in
                # a previous round. If so, we don't want to do anything else
                if checked_features[feature_key]:
                    continue
                
                # add this atom to the corresponding feature
                key_idx = checked_features_idx[feature_key]
                if key_idx == -1:
                    feature_keys.append(feature_key)
                    feature_atoms.append([atom_idx])
                    checked_features_idx[feature_key] = len(feature_keys) - 1
                else:
                    feature_atoms[key_idx].append(atom_idx)

            
            # if we've found results for all features, or no results, we can break
            if len(features_w_results) == num_features or len(features_w_results) == 0:
                break
            
            # indicate these features should be flooded regardless of elf value and
            # add them to our flood/flood edge
            for key, atom_label in zip(features_w_results, feature_atom_labels):
                # get basins belonging to this feature
                basin_indices = np.where(basin_map==key)[0]
                # mark them to be flooded
                basin_flood_mask[basin_indices] = True
                # get coordinates of this basin and add them to our flood/edge
                basin_coords = np.argwhere(np.isin(basin_labels, basin_indices))
                for coord_idx in prange(len(basin_coords)):
                    bi, bj, bk = basin_coords[coord_idx]
                    flood_edge_temp[bi, bj, bk] = True
                    flood_temp[bi, bj, bk] = atom_label
            
            # flood fill
            flood_temp, flood_edge_temp = flood_fill(
                flood_temp,
                flood_edge_temp,
                data,
                max_value,
                neighbor_transforms,
                atom_connections_temp,
                basin_flood_mask,
                basin_labels,
                )
            # check for atoms that flooded into one another
            flood_edge_temp, atom_connections_temp, atom_edge_points = get_connected_atoms(
                flood_temp,
                flood_edge_temp,
                atom_connections_temp,
                neighbor_transforms,
                )
            # check which atoms are surrounded
            surrounded_atoms = check_surrounded_all(
                flood_temp, shape, atom_grid_coords)
            # if none are surrounded, we are finished
            if np.all(~surrounded_atoms):
                break
        for feature in current_features:
            checked_features[feature] = True
    return feature_keys, feature_atoms
    
@njit(parallel=True, cache=True)
def flood_above(
    data,
    flood_mask,
    edge_mask,
    max_value,
    neighbor_transforms,
        ):
    # get supercell shape and single cell shape
    snx, sny, snz = flood_mask.shape
    nx, ny, nz = data.shape
    
    # create a frontier mask for tracking as we go
    frontier_mask = edge_mask.copy()
    
    # flood until no change
    while True:
        frontier_indices = np.argwhere(frontier_mask)
        print(len(frontier_indices))
        if len(frontier_indices) == 0:
            break
        for frontier_idx in prange(len(frontier_indices)):
            i,j,k = frontier_indices[frontier_idx]
            is_edge = False
            for si, sj, sk in neighbor_transforms:
                # get the neighbor in super cell coordinates and single cell
                ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
                sni, snj, snk = wrap_point(i+si, j+sj, k+sk, snx, sny, snz)
                # check if this neighbor is above the required value
                value = data[ni,nj,nk]
                in_bound = value >= max_value
                # if this isn't in bounds, our current point is on an edge
                if not in_bound:
                    is_edge = True
                    continue
                # if this point is not already flooded, flood it and add to our edge
                in_flood = flood_mask[sni,snj,snk]
                if not in_flood:
                    flood_mask[sni,snj,snk] = True
                    frontier_mask[sni,snj,snk] = True
            # remove this point from our frontier so we don't search it again
            frontier_mask[i,j,k] = False
            # note if this is an edge
            edge_mask[i,j,k] = is_edge
    return flood_mask, edge_mask
    
@njit(cache=True)
def get_dimensionality(
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
    connection_num = 0
    for i,j,k in transforms:
        # shift coords
        x = ox + i*nx
        y = oy + j*ny
        z = oz + k*nz
        # check if transformed coord is in the flood
        if flood_mask[x,y,z]:
            connection_num += 1
    # determine the dimensionality.
    if connection_num == 0:
        dimensionality = 0
    elif connection_num == 1:
        dimensionality = 1
    elif connection_num == 3:
        dimensionality = 2
    else:
        dimensionality = 3

    return dimensionality

@njit(parallel=True, cache=True)
def get_dimensionality_all(
    flood_mask,
    grid_shape,
    seed_points,
        ):
    # create an array to track which points are infinite
    dimensionalities = np.zeros(len(seed_points), dtype=np.int64)
    for seed_idx in prange(len(seed_points)):
        seed_coord = seed_points[seed_idx]
        dimensionalities[seed_idx] = get_dimensionality(
            flood_mask,
            grid_shape,
            seed_coord
            )
    return dimensionalities

# @njit
def get_dimensionality_bifurcations(
    possible_elf_values,
    important_values,
    below_groups,
    above_groups,
    data,
    basin_maxima_grid,
    neighbor_transforms,
        ):
    nx, ny, nz = data.shape
    # create a mask to flood fill
    flood_mask = np.zeros((nx*2, ny*2, nz*2), dtype=np.bool_)
    flood_edge = np.copy(flood_mask)
    # create lists to store new elf values and groups
    new_values = []
    new_below_groups = []
    new_above_groups = []
    dimensionalities = []
    # create a mask for basins that have been added to our flood
    added_basins = np.zeros(len(basin_maxima_grid), dtype=np.bool_)
    
    current_groups_idx = -1
    new_dimensionalities = np.array([], dtype=np.int64)
    # loop over our elf values
    for elf_value in possible_elf_values:
        # get the basin groups at this value. This is the first set that exists
        # right at or below this value
        new_groups_idx = 0
        for groups_idx, value in enumerate(important_values):
            if value > elf_value:
                break
            new_groups_idx = groups_idx
        # Note if this is a new set of groups
        is_new_groups = current_groups_idx != new_groups_idx
        if is_new_groups:
            # get our current groups
            current_below_groups = below_groups[new_groups_idx]
            current_above_groups = above_groups[new_groups_idx]
            current_groups_idx = new_groups_idx
        
        # get a seed for each group
        seeds = np.empty((len(current_below_groups), 3), dtype=np.int64)
        for group_idx, group in enumerate(current_below_groups):
            seeds[group_idx] = basin_maxima_grid[group[0]]
            # also seed each basin if not done already
            for basin_idx in group:
                if added_basins[basin_idx]:
                    continue
                added_basins[basin_idx] = True
                i, j, k = basin_maxima_grid[basin_idx]
                flood_mask[i,j,k] = True
                flood_edge[i,j,k] = True
        # flood to the current value
        flood_mask, flood_edge = flood_above(
            data,
            flood_mask,
            flood_edge,
            elf_value,
            neighbor_transforms,
            )
        # get the dimensionalities for each group
        old_dimensionalities = new_dimensionalities.copy()
        new_dimensionalities = get_dimensionality_all(
            flood_mask,
            data.shape,
            seeds,
            )
        # if this is a new set of basin groups, we append no matter what
        if is_new_groups:
            new_values.append(elf_value)
            new_below_groups.append(current_below_groups)
            new_above_groups.append(current_above_groups)
            dimensionalities.append(new_dimensionalities)
            continue
        # otherwise, we only append if we found new dimensionalities
        if not np.all(old_dimensionalities==new_dimensionalities):
            new_values.append(elf_value)
            new_below_groups.append(current_below_groups)
            new_above_groups.append(current_above_groups)
            dimensionalities.append(new_dimensionalities)
    return new_values, new_below_groups, new_above_groups, dimensionalities
    
    