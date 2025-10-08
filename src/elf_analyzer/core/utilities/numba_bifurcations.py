# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.methods.shared_numba import wrap_point

@njit(cache=True)
def _find_root_compress(parents, x):
    """Find root with partial path compression"""
    while x != parents[x]:
        parents[x] = parents[parents[x]]
        x = parents[x]
    return x

@njit(cache=True)
def _find_root(parents, x):
    """Find root without path compression. Parallel friendly"""
    while x != parents[x]:
        x = parents[x]
    return x

@njit(cache=True)
def _union(parents, x, y):        
    rx = _find_root_compress(parents, x)
    ry = _find_root_compress(parents, y)

    parents[rx] = ry
    
@njit(parallel=True, cache=True)
def _reduce_roots(parents):
    new_parents = np.empty_like(parents, dtype=parents.dtype)
    for i in prange(len(parents)):
        current_val = parents[i]
        if current_val == -1:
            # this basin hasn't been added yet. Note it and continue
            new_parents[i] = -1
            continue
        new_parents[i] = _find_root(parents, current_val)
    return new_parents
    
@njit(cache=True)
def flood_above(
    data,
    flood_labels,
    frontier_lists,
    val_idx,
    possible_elf_values,
    flood_basin_connections,
    cell_transforms,
    neighbor_transforms,
    num_basins,
        ):
    """
    Floods features up to the current value
    """
    # get the label that represents an unfilled point
    max_label = len(flood_basin_connections)
    
    # get the current max elf value and list of points on the frontier for this
    # value
    max_value = possible_elf_values[val_idx]
    frontier_list = frontier_lists[val_idx]
    
    # get supercell shape and single cell shape
    snx, sny, snz = flood_labels.shape
    nx, ny, nz = data.shape
    
    # Now we start rounds of flooding to nearby neighbors
    any_new_connection = False
    while True:
        # create a list for new points added to the frontier
        new_frontier_list = [(-1,-1,-1)]
        is_new_frontier = False
        for i,j,k in frontier_list[1:]: # first is always a placeholder for numba typing
            
            # get the basin label at this index
            label = flood_labels[i,j,k]
            
            # Now we look to each neighbor to determine what needs to be done
            # at each
            for si, sj, sk in neighbor_transforms:
                # get the neighbor in super cell and single cell coordinates
                ii = i+si
                jj = j+sj
                kk = k+sk
                ni = ii % nx
                nj = jj % ny
                nk = kk % nz
                sni, snj, snk = wrap_point(ii, jj, kk, snx, sny, snz)
                # get the value and label at this neighbor
                value = data[ni,nj,nk]
                neigh_label = flood_labels[sni,snj,snk]
                # There are two options for what we want to do here:
                    # 1. Flood to a new unfilled point and add it to our frontier
                    # 2. Note a connection to another group of connected basins
                
                # check if this point is above the current cutoff
                above_cutoff = value >= max_value
                # check if this point has been labeled already
                labeled = neigh_label != max_label
                
                if not labeled:
                    is_new_frontier = True
                    # if we are above the cutoff, we want to add this point to
                    # our current frontier. If not, we want to add it to a future one.
                    # We loop over our possible elf values to find the list index
                    # where this point will become part of our active frontier
                    
                    for next_val_idx, next_val in enumerate(possible_elf_values[val_idx:]):
                        if value >= next_val:
                            if next_val_idx == 0:
                                new_frontier_list.append((sni, snj, snk))
                            else:
                                frontier_lists[next_val_idx+val_idx].append((sni, snj, snk))
                            break
                    # now we flood to this point at all cell transforms
                    for cell_idx, (ci, cj, ck) in enumerate(cell_transforms):
                        # get the correct label for this transform
                        cell_label = label + (num_basins*cell_idx)
                        # get the corresponding point at this transform
                        # NOTE: we want to flood to the supercell coordinate not
                        # the single cell. Otherwise, we would always use the
                        # same labels in each cell which would not give us correct connections
                        sci, scj, sck = wrap_point(sni+ci, snj+cj, snk+ck, snx, sny, snz)
                        flood_labels[sci, scj, sck] = cell_label

                elif labeled and above_cutoff:
                    # we may have a new connection. Get the root group of each
                    # label
                    root = _find_root_compress(flood_basin_connections, label)
                    neigh_root = _find_root_compress(flood_basin_connections, neigh_label)
                    
                    # if our roots match, we don't have a new connection
                    if root == neigh_root:
                        continue
                    any_new_connection = True
                    # otherwise, we want to mark which labels are newly connected
                    for cell_idx, (ci, cj, ck) in enumerate(cell_transforms):
                        # get the correct label for this transform
                        cell_label = label + (num_basins*cell_idx)
                        # get the transformed neighbor and label
                        sci, scj, sck = wrap_point(sni+ci, snj+cj, snk+ck, snx, sny, snz)
                        cell_neigh_label = flood_labels[sci,scj,sck]
                        # make our connection
                        _union(flood_basin_connections, cell_label, cell_neigh_label)

        
        # if we didn't find a new frontier in this round of flooding, we break
        if not is_new_frontier:
            break
        # otherwise, copy our new frontier list over and continue
        frontier_list = new_frontier_list
    # Now that we've finished flooding, we clear the frontier list at this point
    # to save space
    frontier_lists[val_idx] = [(-1,-1,-1)]
    # if we made a new connection, we fully reduce our connection paths.
    # TODO: See if this actually improves speed
    if any_new_connection:
        flood_basin_connections = _reduce_roots(flood_basin_connections)
    
    return flood_labels, frontier_lists, flood_basin_connections
    
@njit(cache=True)
def get_dimensionality(
    flood_labels,
    grid_shape,
    seed_coord,
    flood_basin_connections,
    cell_transforms,
        ):
    """
    Finds the dimensionality of a feature from the provided seed
    """
    
    # max_label = len(flood_basin_connections)
    nx, ny, nz = grid_shape
    connections = [
            # surfaces (3)
            [0, 1],  # x
            [0, 2],  # y
            [0, 3],  # z
            # edges (6)
            [0, 4],  # xy
            [0, 5],  # xz
            [0, 6],  # yz
            [3, 1],  # x-z
            [3, 2],  # y-z
            [1, 2],  # -xy
            # corners (4)
            [0, 7],  # x,y,z
            [1, 6],  # -x,y,z
            [2, 5],  # x,-y,z
            [3, 4],  # x,y,-z
        ]
    # get the coords at the base unit cell
    ox, oy, oz = seed_coord
    # get the labels at each transform
    labels = []
    for i, j, k in cell_transforms:
        # shift coords
        x = ox + i
        y = oy + j
        z = oz + k
        # get label
        # NOTE: There should always be a label here due to how we seed labels
        labels.append(_find_root(flood_basin_connections, flood_labels[x,y,z]))
    
    # create a counter for the number of connections
    connection_num = 0
    # loop over our connection directions and see if the labels match
    for i,j in connections:
        # get labels
        label1 = labels[i]
        label2 = labels[j]
        if label1 == label2:
            connection_num += 1

    # determine the dimensionality.
    if connection_num == 0:
        dimensionality = 0
    elif connection_num == 1:
        dimensionality = 1
    elif 1 < connection_num <= 4:
        dimensionality = 2
    elif 5 < connection_num <= 13:
        dimensionality = 3

    return dimensionality

@njit(parallel=True, cache=True)
def get_dimensionality_all(
    flood_mask,
    grid_shape,
    seed_points,
    flood_basin_connections,
    cell_transforms,
        ):
    """
    Finds the dimensionality of a set of features from the provided seeds
    """
    # create an array to track which points are infinite
    dimensionalities = np.zeros(len(seed_points), dtype=np.int64)
    for seed_idx in prange(len(seed_points)):
        seed_coord = seed_points[seed_idx]
        dimensionalities[seed_idx] = get_dimensionality(
            flood_mask,
            grid_shape,
            seed_coord,
            flood_basin_connections,
            cell_transforms,
            )
    return dimensionalities

@njit(cache=True)
def find_bifurcations(
    connection_pairs,
    connection_elfs,
    basin_maxima_grid,
    data,
    neighbor_transforms,
        ):
    """
    Finds the ELF values at which changes in features or feature dimensionalities
    occur
    """
    nx, ny, nz = data.shape
    num_basins = len(basin_maxima_grid)
    # get all possible elf values (including 0.0) and flip from high to low
    possible_elf_values = np.unique(connection_elfs)
    possible_elf_values = np.append(np.flip(possible_elf_values), [0.0])
    
    ###########################################################################
    # Flood Setup
    ###########################################################################
    # create an array to represent where each basin has flooded to
    flood_labels = np.full((nx*2, ny*2, nz*2), num_basins*8, dtype=np.uint16)
    
    # create lists to store which points on the grid need to be iterated over
    # at each elf value
    frontier_lists = []
    for i in range(len(possible_elf_values)):
        frontier_lists.append([(-1,-1,-1)]) # placeholder to indicate type to numba
    # create a mask to represent which basins have been added to our flood
    flood_basins = np.zeros(len(basin_maxima_grid), dtype=np.bool_)
    # create an array representing which basins are connected to each other in
    # our flood
    flood_basin_connections = np.arange(num_basins*8, dtype=np.uint16)
    # get the transforms to the neighboring unit cells in a 2x2 supercell
    cell_transforms = np.array([
            [0, 0, 0],  # -
            [nx, 0, 0],  # x
            [0, ny, 0],  # y
            [0, 0, nz],  # z
            [nx, ny, 0],  # xy
            [nx, 0, nz],  # xz
            [0, ny, nz],  # yz
            [nx, ny, nz],  # xyz
        ], dtype=np.int64)

    ###########################################################################
    # Basin Connections Setup
    ###########################################################################
    # create lists to store each bifurcation
    bifurcation_values = []
    bifurcation_features = []
    bifurcation_feature_indices = []
    bifurcation_dimensionalities = []
    
    # create an array representing which basins are connected to one another
    # at a given value
    basin_connections = np.full(len(basin_maxima_grid), -1, dtype=np.int64)
    # and an array pointing each basin group to its corresponding list index
    # at a given value
    feature_group_indices = basin_connections.copy()
    
    # create an empty set of feature groups representing the groups above the
    # highest value. 
    feature_groups = [[-1]] # -1 is just for typing as numba dislikes the empty list
    feature_ids = [-1]
    # do the same for dimensionalities of the features
    new_dimensionalities = np.array([-1], dtype=np.int64)
    # create a counter for the total number of unique features
    unique_features = 0
    
    ###########################################################################
    # Begin loop
    ###########################################################################
    # loop over elf values from high to low
    previous_elf_value = 1.0e12 # make unreasonably large
    for val_idx, elf_value in enumerate(possible_elf_values):
        
        #######################################################################
        # Find groups of connected basins
        #######################################################################
        # get the new connections that exist at this value
        connection_indices = np.where((connection_elfs>=elf_value) & (connection_elfs < previous_elf_value))
        current_connections = connection_pairs[connection_indices]
        # reset our connections and groups.
        # NOTE: connections don't need to be reset as they will never disappear
        # as we loop downwards in ELF value
        feature_group_indices[:] = -1

        # Loop over basin connections
        for i, j in current_connections:
            # ensure both basins have a connection
            for basin_idx in (i,j):
                if basin_connections[basin_idx] == -1:
                    basin_connections[basin_idx] = basin_idx
            
            # get the roots of each connection
            lower_root = _find_root_compress(basin_connections, i)
            upper_root = _find_root_compress(basin_connections, j)
            
            # if we have the same root, we dont have a new connection
            if lower_root == upper_root:
                continue
            # otherwise we note the new connection
            _union(basin_connections, i, j)

        # reduce our connections
        basin_connections = _reduce_roots(basin_connections)

        # copy our previous groups
        previous_groups = [i for i in feature_groups]
        # create lists to store features basin groups
        feature_groups = []
        feature_seeds = []
        num_features = 0
        
        # get our frontier list
        frontier_list = frontier_lists[val_idx]
        # Now loop over our basin connections and get our groups
        for basin_idx, basin_group in enumerate(basin_connections):
            # skip if this basin isn't assigned
            if basin_group == -1:
                continue
            
            # check if this basin has been seeded already
            seed_point = basin_maxima_grid[basin_idx]
            if not flood_basins[basin_idx]:
                # note that we've seeded this basin
                flood_basins[basin_idx] = True
                # add is point to our frontier list, only at the origin unit cell
                frontier_list.append((seed_point[0],seed_point[1],seed_point[2],))
                # seed this basin in each unit cell in the supercell
                for trans_idx, trans in enumerate(cell_transforms):
                    x, y, z = seed_point + trans
                    # get the adjusted basin index for this cell
                    cell_basin_idx = basin_idx + trans_idx*num_basins
                    # add it to our flood
                    flood_labels[x,y,z] = cell_basin_idx

            
            # check if this group exists yet
            if feature_group_indices[basin_group] == -1:
                # create a new feature
                feature_groups.append([basin_idx])
                feature_group_indices[basin_group] = num_features
                num_features += 1
                # add a seed point for this feature
                feature_seeds.append(seed_point)
            
            # otherwise, append it to the existing feature
            else:
                group_index = feature_group_indices[basin_group]
                feature_groups[group_index].append(basin_idx)

                
        # We now have the groups that exist at the current level. We want to
        # check if they are different from the last set of groups. By construction,
        # the groups will always be ordered from lowest basin to highest
        
        # copy our previous group ids
        previous_feature_ids = [i for i in feature_ids]
        feature_ids = []
        same_groups = True
        for group in feature_groups:
            # check to see if this group exists in the previous groups
            group_found = False
            for pgroup, pgroup_idx in zip(previous_groups, previous_feature_ids):
                if len(pgroup) != len(group):
                    continue
                # check if all entries in the group equal the other group
                is_previous = True
                for i, j in zip(group, pgroup):
                    if i!=j:
                        is_previous = False
                        break
                if is_previous:
                    # note we found a previous group and add the
                    # features index
                    group_found = True
                    feature_ids.append(pgroup_idx)
                    break
            if not group_found:
                # Note we found a new feature
                same_groups = False
                feature_ids.append(unique_features)
                unique_features += 1
        
        #######################################################################
        # Flood fill and get dimensionalities
        #######################################################################
        
        # We also want to calculate the dimensionality of each feature, regardless
        # of if we have a new group or not.
        # Flood fill to the current value
        flood_labels, frontier_lists, flood_basin_connections = flood_above(
            data,
            flood_labels,
            frontier_lists,
            val_idx,
            possible_elf_values,
            flood_basin_connections,
            cell_transforms,
            neighbor_transforms,
            num_basins,
            )
        # get the new dimensionalities
        old_dimensionalities = new_dimensionalities.copy()
        new_dimensionalities = get_dimensionality_all(
            flood_labels,
            data.shape,
            feature_seeds,
            flood_basin_connections,
            cell_transforms,
            )
        
        # if this is a new set of basin groups, we append no matter what
        if not same_groups:
            bifurcation_values.append(elf_value)
            bifurcation_features.append(feature_groups)
            bifurcation_feature_indices.append(feature_ids)
            bifurcation_dimensionalities.append(new_dimensionalities)
        # otherwise, we only append if we found new dimensionalities
        elif not np.all(old_dimensionalities==new_dimensionalities):
            bifurcation_values.append(elf_value)
            bifurcation_features.append(feature_groups)
            bifurcation_feature_indices.append(feature_ids)
            bifurcation_dimensionalities.append(new_dimensionalities)
        
        # update our previous elf value
        previous_elf_value = elf_value
            
        # BUG: This is returning only one value for each currently
    return bifurcation_values, bifurcation_features, bifurcation_feature_indices, bifurcation_dimensionalities
        
        
    
    
    
    
    
    
    
    