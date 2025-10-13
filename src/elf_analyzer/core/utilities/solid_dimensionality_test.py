# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 10:07:38 2025

@author: Sam
"""

import numpy as np
from numba import njit

@njit(cache=True)
def find_root(parent, x):
    """Find root with partial path compression"""
    while x != parent[x]:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

@njit(fastmath=True, cache=True)
def find_root_with_shift(parent, offset, x):
    """Find root of x, applying path compression for both parent and offset."""
    if parent[x] != x:
        root, off = find_root_with_shift(parent, offset, parent[x])
        if off[0] != 0 or off[1] != 0 or off[2] != 0:
            offset[x] += off
        parent[x] = root
        return root, offset[x]
    else:
        return x, offset[x]

@njit(cache=True, inline="always")
def union(parents, x, y):        
    rx = find_root(parents, x)
    ry = find_root(parents, y)

    parents[rx] = ry
    
@njit(cache=True, inline="always")
def reduce_roots(parents):
    new_parents = np.empty_like(parents, dtype=parents.dtype)
    for i in range(len(parents)):
        current_val = parents[i]
        if current_val == -1:
            # this basin hasn't been added yet. Note it and continue
            new_parents[i] = -1
            continue
        new_parents[i] = find_root(parents, current_val)
    return new_parents

@njit(fastmath=True, cache=True, inline="always")
def union_with_shift(parent, offset, a, b, si, sj, sk, has_shift):
    """
    Union two elements (a,b) with known periodic shift between them.
    Returns (root, cycle_vector or None).
    If they were already connected, the difference gives a cycle translation.
    """
    # get root index and offsets for each index
    ra, (ox, oy, oz) = find_root_with_shift(parent, offset, a)
    rb, (ox1, oy1, oz1) = find_root_with_shift(parent, offset, b)
    
    has_off = False
    if ox !=0 or oy !=0 or oz !=0:
        has_off = True
        
    has_off1 = False
    if ox1 !=0 or oy1 !=0 or oz1 !=0:
        has_off1 = True
    
    # calculate total offset
    if has_off and has_off1:
        ox = ox - ox1
        oy = oy - oy1
        oz = oz - oz1
    elif has_off1: # and not has_o
        ox = - ox1
        oy = - oy1
        oz = - oz1
    # if has_o1 and not has_o, we would update ox to be ox anyways
        
    
    # calculate the total offset vector at this root
    if has_shift:
        cx = ox - si
        cy = oy - sj
        cz = oz - sk
    else:
        cx = ox
        cy = oy
        cz = oz

    if ra == rb:
        # Already connected -> found a loop. Return the cycle translation
        return ra, cx, cy, cz
    else:
        # Merge smaller into larger tree for stability and accumulate offset
        parent[rb] = ra
        offset[rb] = (cx, cy, cz)
        return ra, 0, 0, 0
    
@njit(fastmath=True, inline="always", cache=True)
def flat_to_coords(idx, nx, ny, nz):
    i = idx // (ny * nz)
    j = (idx % (ny * nz)) // nz
    k = idx % nz
    return i, j, k


@njit(fastmath=True, inline="always", cache=True)
def coords_to_flat(i, j, k, ny, nz):
    return i * (ny * nz) + j * nz + k

@njit(inline="always", cache=True)
def wrap_point_w_shift(
    i,j,k,nx,ny,nz):

    has_shift = False
    si, sj, sk = (0,0,0)
    if i >= nx:
        i -= nx
        si = 1
        has_shift = True
    elif i < 0:
        i += nx
        si = -1
        has_shift = True
    if j >= ny:
        j -= ny
        sj = 1
        has_shift = True
    elif j < 0:
        j += ny
        sj = -1
        has_shift = True
    if k >= nz:
        k -= nz
        sk = 1
        has_shift = True
    elif k < 0:
        k += nz
        sk = -1
        has_shift = True
    return i, j, k, si, sj, sk, has_shift

@njit(cache=True)
def compress_cycles(
        parent, 
        offset,
        cycles, 
        cycle_roots,
        ):
    """
    Compress and deduplicate cycles by mapping to final roots and removing
    redundant or linearly dependent ones.
    """
    new_cycles = [(-1, -1, -1)]
    new_roots = [-1]
    # remove numba typing placeholders
    new_cycles = new_cycles[1:]
    new_roots = new_roots[1:]
    
    merged_cycles = [[(-1,-1,-1)]]
    merged_roots = [-1]
    merged_cycles = merged_cycles[1:]
    merged_roots = merged_roots[1:]

    # Step 1: map each cycle to its final root
    for i in range(len(cycles)):  # skip placeholder
        # get the root and cycle
        root = cycle_roots[i]
        cycle = cycles[i]

        final_root, _ = find_root_with_shift(parent, offset, root)
        
        # Try to find an existing group for this root
        found = False
        for merged_root, merged_cycle_list in zip(merged_roots, merged_cycles):
            if merged_root == final_root:
                found = True
                break
        # if no root was found, create a new group
        if not found:
            merged_cycles.append([cycle])
            merged_roots.append(final_root)
            # This cycle must be unique so we also add it to our compressed list
            new_cycles.append(cycle)
            new_roots.append(final_root)
            continue
        
        # otherwise, check if the cycle already exists in this group
        cycle_exists = False
        for new_cycle in merged_cycle_list:
            if cycle == new_cycle:
                cycle_exists = True
                break
        if not cycle_exists:
            merged_cycle_list.append(cycle)
            new_cycles.append(cycle)
            new_roots.append(final_root)

    return new_cycles, new_roots, merged_cycles, merged_roots


@njit(cache=True)
def get_component_dimensionality(
        solid,
        previous_solid,
        parent,
        offset,
        cycles,
        cycle_roots,
        neighbors,
        ):
    """
    Compute dimensionality (0D–3D) for each connected solid region
    under 26-neighbor periodic connectivity.
    Returns a dict: {label_id: dimensionality}
    """
    nx, ny, nz = solid.shape
    # N = nx * ny * nz
    
    # create list tracker for roots (numba friendly)
    if cycles is None:
        cycles = [(-1,-1,-1)] # placeholder for numba typing
        cycle_roots = [-1]
        cycles = cycles[1:]
        cycle_roots = cycle_roots[1:]
        
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                
                # NOTE: Doing a check like this has such a small time cost
                # that I didn't see a difference between doing a mock lookup/continue
                # for a 30^3 cube and 400^3 cube.
                if not solid[i,j,k] or previous_solid[i,j,k]:
                    continue
                
                idx = coords_to_flat(i, j, k, ny, nz)
                
                for di, dj, dk in neighbors:
                    # get wrapped neighbor and get any shift across a periodic
                    # boundary
                    ni, nj, nk, si, sj, sk, has_shift = wrap_point_w_shift(i+di, j+dj, k+dk, nx, ny, nz)
        
                    if not solid[ni, nj, nk]:
                        continue
        
                    # get neighbors flattened index
                    neigh_idx = coords_to_flat(ni, nj, nk, ny, nz)

                    # accumulate offset/shift and check for cycle
                    root, ci, cj, ck = union_with_shift(parent, offset, idx, neigh_idx, si, sj, sk, has_shift)
                    if ci != 0 or cj != 0 or ck != 0:
                        cycle_roots.append(root)
                        cycles.append((ci,cj,ck))

        
    # Reduce to unique cycles and group per root
    cycles, cycle_roots, merged_cycles, merged_cycle_roots = compress_cycles(
        parent=parent,
        offset=offset,
        cycles=cycles, 
        cycle_roots=cycle_roots,
        )

    # Compute dimensionality from cycle vectors
    dims = []
    for root, cyc_list in zip(merged_cycle_roots, merged_cycles):
        if len(cyc_list) == 0:
            dims.append(0) # finite cluster, no wrapping
            continue
        M = np.array(cyc_list, dtype=np.float32)
        rank = np.linalg.matrix_rank(M)
        dims.append(rank)

    return merged_cycle_roots, dims, cycles, cycle_roots

@njit(cache=True, inline='always')
def get_dimensionality(
    parent,
    offset,
    roots,
    dims,
    feature_point,
    nx, ny, nz,
        ):
    x,y,z = feature_point
    idx = coords_to_flat(x, y, z, ny, nz)
    # get root
    root_idx, _ = find_root_with_shift(parent, offset, idx)
    # check root dimensionality
    final_dim = 0
    for root, dim in zip(roots, dims):
        if root == root_idx:
            final_dim = dim
            break
    return final_dim

@njit(cache=True)
def find_bifurcations(
    connection_pairs,
    connection_elfs,
    basin_maxima_grid,
    basin_maxima_ref_values,
    data,
    neighbor_transforms,
        ):
    """
    Finds the ELF values at which changes in features or feature dimensionalities
    occur
    """
    nx, ny, nz = data.shape
    N = nx * ny * nz
    # num_basins = len(basin_maxima_grid)
    # get all possible elf values and flip to move from high to low
    possible_elf_values = np.flip(np.unique(connection_elfs))
    
    ###########################################################################
    # Dimensionality Setup
    ###########################################################################
    new_solid = np.zeros((nx,ny,nz), dtype=np.bool_)
    point_connections = np.arange(N, dtype=np.uint32)
    point_offsets = np.zeros((N, 3), dtype=np.int8) # can this be int8?
    cycles = None
    cycle_roots=None

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
    feature_groups = feature_groups[1:]
    feature_ids = feature_ids[1:]
    # do the same for dimensionalities of the features
    new_dimensionalities = [-1]
    new_dimensionalities = new_dimensionalities[1:]
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
            lower_root = find_root(basin_connections, i)
            upper_root = find_root(basin_connections, j)
            
            # if we have the same root, we dont have a new connection
            if lower_root == upper_root:
                continue
            # otherwise we note the new connection
            union(basin_connections, i, j)

        # reduce our connections
        basin_connections = reduce_roots(basin_connections)

        # copy our previous groups
        previous_groups = [i for i in feature_groups]
        # create lists to store features basin groups
        feature_groups = []
        feature_points = []
        num_features = 0
        

        # Now loop over our basin connections and get our groups
        for basin_idx, basin_group in enumerate(basin_connections):
            # skip if this basin isn't assigned
            if basin_group == -1:
                continue
            
            # check if this group exists yet
            if feature_group_indices[basin_group] == -1:
                # create a new feature
                feature_groups.append([basin_idx])
                feature_group_indices[basin_group] = num_features
                num_features += 1
                # add a point sitting in this feature
                feature_points.append(basin_maxima_grid[basin_idx])
            
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
        # track if we have at least one reducible group
        all_irreducible = False
        for group in feature_groups:
            group_len = len(group)
            if group_len > 1:
                all_irreducible = False
            # check to see if this group exists in the previous groups
            group_found = False
            for pgroup, pgroup_idx in zip(previous_groups, previous_feature_ids):
                if len(pgroup) != group_len:
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
        # Update Dimensionalities
        #######################################################################
        # copy previous dimensionalities
        old_dimensionalities = [i for i in new_dimensionalities]
        new_dimensionalities = []
        if not all_irreducible:
            previous_solid = new_solid
            new_solid = data >= elf_value
            # We also want to calculate the dimensionality of each feature, regardless
            # of if we have a new group or not.
            # Flood fill to the current value

            roots, dims, cycles, cycle_roots = get_component_dimensionality(
                new_solid,
                previous_solid,
                parent=point_connections,
                offset=point_offsets,
                cycles=cycles,
                cycle_roots=cycle_roots,
                neighbors=neighbor_transforms,
                )

            for feature_point in feature_points:
                new_dim = get_dimensionality(
                    parent=point_connections,
                    offset=point_offsets,
                    roots=roots,
                    dims=dims,
                    feature_point=feature_point,
                    nx=nx,
                    ny=ny,
                    nz=nz
                        )
                new_dimensionalities.append(new_dim)
        
        
        # if this is a new set of basin groups, we append no matter what
        if not same_groups:
            bifurcation_values.append(elf_value)
            bifurcation_features.append(feature_groups)
            bifurcation_feature_indices.append(feature_ids)
            bifurcation_dimensionalities.append(new_dimensionalities)
        # otherwise, we only append if we found new dimensionalities
        else:
            all_same = True
            for i, j in zip(old_dimensionalities, new_dimensionalities):
                if i != j:
                    all_same = False
                    break
            if not all_same:
                bifurcation_values.append(elf_value)
                bifurcation_features.append(feature_groups)
                bifurcation_feature_indices.append(feature_ids)
                bifurcation_dimensionalities.append(new_dimensionalities)
        
        # update our previous elf value
        previous_elf_value = elf_value
        
    # add a final feature that appears at the lowest possible value and contains
    # all basins
    bifurcation_values.append(data.min())
    bifurcation_features.append(bifurcation_features[-1])
    bifurcation_feature_indices.append(bifurcation_feature_indices[-1])
    bifurcation_dimensionalities.append(bifurcation_dimensionalities[-1])

    
    #######################################################################
    # Organize Features
    #######################################################################
    # reverse values to go from low to high
    bifurcation_values.reverse()
    bifurcation_features.reverse()
    bifurcation_feature_indices.reverse()
    bifurcation_dimensionalities.reverse()
    
    # Create arrays to track features
    feature_basins = [[] for i in range(num_features)]
    feature_min_elfs = np.empty(range(num_features), dtype=np.float64)
    feature_max_elfs = np.empty(range(num_features), dtype=np.float64)
    feature_dims = np.empty(range(num_features), dtype=np.int64)
    feature_parents = np.empty(range(num_features), dtype=np.int64)
    
    # Now we loop over our elf values.
    # NOTE: the basins at each value are the ones that exist at or below that
    # value. Therefore, the nodes that appear right above that value are those
    # in the next index of the list
    feat_count = 0
    for bif_idx, elf_value in enumerate(bifurcation_values[:-1]):
        # get the features that exist exactly at this value
        old_feature_indices = bifurcation_feature_indices[bif_idx]
        old_dimensions = bifurcation_dimensionalities[bif_idx]
        # get the features that appear right above this value
        new_features = bifurcation_features[bif_idx+1]
        new_feature_indices = bifurcation_feature_indices[bif_idx+1]
        new_dimensions = bifurcation_dimensionalities[bif_idx+1]
        # Now we loop over the new features and add any new ones to our graph
        for feat_idx, feat_basins, feat_dim in zip(
                new_feature_indices,
                new_features,
                new_dimensions
                ):
            # check if this feature exists in the previous set of indices
            new_node = True
            for prev_idx, prev_dim in zip(old_feature_indices, old_dimensions):
                if prev_idx == feat_idx and prev_dim == feat_dim:
                    # This feature existed previously
                    new_node = False
                    break
            if not new_node:
                continue
            
            # if we're still here, this is a new feature and we record its attributes
            feature_basins[feat_count] = feat_basins
            feature_min_elfs[feat_count] = elf_value
            feature_dims[feat_count] = feat_dim
            
            # find the parent of this feature
            possible_basins = feature_basins[:feat_count]
            possible_basins.reverse()

            for idx, (parent_basins) in enumerate(possible_basins):
                if np.all(np.isin(feat_basins, parent_basins)):
                    parent_idx = feat_count - idx - 1
                    break
            # add the parent connection
            feature_parents[feat_count] = parent_idx
            
            # note this parent disappears at this value
            feature_max_elfs[parent_idx] = elf_value
            
            # if this feature is irreducible, add its max value
            if len(feat_basins) == 1:
                feature_max_elfs[feat_count] = basin_maxima_ref_values[feat_basins[0]]
            
            # note we found a new feature
            feat_count += 1
    
            
    return (
        feature_basins,
        feature_min_elfs,
        feature_max_elfs,
        feature_dims,
        feature_parents,
        )
