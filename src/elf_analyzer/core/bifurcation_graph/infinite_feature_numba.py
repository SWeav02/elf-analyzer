# -*- coding: utf-8 -*-

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

from baderkit.core.methods.shared_numba import wrap_point

@njit(cache=True, parallel=True)
def find_connections(
        labeled_array: NDArray[np.int64],
        data: NDArray[np.float64],
        edge_mask,
        basin_num: np.int64,
        neighbor_transforms: NDArray[np.int64],
        ):
    """
    Finds the values at which basins connect to on another locally
    """
    nx, ny, nz = labeled_array.shape

    # create arrays to track connection values
    conn_values = np.empty_like(data, dtype=np.float64)
    conn_neighs = np.empty_like(data, dtype=np.uint16)
    
    # loop over edges and calculate their potential bifurcation values
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                # skip points that aren't on the edge
                if not edge_mask[i,j,k]:
                    continue
                
                # get this points elf value and basin label
                label = labeled_array[i,j,k]
                if label == -1:
                    # shouldn't happen if bader is working properly
                    continue
                
                value = data[i,j,k]
                
                # we want to find the highest value that connects this point
                # to one of its neighbors (in another basin). This is the first
                # value at which a new connection would occur due to this point.
                # We also want to find the other point partaking in this connection.
                # if there are multiple possibilities, we want the one with the
                # lowest value.
                best_lower_conn_val = -1e12 # extremely low value
                best_upper_conn_val = 1e12
                conn_neigh = -1
                # loop over neighbors
                for si, sj, sk in neighbor_transforms:
                    # wrap points
                    ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                    # skip non-edges
                    if not edge_mask[ii,jj,kk]:
                        continue
                    
                    neigh_label = labeled_array[ii,jj,kk]
                    # skip points in the same basin
                    if neigh_label == label or neigh_label == -1:
                        continue
                    
                    neigh_value = data[ii,jj,kk]
                    
                    # get the value at which the points would connect if scanning
                    # up the ELF
                    lower_conn_value = min(value, neigh_value)
                    
                    # skip if this is lower than the current best
                    if lower_conn_value < best_lower_conn_val:
                        continue
                    
                    upper_conn_value = max(value, neigh_value)
                    # if this connection is better than previous ones, update
                    if lower_conn_value > best_lower_conn_val:
                        best_lower_conn_val = lower_conn_value
                        best_upper_conn_val = upper_conn_value
                        conn_neigh = neigh_label
                    elif lower_conn_value == best_lower_conn_val and upper_conn_value < best_upper_conn_val:
                        best_upper_conn_val = upper_conn_value
                        conn_neigh = neigh_label

                # note the highest connected basin still below this point
                conn_values[i,j,k] = best_lower_conn_val
                conn_neighs[i,j,k] = conn_neigh
    
    # Now we have a record of the highest point at which each point connects to
    # another basin. A point is a bifurcation if none of the adjacent neighbors in the
    # same basin connect to the same neighboring basin at a higher value and if
    # none of the adjacent neighbors in the neighboring basin have a higher value
    lower_points = []
    upper_points = []
    values = []
    
    # TODO: This could actually be done in parallel and then the results collected
    # into lists in an additional loop. I'm just not sure how much faster that
    # would be and it would increase memory usage
    
    # loop over each edge voxel
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # skip points that aren't on the edge
                if not edge_mask[i,j,k]:
                    continue
                
                label = labeled_array[i,j,k]
                if label == -1:
                    # shouldn't happen if bader is working properly
                    continue
                
                # get the label of the connected basin and the value they connect at
                conn_neigh = conn_neighs[i,j,k]
                conn_value = conn_values[i,j,k]
                
                is_bif = True
                # loop over neighbors
                for si, sj, sk in neighbor_transforms:
                    # wrap points
                    ii, jj, kk = wrap_point(i + si, j + sj, k + sk, nx, ny, nz)
                    # skip non-edges
                    if not edge_mask[ii,jj,kk]:
                        continue
                    
                    neigh_label = labeled_array[ii, jj, kk]
                    # skip neighbors without labels
                    if neigh_label == -1:
                        continue
                    
                    # skip neighs with lower connection values
                    neigh_conn_value = conn_values[ii,jj,kk]
                    if neigh_conn_value <= conn_value:
                        continue
                    
                    # get this neighbors connected basin
                    neigh_conn_neigh = conn_neighs[ii,jj,kk]
                    
                    # if the neighbor has the same basin connection pair, this
                    # is not a bifurcation
                    if (
                        (neigh_label == label and neigh_conn_neigh == conn_neigh)
                    or (neigh_label == conn_neigh and neigh_conn_neigh == label)
                    ):
                        is_bif = False
                        break

                # if this is a bifurcation, add it to our list
                if is_bif:
                    lower_label = min(label, conn_neigh)
                    upper_label = max(label, conn_neigh)
                    lower_points.append(lower_label)
                    upper_points.append(upper_label)
                    values.append(conn_value)
                    
    return np.array(lower_points, dtype=np.int64), np.array(upper_points, dtype=np.int64), np.array(values, dtype=np.float64)

@njit(cache=True)
def find_root(parent, x):
    """Find root with partial path compression"""
    while x != parent[x]:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

@njit(cache=True)
def find_root_no_compression(parent, x):
    while x != parent[x]:
        x = parent[x]
    return x
    
@njit(cache=True, inline="always")
def find_root_with_shift(parent, offset_x, offset_y, offset_z, x):
    """
    Path-halving find-with-offsets.
    - parent: int32[:] parent pointers (root has parent[root] == root)
    - offset_*: int32[:] offsets such that pos(parent[i]) = pos(i) + offset(i)
    - x: int index
    Returns: root, cx, cy, cz (cumulative offset from x -> root)
    """
    # local aliasing to avoid repeated global lookups
    y = x

    # Path-halving loop: compress path by setting parent[y] = parent[parent[y]]
    # and updating offset[y] to remain consistent.
    # This reduces the path length quickly with fewer writes than full compression.
    while parent[y] != y and parent[parent[y]] != parent[y]:
        p = parent[y]
        # add p's offset into y so that y points to p's parent consistently
        offset_x[y] += offset_x[p]
        offset_y[y] += offset_y[p]
        offset_z[y] += offset_z[p]
        # set y to point to grandparent
        parent[y] = parent[p]
        # advance y (we short-circuited one level)
        y = parent[y]

    # Final climb to accumulate the cumulative offset; path is now short.
    cx = 0
    cy = 0
    cz = 0
    y = x
    while parent[y] != y:
        cx += offset_x[y]
        cy += offset_y[y]
        cz += offset_z[y]
        y = parent[y]

    return y, cx, cy, cz

@njit(cache=True, inline="always")
def find_root_with_shift_no_compress(parent, offset_x, offset_y, offset_z, x):
    cx = 0
    cy = 0
    cz = 0
    while parent[x] != x:
        cx += offset_x[x]
        cy += offset_y[x]
        cz += offset_z[x]
        x = parent[x]

    return x, cx, cy, cz

@njit(cache=True, inline="always")
def union(parents, x, y):        
    rx = find_root(parents, x)
    ry = find_root(parents, y)

    parents[rx] = ry
    
@njit(cache=True, inline="always")
def union_with_shift(root_mask, parent, offset_x, offset_y, offset_z, size, a, b, si, sj, sk):
    """
    Union a and b with periodic shift (si,sj,sk) between them where
    b is neighbor of a (so the geometric relation between a and b includes shift).
    Returns (root, cx, cy, cz). If a and b were already connected, returns cycle vector (cx,cy,cz).
    If merged, returns (new_root, 0,0,0).
    """
    ra, ox, oy, oz = find_root_with_shift(parent, offset_x, offset_y, offset_z, a)
    rb, ox1, oy1, oz1 = find_root_with_shift(parent, offset_x, offset_y, offset_z, b)
    
        
    if ra == rb:
        # no need to combine
        return
    
    cx = si + ox1 - ox
    cy = sj + oy1 - oy
    cz = sk + oz1 - oz


    # union-by-size: attach smaller under larger
    if size[ra] < size[rb]:
        # attach ra under rb. We must compute offset for ra => rb.
        # We currently have cx = pos(rb) - pos(ra) (by derivation above),
        # if we attach ra -> rb then off[ra] = cx (so pos(rb)=pos(ra)+off[ra])
        parent[ra] = rb
        offset_x[ra] = cx
        offset_y[ra] = cy
        offset_z[ra] = cz
        size[rb] += size[ra]
        if not root_mask[rb]:
            root_mask[rb] = True
        if root_mask[ra]:
            root_mask[ra] = False

    else:
        # attach rb under ra. Then we need off[rb] such that pos(ra) = pos(rb) + off[rb].
        # Since cx = pos(rb) - pos(ra), off[rb] must be -cx.
        parent[rb] = ra
        offset_x[rb] = -cx
        offset_y[rb] = -cy
        offset_z[rb] = -cz
        size[ra] += size[rb]
        if not root_mask[ra]:
            root_mask[ra] = True
        if root_mask[rb]:
            root_mask[rb] = False

    
@njit(cache=True, parallel=True, inline="always")
def compress_roots(parents):
    new_parents = np.empty_like(parents, dtype=parents.dtype)
    for i in prange(len(parents)):
        current_val = parents[i]
        if current_val == -1:
            # this basin hasn't been added yet. Note it and continue
            new_parents[i] = -1
            continue
        new_parents[i] = find_root(parents, current_val)
    return new_parents
    
@njit(fastmath=True, inline="always", cache=True)
def flat_to_coords(idx, ny_nz, nz):
    i = idx // (ny_nz)
    j = (idx % (ny_nz)) // nz
    k = idx % nz
    return i, j, k


@njit(fastmath=True, inline="always", cache=True)
def coords_to_flat(i, j, k, ny_nz, nz):
    return i * ny_nz + j * nz + k

@njit(inline="always", cache=True)
def wrap_point_w_shift(
    i,j,k,nx,ny,nz):

    si, sj, sk = (0,0,0)
    if i >= nx:
        i -= nx
        si = 1
    elif i < 0:
        i += nx
        si = -1
    if j >= ny:
        j -= ny
        sj = 1
    elif j < 0:
        j += ny
        sj = -1
    if k >= nz:
        k -= nz
        sk = 1
    elif k < 0:
        k += nz
        sk = -1
    return i, j, k, si, sj, sk

@njit(cache=True, inline="always")
def shift_to_index(cx, cy, cz):
    # each value can range from -2 to 2, essentially giving us a base 5 system
    index = (cx + 2) * 25 + (cy + 2) * 5 + (cz + 2)
    return index
    
@njit(cache=True, inline="always")
def index_to_shift(index):
    cx = index // 25 - 2
    cy = (index % 25) // 5 - 2
    cz = index % 5 - 2
    return cx, cy, cz

@njit(parallel=True, inline="always")
def find_cycles(
        solid, 
        previous_solid,
        old_cycles,
        parent, 
        offset_x, 
        offset_y, 
        offset_z, 
        old_roots, 
        root_mask,
        neighbors,
        ):
    nx, ny, nz = solid.shape
    ny_nz = ny * nz
    
    # get the current roots
    new_roots = np.nonzero(root_mask)[0]
    n_roots = len(new_roots)
    
    # create a new array for cycles in the current mask
    new_cycles = np.zeros((n_roots, 125), dtype=np.bool_)
    
    # add cycles from the previous round
    for old_root, old_cycle in zip(old_roots, old_cycles):
        # get the new root
        new_root = find_root_no_compression(parent, old_root)
        new_root_idx = np.searchsorted(new_roots, new_root)
        # update to include previous values
        new_cycles[new_root_idx] = old_cycle
        
    # now iterate over new points and find new cycles
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if not solid[i,j,k] or previous_solid[i,j,k]:
                    continue
                idx = coords_to_flat(i, j, k, ny_nz, nz)

                for di, dj, dk in neighbors:
                    ni, nj, nk, si, sj, sk = wrap_point_w_shift(i+di, j+dj, k+dk, nx, ny, nz)
                    if not solid[ni,nj,nk]:
                        continue
                    neigh_idx = coords_to_flat(ni, nj, nk, ny_nz, nz)

                    ra, ox, oy, oz = find_root_with_shift_no_compress(parent, offset_x, offset_y, offset_z, idx)
                    rb, ox1, oy1, oz1 = find_root_with_shift_no_compress(parent, offset_x, offset_y, offset_z, neigh_idx)

                    if ra != rb:
                        continue

                    cx = ox - ox1 - si
                    cy = oy - oy1 - sj
                    cz = oz - oz1 - sk

                    if cx == 0 and cy == 0 and cz == 0:
                        continue

                    root_idx = np.searchsorted(new_roots, ra)
                    cycle_idx = shift_to_index(cx, cy, cz)
                    if not new_cycles[root_idx, cycle_idx]:
                        new_cycles[root_idx, cycle_idx] = True


    return new_cycles, new_roots

@njit(parallel=True, cache=True, inline="always")
def get_root_dims(cycles):
    # create array to store dims
    dimensionalities = np.zeros(cycles.shape[0], dtype=np.int8)
    for root_idx in prange(cycles.shape[0]):
        cycle_shifts = cycles[root_idx]
        cycle_list = []
        for cycle_idx, is_cycle in enumerate(cycle_shifts):
            if not is_cycle:
                continue
            cx, cy, cz = index_to_shift(cycle_idx)
            cycle_list.append((cx, cy, cz))
        if len(cycle_list) == 0:
            dimensionalities[root_idx] = 0 # finite, no wrapping
            continue

        M = np.array(cycle_list, dtype=np.float32)
        rank = np.linalg.matrix_rank(M)
        dimensionalities[root_idx] = rank
    return dimensionalities
            

@njit(cache=True)
def get_connected_features(
        solid,
        previous_solid,
        root_mask,
        roots,
        cycles,
        parent,
        offset_x,
        offset_y,
        offset_z,
        size,
        neighbors,
        ):
    """
    Compute dimensionality (0D–3D) for each connected solid region
    under 26-neighbor periodic connectivity.
    Returns a dict: {label_id: dimensionality}
    """
    nx, ny, nz = solid.shape
    ny_nz = ny*nz

        
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                
                # NOTE: Doing a check like this has such a small time cost
                # that I didn't see a difference between doing a mock lookup/continue
                # for a 30^3 cube and 400^3 cube.
                if not solid[i,j,k] or previous_solid[i,j,k]:
                    continue
                
                idx = coords_to_flat(i, j, k, ny_nz, nz)
                
                found_neigh = False
                for di, dj, dk in neighbors:
                    # get wrapped neighbor and get any shift across a periodic
                    # boundary
                    ni, nj, nk, si, sj, sk = wrap_point_w_shift(i+di, j+dj, k+dk, nx, ny, nz)
        
                    if not solid[ni, nj, nk]:
                        continue
                    found_neigh = True
                    # get neighbors flattened index
                    neigh_idx = coords_to_flat(ni, nj, nk, ny_nz, nz)

                    # accumulate offset/shift and check for cycle
                    union_with_shift(root_mask, parent, offset_x, offset_y, offset_z, size, idx, neigh_idx, si, sj, sk)
                
                # if we have no neighbors, we are a root with ourself
                if not found_neigh:
                    root_mask[idx] = True
                        
    # Now we find the unique cycles that occur for each root. Each cycle can
    # only take the form (cx, cy, cz) where each value is in -2, -1, 0, 1, 2.
    # This is essentially a base 5 system and we can convert any cycle to one
    # of 125 possible values. Thus we need an array of shape len(roots)x125
    cycles, roots = find_cycles(
        solid=solid, 
        previous_solid=previous_solid,
        old_cycles=cycles,
        parent=parent, 
        offset_x=offset_x, 
        offset_y=offset_y, 
        offset_z=offset_z, 
        old_roots=roots, 
        root_mask=root_mask,
        neighbors=neighbors,
        )
    
    # Now get dimensionalities of each root
    dimensionalities = get_root_dims(cycles)
    
    return root_mask, parent, offset_x, offset_y, offset_z, roots, cycles, dimensionalities

@njit(cache=True, inline='always')
def get_feature_dimensionality(
    parent,
    offset_x,
    offset_y,
    offset_z,
    roots,
    dims,
    feature_point,
    ny_nz, nz,
        ):
    x,y,z = feature_point
    idx = coords_to_flat(x, y, z, ny_nz, nz)
    # get root
    root_idx, _, _, _ = find_root_with_shift(parent, offset_x, offset_y, offset_z, idx)
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
    ny_nz = ny*nz
    N = nx * ny * nz
    
    num_basins = len(basin_maxima_grid)

    # get all possible elf values and flip to move from high to low
    possible_values = np.flip(np.unique(connection_elfs))
    
    ###########################################################################
    # Dimensionality Setup
    ###########################################################################
    new_solid = np.zeros((nx,ny,nz), dtype=np.bool_) # initial empty solid
    root_mask = np.zeros(N, dtype=np.bool_) # nothing is a root yet
    roots = np.empty(0, dtype=np.int64) # empty 1D array as nothing is a root yet
    cycles = np.empty((0,125), dtype=np.bool_) # empty as we have no cycles yet
    parent = np.arange(N, dtype=np.uint32) # All voxels point to themselves
    offset_x = np.zeros(N, dtype=np.int8) # no offsets to start
    offset_y = np.zeros(N, dtype=np.int8)
    offset_z = np.zeros(N, dtype=np.int8)
    size = np.ones(N, dtype=np.uint16) # no size to start

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
    basin_connections = np.full(num_basins, -1, dtype=np.int64)
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
    previous_value = 1.0e12 # make unreasonably large
    for val_idx, value in enumerate(possible_values):
        #######################################################################
        # Find groups of connected basins
        #######################################################################
        # get the new connections that exist at this value
        connection_indices = np.where((connection_elfs>=value) & (connection_elfs < previous_value))
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
        basin_connections = compress_roots(basin_connections)

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
        # Update Dimensionalities
        #######################################################################
        # copy previous dimensionalities
        old_dimensionalities = [i for i in new_dimensionalities]
        new_dimensionalities = []

        previous_solid = new_solid
        new_solid = data >= value
        # calculate new dimensionalities
        root_mask, parent, offset_x, offset_y, offset_z, roots, cycles, dims = get_connected_features(
            solid=new_solid,
            previous_solid=previous_solid,
            root_mask=root_mask,
            roots=roots,
            cycles=cycles,
            parent=parent,
            offset_x=offset_x, 
            offset_y=offset_y, 
            offset_z=offset_z,
            size=size,
            neighbors=neighbor_transforms,
            )

        for idx, (feature_point, feature_id) in enumerate(zip(feature_points, feature_ids)):
            # get this features dimensionality
            new_dim = get_feature_dimensionality(
                parent=parent,
                offset_x=offset_x, 
                offset_y=offset_y, 
                offset_z=offset_z,
                roots=roots,
                dims=dims,
                feature_point=feature_point,
                ny_nz=ny_nz,
                nz=nz
                    )
            new_dimensionalities.append(new_dim)
            # check if this feature exists in the previous group and if so, check
            # if it has the same dimensionality
            for prev_idx, prev_dim in zip(previous_feature_ids, old_dimensionalities):
                if prev_idx == feature_id and prev_dim != new_dim:
                    # this is actually a new feature, so we update its id
                    feature_ids[idx] = unique_features
                    unique_features += 1
                    same_groups = False
                    break
        
        # if a new feature appeared, we append the current groups
        if not same_groups:
            bifurcation_values.append(value)
            bifurcation_features.append(feature_groups)
            bifurcation_feature_indices.append(feature_ids)
            bifurcation_dimensionalities.append(new_dimensionalities)
        
        # update our previous elf value
        previous_value = value
        
        # if we've found a single feature containing all basins that is 3D, we
        # can break as no further bifurcations will be found
        if len(feature_groups) == 1:
            if len(feature_groups[0]) == num_basins and new_dimensionalities[0] == 3:
                break
    
    #######################################################################
    # Organize Features
    #######################################################################
    # reverse values to go from low to high
    bifurcation_values.reverse()
    bifurcation_features.reverse()
    bifurcation_feature_indices.reverse()
    bifurcation_dimensionalities.reverse()
    
    # Create arrays to track features
    feature_basins = [[-1] for i in range(unique_features)]
    feature_min_values = np.empty(unique_features, dtype=np.float64)
    feature_max_values = np.empty(unique_features, dtype=np.float64)
    feature_dims = np.empty(unique_features, dtype=np.int64)
    feature_parents = np.empty(unique_features, dtype=np.int64)
    
    # Add our initial features where all possible basin connections exist. This
    # will be at our lowest value.
    # BUGFIX: There could be multiple atoms/molecules that
    # are separated by vacuum resulting in more than one root in our graph
    min_data = data.min()
    feat_count = 0
    for feat_basins, feat_dim in zip(bifurcation_features[0], bifurcation_dimensionalities[0]):
        feature_basins[feat_count] = feat_basins
        feature_min_values[feat_count] = min_data
        feature_dims[feat_count] = feat_dim
        feature_parents[feat_count] = -1 # No parent
        feat_count += 1

    # Now we loop over our elf values.
    # NOTE: the basins at each value are the ones that exist at or below that
    # value. Therefore, the nodes that appear right above that value are those
    # in the next index of the list
    for bif_idx, value in enumerate(bifurcation_values[:-1]):
        # get the features that exist exactly at this value
        old_feature_indices = bifurcation_feature_indices[bif_idx]
        # old_dimensions = bifurcation_dimensionalities[bif_idx]
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
            if feat_idx in old_feature_indices:
                continue
            
            # if we're still here, this is a new feature and we record its attributes
            feature_basins[feat_count] = feat_basins
            feature_min_values[feat_count] = value
            feature_dims[feat_count] = feat_dim
            
            # find the parent of this feature. We need to iterate backwards over
            # the previously found features and find the first one containing
            # the basins in the current feature
            possible_basins = feature_basins[:feat_count]
            possible_basins.reverse()

            for idx, (parent_basins) in enumerate(possible_basins):
                if np.all(np.isin(feat_basins, parent_basins)):
                    parent_idx = feat_count - idx - 1
                    break
            # add the parent connection
            feature_parents[feat_count] = parent_idx
            
            # note this parent disappears at this value
            feature_max_values[parent_idx] = value
            
            # if this feature is irreducible, add its max value
            if len(feat_basins) == 1:
                feature_max_values[feat_count] = basin_maxima_ref_values[feat_basins[0]]
            
            # note we found a new feature
            feat_count += 1
    
            
    return (
        feature_basins,
        feature_min_values,
        feature_max_values,
        feature_dims,
        feature_parents,
        )
