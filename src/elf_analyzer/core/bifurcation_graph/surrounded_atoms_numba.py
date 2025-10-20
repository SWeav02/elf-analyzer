# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange

from baderkit.core.methods.shared_numba import wrap_point

from elf_analyzer.core.bifurcation_graph.infinite_feature_numba import (
    coords_to_flat,
    find_root_no_compression,
    get_connected_features,
    union,
    find_root
    )

@njit(cache=True)
def get_connected_voids(
        solid,
        previous_solid,
        basin_labels,
        basin_feature_map,
        num_features,
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

    # get the connected points for this solid
    (
     root_mask, 
     parent, 
     offset_x, 
     offset_y, 
     offset_z, 
     roots, 
     cycles, 
     dimensionalities
     ) = get_connected_features(
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
            )
    
    # create a mask to indicate the number of connections between features and
    # and voids
    connection_counts = np.zeros((len(roots), num_features), dtype=np.uint32)
    # loop over indices and count connections
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                
                # skip anything not in the solid
                if not solid[i,j,k]:
                    continue
                
                root_idx = -1 # we don't calculate these unless we have to
                # loop over neighbors
                for di, dj, dk in neighbors:
                    # get wrapped neighbor
                    ni, nj, nk = wrap_point(i+di, j+dj, k+dk, nx, ny, nz)
                    # skip anything in the solid
                    if solid[ni, nj, nk]:
                        continue
                    
                    # otherwise, get the connection here
                    if root_idx == -1:
                        idx = coords_to_flat(i, j, k, ny_nz, nz)
                        root = find_root_no_compression(parent, idx)
                        root_idx = np.searchsorted(roots, root)
                        
                    neigh_basin = basin_labels[ni, nj, nk]
                    feature = basin_feature_map[neigh_basin]
                    
                    connection_counts[root_idx, feature] += 1

    return (
        root_mask, 
        parent, 
        offset_x, 
        offset_y, 
        offset_z, 
        roots, 
        cycles, 
        dimensionalities,
        connection_counts,
        )

@njit(cache=True, inline="always")
def union_w_roots(parents, x, y, root_mask):        
    rx = find_root(parents, x)
    ry = find_root(parents, y)

    parents[rx] = ry
    
    if root_mask[rx]:
        root_mask[rx] = False
    if not root_mask[ry]:
        root_mask[ry] = True

@njit(cache=True)
def trans_to_idx(i, j, k, size):
    val_range = size*2+1
    return (i+size)*(val_range**2) + (j+size)*val_range + (k+size)

@njit(cache=True, inline="always")
def get_box_connections(size):
    shifts = []
    lower_connection = []
    upper_connection = []
    for i in range(-size, size+1):
        for j in range(-size, size+1):
            for k in range(-size, size+1):
                shifts.append((i,j,k))
                idx = trans_to_idx(i,j,k,size)
                for ni in range(-1, 2):
                    for nj in range(-1, 2):
                        for nk in range(-1, 2):
                            # skip center
                            if ni ==0 and nj ==0 and nk==0:
                                continue
                            # get shifted point
                            si = i+ni
                            sj = j+nj
                            sk = k+nk
                            # skip values outside range
                            if abs(si) > size or abs(sj) > size or abs(sk) > size:
                                continue
                            # get index
                            neigh_idx = trans_to_idx(si, sj, sk, size)
                            # skip previous indices to avoid repeat connections
                            if neigh_idx < idx:
                                continue
                            lower_connection.append(idx)
                            upper_connection.append(neigh_idx)
    
    lower_connection = np.array(lower_connection, dtype=np.uint16)
    upper_connection = np.array(upper_connection, dtype=np.uint16)
    connections = np.column_stack((lower_connection, upper_connection))
    return np.array(shifts, dtype=np.int8), connections

@njit(cache=True, inline="always")
def check_bifurcation(
    i,j,k,
    value,
    data,
    shifts,
    shift_connections,
    greater,
        ):
    nx, ny, nz = data.shape
    
    # create trackers for neighbors
    value_mask = np.zeros(125, dtype=np.uint8)
    connections = np.arange(125, dtype=np.uint8)
    root_mask = np.zeros(125, dtype=np.bool_)

    # mark mask
    for shift_idx, (si, sj, sk) in enumerate(shifts):
        ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
        neigh_value = data[ni,nj,nk]
        if (greater and neigh_value > value) or (not greater and neigh_value < value):
            value_mask[shift_idx] = 2
            root_mask[shift_idx] = True
            
        elif neigh_value == value:
            value_mask[shift_idx] = 1
    
    include_indices = []
    # iterate over connections and make unions
    for connection_idx, (shift1, shift2) in enumerate(shift_connections):
        connection_type = min(value_mask[shift1], value_mask[shift2])
        if connection_type == 0:
            continue
        elif connection_type == 2:
            union_w_roots(connections, shift1, shift2, root_mask)
        elif connection_type == 1:
            include_indices.append(connection_idx)
    
    n_exclude_groups = np.count_nonzero(root_mask)
    
    # make unions for exact vals
    for connection_idx in include_indices:
        shift1, shift2 = shift_connections[connection_idx]
        union_w_roots(connections, shift1, shift2, root_mask)
    n_include_groups = np.count_nonzero(root_mask)
    
    
    is_bif = n_exclude_groups != n_include_groups
    
    return is_bif

@njit(parallel=True, cache=True)
def find_potential_bifs(
    data,
    edge_mask,
    neighbor_transforms,
    greater=False
        ):
    nx, ny, nz = data.shape
        
    bif_mask = np.zeros_like(data, dtype=np.bool_)
    
    # We want to find bifurcations that make holes in our features. Imagine creating
    # an isosurface and constructing solids from the values above and below it.
    # A hole occurs in a solid where two parts of the reverse solid connect
    # for the first time.
    
    # For speed, we map out possible connections for each neighbor. We get
    # neighbor connections for first and second neighbors (3x3x3, 5x5x5)
    trans, trans_connections = get_box_connections(2)

    # loop over each edge point and find bifurcation voxels
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                
                # skip anything not on the edge
                if not edge_mask[i,j,k]:
                    continue
                
                value = data[i,j,k]
                # iterate over first neighbors and filter out points that don't
                # have opposite neighbors below this value
                candidate = False
                for si, sj, sk in neighbor_transforms[:13]:
                    ui, uj, uk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
                    value_above = data[ui,uj,uk]
                    if greater and value_above < value:
                        continue
                    elif not greater and value_above > value:
                        continue
                    bi, bj, bk = wrap_point(i-si, j-sj, k-sk, nx, ny, nz)
                    value_below = data[bi,bj,bk]
                    if (greater and value_below < value) or (not greater and value_below > value):
                        continue
                    candidate = True
                    break
                    
                if not candidate:
                    continue
                # check if point is a potential bifurcation
                is_bif = check_bifurcation(
                    i, j, k, 
                    value,
                    data, 
                    trans, 
                    trans_connections,
                    greater
                    )

                if is_bif:
                    bif_mask[i,j,k] = True
                
                
    return bif_mask

# @njit(parallel=True)
# def find_hole_bifs(
#     data,
#     edge_mask,
#         ):
#     nx, ny, nz = data.shape
        
#     hole_mask = np.zeros_like(data, dtype=np.bool_)
    
#     # We want to find bifurcations that make holes in our features. Imagine creating
#     # an isosurface and constructing solids from the values below it. A hole
#     # occurs at voxels where two solids connect at that voxels value. Locally,
#     # we can mask out values less than the voxel. If there are two distinct
#     # regions, they would connect at this voxels value, indicating it is a bifurcation
#     # point.
    
#     # For speed, we map out possible connections for each neighbor. We get
#     # neighbor connections for first and second neighbors (3x3x3, 5x5x5)
#     first_neighs, first_neigh_connections = get_box_connections(1)
#     second_neighs, second_neigh_connections = get_box_connections(2)
#     neighbor_indices = []
#     neighbor_neighs = []
#     for si, sj, sk in neighbor_transforms:
#         current_idx = trans_to_idx(si, sj, sk)
#         neigh_neighs = []
#         for neigh_idx, (sii, sjj, skk) in enumerate(neighbor_transforms):
#             iii = si+sii
#             jjj = sj+sjj
#             kkk = sk+skk
#             if abs(iii) == 2 or abs(jjj) == 2 or abs(kkk) == 2:
#                 continue
#             neigh_idx = trans_to_idx(iii, jjj, kkk)
#             if neigh_idx == 13 or neigh_idx < current_idx:
#                 # don't add the central point or previous neighbors
#                 continue
#             neigh_neighs.append(neigh_idx)
#         neighbor_indices.append(current_idx)
#         neighbor_neighs.append(np.array(neigh_neighs, dtype=np.uint8))
#     breakpoint()
#     # loop over each edge point and find bifurcation voxels
#     for i in prange(nx):
#         for j in range(ny):
#             for k in range(nz):
                
#                 # skip anything not on the edge
#                 if not edge_mask[i,j,k]:
#                     continue
                
#                 # get the value at this point
#                 value = data[i,j,k]
                
#                 # now we find the neighbors with values below this point and group
#                 # them if they border one another. If there are two or more groups,
#                 # this point is the only thing separating them, indicating a
#                 # bifurcation point
#                 neighbor_mask = np.zeros(27, dtype=np.bool_)
#                 for neigh_idx, (si, sj, sk) in zip(neighbor_indices, neighbor_transforms):
#                     # get wrapped neighbor
#                     ni, nj, nk = wrap_point(i+si, j+sj, k+sk, nx, ny, nz)
#                     # set mask
#                     if data[ni, nj, nk] <= value:
#                         neighbor_mask[neigh_idx] = True
#                 # now we find groups
#                 neigh_groups = np.arange(len(neighbor_mask), dtype=np.uint8)
#                 for neigh_idx, neigh_neighs in zip(neighbor_indices, neighbor_neighs):
#                     if not neighbor_mask[neigh_idx]:
#                         continue
#                     for neigh_neigh in neigh_neighs:
#                         if neighbor_mask[neigh_neigh]:
#                             union(neigh_groups, neigh_idx, neigh_neigh)
                
#                 # loop over neighbors and check if the opposing neighbor is part
#                 # of a different group
#                 for neigh_idx in neighbor_indices[:13]:
#                     if not neighbor_mask[neigh_idx]:
#                         continue
#                     rev_idx = 27-neigh_idx-1
#                     if not neighbor_mask[rev_idx]:
#                         continue
#                     root = find_root(neigh_groups, neigh_idx)
#                     rev_root = find_root(neigh_groups, rev_idx)
#                     # if opposites belong to different groups, this is a bifurcation
#                     if root != rev_root:
#                         hole_mask[i,j,k] = True
#                     break
                
#     hole_indices = np.argwhere(hole_mask)
#     hole_values = []
#     for i,j,k in hole_indices:
#         hole_values.append(data[i,j,k])
                
#     return np.array(hole_values, dtype=np.float64)

# @njit(cache=True)
def find_atom_features(
        atom_coords,
        parent,
        solid,
        basin_labels,
        basin_feature_map,
        all_features,
        connection_counts,
        void_roots,
        feature_dims,
        void_dims,
        surrounding_features,
        ):
    nx, ny, nz = basin_labels.shape
    ny_nz = ny*nz
    i,j,k = atom_coords

    # get the void or feature the atom sits in
    if solid[i,j,k]:
        atom_flat_idx = coords_to_flat(i,j,k,ny_nz,nz)
        root = find_root_no_compression(parent, atom_flat_idx)
        current_group = np.searchsorted(void_roots, root)
        is_void = True
    else:
        basin_idx = basin_labels[i,j,k]
        current_group = basin_feature_map[basin_idx]
        is_void = False

    # iteratively search for a group (feature or void) of the opposite type
    # that surrounds the current group. For a group to be "surrounded" it must
    # follow a set of rules:
        # 1. The group itself must be finite (0D).
        # 2. The surrounding group must have the most connecting points of any
        # possible group
    # NOTE: Any finite feature/void must by definition be surrounded by a group
    # of the opposite type or it wouldn't be finite. This group will also always
    # have more surface area in contact with the feature/void than any other
    # group. In other words, the most connecting points.
    
    while True:
        if is_void:
            # get dimensionality
            void_dim = void_dims[current_group]
            # if the void is infinite, it can't be surrounded and we break
            if void_dim > 0:
                break
            # otherwise, we find the feature with the most connections to this
            # void
            current_group = np.argmax(connection_counts[current_group])
            is_void = False
        else:
            # add this feature to our list
            feature_idx = all_features[current_group]
            surrounding_features.append(feature_idx)
            # get its dimensionality
            feature_dim = feature_dims[feature_idx]
            # if the feature is infinite, it can't be surrounded and we break
            if feature_dim > 0:
                break
            # otherwise, we find the void with the most connections to this
            # feature
            current_group = np.argmax(connection_counts[:,current_group])
            is_void = True
    return surrounding_features
    
# @njit(parallel=True,cache=True)
def find_all_atom_features(
        atom_grid_coords,
        parent,
        solid,
        basin_labels,
        basin_feature_map,
        all_features,
        connection_counts,
        void_roots,
        feature_dims,
        void_dims,
        ):
    
    # create lists to store which features surround each atom
    all_surrounding_features = []
    for i in range(len(atom_grid_coords)):
        feature_list = [-1]
        feature_list = feature_list[1:] # -1 is placeholder for numba typing
        all_surrounding_features.append(feature_list) 
        
    # TODO: Create tracker for if atom sits inside feature vs. surrounded by it?
    # Should be easy, but I'm not sure I would use it yet
    for atom_idx in prange(len(atom_grid_coords)):
        atom_coords = atom_grid_coords[atom_idx]
        surrounding_features = all_surrounding_features[atom_idx]
        
        find_atom_features(
            atom_coords,
            parent,
            solid,
            basin_labels,
            basin_feature_map,
            all_features,
            connection_counts,
            void_roots,
            feature_dims,
            void_dims,
            surrounding_features,
            )

    return all_surrounding_features

# @njit(cache=True)
def get_features_surrounding_atoms(
        feature_basins,
        feature_min_values,
        feature_max_values,
        feature_dims,
        atom_grid_coords,
        neighbor_transforms,
        basin_labels,
        data,
        num_basins,
        ):
    nx, ny, nz = data.shape
    N = nx * ny * nz
    
    # get unique elf values to iterate over
    unique_elfs = np.unique(feature_min_values)
    
    # create map from basin labels to feature labels
    basin_feature_map = np.empty(num_basins, dtype=np.uint32)
    
    # create an initial empty solid
    new_solid = np.zeros((nx,ny,nz), dtype=np.bool_) # initial empty solid
    root_mask = np.zeros(N, dtype=np.bool_) # nothing is a root yet
    roots = np.empty(0, dtype=np.int64) # empty 1D array as nothing is a root yet
    cycles = np.empty((0,125), dtype=np.bool_) # empty as we have no cycles yet
    parent = np.arange(N, dtype=np.uint32) # All voxels point to themselves
    offset_x = np.zeros(N, dtype=np.int8) # no offsets to start
    offset_y = np.zeros(N, dtype=np.int8)
    offset_z = np.zeros(N, dtype=np.int8)
    size = np.ones(N, dtype=np.uint16) # no size to start
    
    # create lists to track which features surround atoms
    feature_atoms = []
    for i in range(len(feature_basins)):
        feature_list = [-1]
        feature_list = feature_list[1:]
        feature_atoms.append(feature_list)
    
    for elf_value in unique_elfs:
        # NOTE: I don't think we need to set the basin_feature_map to a default
        # value because it shouldn't be possible for a void to touch a feature
        # other than one in this list
        
        new_features = []
        all_features = []
        num_features = 0
        # get the features that exist at this elf value and construct basin map
        for feature_idx, (min_value, max_value) in enumerate(zip(feature_min_values, feature_max_values)):
            
            if min_value <= elf_value and max_value > elf_value:
                # This feature currently has some volume in the isosolid
                # create map for each basin in this feature
                # add it to our list
                all_features.append(feature_idx)
                # map each basin in this feature back to the feature
                for basin_idx in feature_basins[feature_idx]:
                    basin_feature_map[basin_idx] = num_features
                # note we found a new feature
                num_features += 1
                
                # if the feature just appeared, add it to our list
                if min_value == elf_value:
                    # this feature just appeared.
                    new_features.append(feature_idx)
                
            else:
                continue
        
        # get void/feature connection information at this elf value
        previous_solid = new_solid
        new_solid = data <= elf_value
        # TODO: I can probably improve speed further by freezing features/basins
        # that have been found to not surround anything, decreasing the number
        # of voxels that need to be checked for unions
        
        (
        root_mask, 
        parent, 
        offset_x, 
        offset_y, 
        offset_z, 
        roots, 
        cycles, 
        dimensionalities,
        connection_counts,
            ) = get_connected_voids(
                solid=new_solid,
                previous_solid=previous_solid,
                basin_labels=basin_labels,
                basin_feature_map=basin_feature_map,
                num_features=num_features,
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
                
        # Get which features surround each atom
        all_surrounding_features = find_all_atom_features(
                atom_grid_coords=atom_grid_coords,
                parent=parent,
                solid=new_solid,
                basin_labels=basin_labels,
                basin_feature_map=basin_feature_map,
                all_features=all_features,
                connection_counts=connection_counts,
                void_roots=roots,
                feature_dims=feature_dims,
                void_dims=dimensionalities,
                )
        
        # track if no atoms are surrounded
        no_atoms_surrounded = True
        # Add atoms to feature lists
        for atom_idx, feature_list in enumerate(all_surrounding_features):
            if len(feature_list) == 0:
                continue
            no_atoms_surrounded = False
            for feature_idx in feature_list: # skip placeholder value
                if feature_idx in new_features:
                    feature_atoms[feature_idx].append(atom_idx)
        # if no atoms are surrounded, they never will be again and we are done
        # here
        if no_atoms_surrounded:
            break
        
    return feature_atoms





