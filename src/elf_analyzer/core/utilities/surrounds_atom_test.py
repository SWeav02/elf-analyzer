# -*- coding: utf-8 -*-

import numpy as np
from numba import njit

from elf_analyzer.core.utilities.solid_dimensionality_test import (
    find_root_with_shift,
    union_with_shift,
    coords_to_flat,
    wrap_point_w_shift,
    compress_cycles
    )

@njit
def compress_contacts(
    parent,
    offset,
    connected_features,
    connected_voids,
        ):
    reduced_connected_voids = []
    reduced_connected_features = []
    connection_counts = []

    for base_root, feature in zip(connected_voids, connected_features):
        # get the true root
        root, _ = find_root_with_shift(parent, offset, base_root)
        # check if root/feature combo already in list
        found = False
        for connection_idx, (rroot, rfeature) in enumerate(zip(reduced_connected_voids, reduced_connected_features)):
            if rroot == root and rfeature == feature:
                found = True
                # add a count
                connection_counts[connection_idx] += 1
                break
        
        # if new combo, add entry to lists

        if not found:
            reduced_connected_voids.append(root)
            reduced_connected_features.append(feature)
            connection_counts.append(1)


    return reduced_connected_voids, reduced_connected_features, connection_counts

@njit
def get_void_dimensionality(
        solid,
        previous_solid,
        basin_labels,
        basin_feature_map,
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
        
    # create array tracker for the number of connections between each feature
    # and void
    connected_features = []
    connected_voids = []

        
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
                        # check what feature this is
                        basin_label = basin_labels[ni, nj, nk]
                        feature = basin_feature_map[basin_label]
                        connected_features.append(feature)
                        connected_voids.append(idx)
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

            
    # Get unique void/feature connections and counts
    connected_voids, connected_features, connection_counts = compress_contacts(
        parent, 
        offset, 
        connected_features, 
        connected_voids)


    return (
        merged_cycle_roots, 
        dims, 
        cycles, 
        cycle_roots, 
        connected_voids, 
        connected_features, 
        connection_counts,
        )

@njit
def find_atom_features(
        atom_coords,
        parent,
        offset,
        solid,
        basin_labels,
        basin_feature_map,
        feature_dims,
        void_roots,
        void_dims,
        connected_voids, 
        connected_features, 
        connection_counts,
        surrounding_features,
        ):
    nx, ny, nz = basin_labels.shape
    i,j,k = atom_coords

    # get the void or feature the atom sits in
    if solid[i,j,k]:
        atom_flat_idx = coords_to_flat(i,j,k,ny,nz)
        current_group, _ = find_root_with_shift(parent, offset, atom_flat_idx)
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
            for void_root, void_dim in zip(void_roots, void_dims):
                if void_root == current_group:
                    break
            # if the void is infinite, it can't be surrounded and we break
            if void_dim > 0:
                break
            # otherwise, we find the feature with the most connections to this
            # void
            best_feature = -1
            best_count = 0
            for void_root, feature_idx, count in zip(connected_voids, connected_features, connection_counts):
                if void_root != current_group:
                    continue
                if count > best_count:
                    best_feature = feature_idx
                    best_count = count
            # We've found our feature and can continue to our next iteration
            current_group = best_feature
            is_void = False
        else:
            # add this feature to our list
            surrounding_features.append(current_group)
            # get its dimensionality
            feature_dim = feature_dims[current_group]
            # if the feature is infinite, it can't be surrounded and we break
            if feature_dim > 0:
                break
            # otherwise, we find the void with the most connections to this
            # feature
            best_void = -1
            best_count = 0
            for void_root, feature_idx, count in zip(connected_voids, connected_features, connection_counts):
                if feature_idx != current_group:
                    continue
                if count > best_count:
                    best_void = void_root
                    best_count = count
            # We've found our void and can continue to our next iteration
            current_group = best_void
            is_void = True
    return surrounding_features
    
@njit(parallel=True)
def find_all_atom_features(
        atom_grid_coords,
        parent,
        offset,
        solid,
        basin_labels,
        basin_feature_map,
        feature_dims,
        void_roots,
        void_dims,
        connected_voids, 
        connected_features, 
        connection_counts,
        ):
    
    # create lists to store which features surround each atom
    all_surrounding_features = []
    for i in range(len(atom_grid_coords)):
        feature_list = [-1]
        feature_list = feature_list[1:] # -1 is placeholder for numba typing
        all_surrounding_features.append(feature_list) 
        
    # TODO: Create tracker for if atom sits inside feature vs. surrounded by it?
    # Should be easy, but I'm not sure I would use it yet
        
    for atom_idx, atom_coords in enumerate(atom_grid_coords):
        surrounding_features = all_surrounding_features[atom_idx]
        find_atom_features(
            atom_coords,
            parent,
            offset,
            solid,
            basin_labels,
            basin_feature_map,
            feature_dims,
            void_roots,
            void_dims,
            connected_voids, 
            connected_features, 
            connection_counts,
            surrounding_features
            )
    return all_surrounding_features

@njit
def get_features_surrounding_atoms(
        feature_basins,
        feature_min_elfs,
        feature_max_elfs,
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
    unique_elfs = np.unique(feature_min_elfs)
    
    # create map from basin labels to feature labels
    basin_feature_map = np.empty(num_basins, dtype=np.uint32)
    
    # create an initial empty solid
    new_solid = np.zeros((nx,ny,nz), dtype=np.bool_)
    point_connections = np.arange(N, dtype=np.uint32)
    point_offsets = np.zeros((N, 3), dtype=np.int8) # can this be int8?
    cycles = None
    cycle_roots=None
    
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
        # get the features that exist at this elf value and construct basin map
        for feature_idx, (min_elf, max_elf) in enumerate(zip(feature_min_elfs, feature_max_elfs)):
            
            if min_elf <= elf_value and max_elf > elf_value:
                # This feature currently has some volume in the isosolid
                # create map for each basin in this feature
                for basin_idx in feature_basins[feature_idx]:
                    basin_feature_map[basin_idx] = feature_idx
            else:
                continue
            
            if min_elf == elf_value:
                # this feature just appeared.
                new_features.append(feature_idx)
        
        # get void/feature connection information at this elf value
        previous_solid = new_solid
        new_solid = data <= elf_value
        # TODO: I can probably improve speed further by freezing features/basins
        # that have been found to not surround anything, decreasing the number
        # of voxels that need to be checked for unions
        
        (
        void_roots, 
        void_dims, 
        cycles, 
        cycle_roots, 
        connected_voids, 
        connected_features, 
        connection_counts,
            ) = get_void_dimensionality(
                solid=new_solid,
                previous_solid=previous_solid,
                basin_labels=basin_labels,
                basin_feature_map=basin_feature_map,
                parent=point_connections,
                offset=point_offsets,
                cycles=cycles,
                cycle_roots=cycle_roots,
                neighbors=neighbor_transforms,
                )
                
        # Get which features surround each atom
        all_surrounding_features = find_all_atom_features(
                atom_grid_coords=atom_grid_coords,
                parent=point_connections,
                offset=point_offsets,
                solid=new_solid,
                basin_labels=basin_labels,
                basin_feature_map=basin_feature_map,
                feature_dims=feature_dims,
                void_roots=void_roots,
                void_dims=void_dims,
                connected_voids=connected_voids, 
                connected_features=connected_features, 
                connection_counts=connection_counts,
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





