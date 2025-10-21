# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange

from baderkit.core.methods.shared_numba import wrap_point, coords_to_flat

from elf_analyzer.core.bifurcation_graph.infinite_feature_numba import get_connected_features
from elf_analyzer.core.utilities.union_find import find_root_no_compression

@njit(cache=True, parallel=True)
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
    for i in prange(nx):
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
                    
                    # this isn't technically safe, but I think it will usually
                    # have very little effect and it makes a big difference in
                    # speed
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



@njit(cache=True)
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
    
@njit(parallel=True,cache=True)
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

@njit(cache=True)
def get_features_surrounding_atoms(
        possible_values,
        feature_basins,
        feature_min_values,
        feature_max_values,
        feature_dims,
        feature_parents,
        atom_grid_coords,
        neighbor_transforms,
        basin_labels,
        data,
        num_basins,
        ):
    nx, ny, nz = data.shape
    N = nx * ny * nz
    
    # create map from basin labels to feature labels
    basin_feature_map = np.empty(num_basins, dtype=np.uint32)
    
    # create a map for old features to new
    new_feature_map = np.full(len(feature_parents), -1, dtype=np.int64)
    
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
    
    # create lists for updated features
    new_feature_basins = []
    new_feature_min_values = []
    new_feature_max_values = []
    new_feature_dims = []
    new_feature_parents = []
    
    # create lists to track which features surround atoms
    all_feature_atoms = []
    
    current_features = [-1]
    current_features = current_features[1:]
    current_feature_atoms = [[-1]]
    current_feature_atoms = current_feature_atoms[1:]
    # iterate over all possible values
    for current_value in possible_values:
        # NOTE: I don't think we need to set the basin_feature_map to a default
        # value because it shouldn't be possible for a void to touch a feature
        # other than one in this list
        previous_features = current_features.copy()
        current_features = []
        new_features = []
        num_features = 0
        # get the features that exist at this elf value and construct basin map
        for feature_idx, (min_value, max_value) in enumerate(zip(feature_min_values, feature_max_values)):
            
            if min_value <= current_value and max_value > current_value:
                # This feature currently has some volume in the isosolid
                # create map for each basin in this feature
                # add it to our list
                current_features.append(feature_idx)
                # map each basin in this feature back to the feature
                for basin_idx in feature_basins[feature_idx]:
                    basin_feature_map[basin_idx] = num_features
                # note we found a new feature
                num_features += 1
                if min_value == current_value:
                    new_features.append(feature_idx)

        
        # get void/feature connection information at this elf value
        previous_solid = new_solid
        new_solid = data <= current_value
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
                all_features=current_features,
                connection_counts=connection_counts,
                void_roots=roots,
                feature_dims=feature_dims,
                void_dims=dimensionalities,
                )
        
        # Get the atoms surrounded by each feature
        # track if no atoms are surrounded
        no_atoms_surrounded = True
        previous_feature_atoms = current_feature_atoms.copy()
        current_feature_atoms = []
        for i in range(len(current_features)):
            feature_list = [-1]
            feature_list = feature_list[1:]
            current_feature_atoms.append(feature_list)
        # Add atoms to feature lists
        for atom_idx, feature_list in enumerate(all_surrounding_features):
            if len(feature_list) == 0:
                continue
            no_atoms_surrounded = False
            for feature_idx in feature_list:
                for sub_idx, current_feature_idx in enumerate(current_features):
                    if feature_idx == current_feature_idx:
                        current_feature_atoms[sub_idx].append(atom_idx)
                        break
        
        # append any new features
        for feature_idx, feature_atoms in zip(current_features, current_feature_atoms):
            append = False
            is_new = False
            if feature_idx in new_features:
                append = True
            else:
                # check if this feature changed the number of atoms it contains
                for old_feature_idx, old_feature_atoms in zip(previous_features, previous_feature_atoms):
                    if old_feature_idx == feature_idx:
                        if len(old_feature_atoms) != len(feature_atoms):
                            append = True
                            is_new = True
                        break
            if append:
                new_feature_basins.append(feature_basins[feature_idx])
                new_feature_min_values.append(current_value)
                new_feature_max_values.append(feature_max_values[feature_idx])
                new_feature_dims.append(feature_dims[feature_idx])
                all_feature_atoms.append(feature_atoms)
                if is_new:
                    previous_idx = new_feature_map[feature_idx]
                    # This feature changed atom counts. Update its old max
                    # value to be the current value
                    new_feature_max_values[previous_idx] = current_value
                    # set a new mapping for this feature
                    new_feature_map[feature_idx] = len(new_feature_parents)
                    # set the parent to the previous version surrounding a different
                    # number of atoms
                    new_feature_parents.append(previous_idx)
                else:
                    # update this features index
                    new_feature_map[feature_idx] = len(new_feature_parents)
                    # get the old parent for this feature
                    old_parent = feature_parents[feature_idx]
                    if old_parent == -1:
                        new_feature_parents.append(-1)
                        continue
                    # get the parents new index
                    new_parent = new_feature_map[old_parent]
                    new_feature_parents.append(new_parent)
                    
        # if no atoms are surrounded, they never will be again and we are done
        # here
        if no_atoms_surrounded:
            break
    
    # we need to fill in the data for any remaining features that don't surround
    # atoms
    for feature_idx in range(len(feature_min_values)):
        # check if this feature was ever given a mapping
        if not new_feature_map[feature_idx] == -1:
            continue
        # otherwise, we add all of the needed features
        new_feature_basins.append(feature_basins[feature_idx])
        new_feature_min_values.append(feature_min_values[feature_idx])
        new_feature_max_values.append(feature_max_values[feature_idx])
        new_feature_dims.append(feature_dims[feature_idx])
        all_feature_atoms.append([-1])
        all_feature_atoms[-1] = all_feature_atoms[-1][1:]
        # update this features index
        new_feature_map[feature_idx] = len(new_feature_parents)
        # get the old parent for this feature
        old_parent = feature_parents[feature_idx]
        if old_parent == -1:
            new_feature_parents.append(-1)
            continue
        # get the parents new index
        new_parent = new_feature_map[old_parent]
        new_feature_parents.append(new_parent)
        
    return (
        new_feature_basins, 
        np.array(new_feature_min_values, dtype=np.float64), 
        np.array(new_feature_max_values, dtype=np.float64), 
        np.array(new_feature_dims, dtype=np.int64), 
        np.array(new_feature_parents, dtype=np.int64),
        all_feature_atoms,
        )





