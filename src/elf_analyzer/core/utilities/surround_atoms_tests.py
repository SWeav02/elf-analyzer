# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 14:54:29 2025

@author: sammw
"""

from numba import njit, prange
import numpy as np

from baderkit.core.methods.shared_numba import wrap_point
        
from elf_analyzer.core import ElfAnalyzer

@njit
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

@njit(parallel=True)
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

@njit(parallel=True)
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
            # if this point is below our limit, fill it
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

@njit
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
        print(len(edge_indices))
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

@njit
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

@njit(parallel=True)
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
        
@njit(parallel=True)
def get_surrounded_atoms(
    reducible_keys,
    basins,
    appear_at,
    disappear_at,
    basin_labels,
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

analyzer = ElfAnalyzer.from_vasp() 
num_basins = len(analyzer.basin_maxima_frac)       
shape = analyzer.reference_grid.shape
graph = analyzer.bifurcation_graph
neighbor_transforms, _ = analyzer.reference_grid.point_neighbor_transforms
basin_labels = analyzer.basin_labels
data = analyzer.reference_grid.total

reducible_keys = []
basins = []
appear_at = []
disappear_at = []
for node in graph:
    if node.reducible:
        reducible_keys.append(node.key)
        basins.append(node.basins)
        appear_at.append(node.appears_at)
        disappear_at.append(node.disappears_at)
reducible_keys = np.array(reducible_keys, dtype=np.int64)
appear_at = np.array(appear_at, dtype=np.float64)
disappear_at = np.array(disappear_at, dtype=np.float64)

atom_grid_coords = analyzer.reference_grid.frac_to_grid(analyzer.structure.frac_coords)
atom_grid_coords = np.round(atom_grid_coords).astype(np.int64) % analyzer.reference_grid.shape

feature_keys, feature_atoms = get_surrounded_atoms(
    reducible_keys,
    basins,
    appear_at,
    disappear_at,
    basin_labels,
    data,
    atom_grid_coords,
    neighbor_transforms,
    )