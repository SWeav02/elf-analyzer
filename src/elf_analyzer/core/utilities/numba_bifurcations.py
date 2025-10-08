# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from baderkit.core.methods.shared_numba import wrap_point

@njit
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
   

# @njit
# def flood_above(
#     data,
#     flood_labels,
#     edge_indices,
#     edge_neigh_vals,
#     flood_basin_connections,
#     max_value,
#     neighbor_transforms,
#         ):
#     max_label = len(flood_basin_connections)
#     # get supercell shape and single cell shape
#     snx, sny, snz = flood_labels.shape
#     nx, ny, nz = data.shape
    
#     # create a list to store new edges
#     new_edge_indices = []
#     new_edge_neigh_vals = []
#     # get the edge indices that border a point with an unlabeled value above
#     # the current cutoff
#     frontier_indices = []
#     for edge_index, edge_neigh_val in zip(edge_indices, edge_neigh_vals):
#         if edge_neigh_val >= max_value:
#             frontier_indices.append(edge_index)
#         else:
#             new_edge_indices.append(edge_index)
#             new_edge_neigh_vals.append(edge_neigh_val)
    
#     # flood until no change
#     while True:
#         if len(frontier_indices) == 0:
#             break
#         # create a list to store new points
#         new_frontier_indices = []
#         for i,j,k in frontier_indices:
#             # get the label at this index
#             label = flood_basin_connections[flood_labels[i,j,k]]
#             is_edge = False
#             # create a tracker for the highest neigh value that isn't available at
#             # the current cutoff
#             highest_unavail_neigh = -1.0
#             for si, sj, sk in neighbor_transforms:
#                 # get the neighbor in super cell and single cell coordinates
#                 ni = (i+si) % nx
#                 nj = (j+sj) % ny
#                 nk = (k+sk) % nz
#                 sni, snj, snk = wrap_point(i+si, j+sj, k+sk, snx, sny, snz)
#                 # get the value and label at this neighbor
#                 value = data[ni,nj,nk]
#                 neigh_label = flood_labels[sni,snj,snk]
#                 # check if this point is above the current and next elf value
#                 above_cutoff = value >= max_value
#                 labeled = neigh_label != max_label
                
#                 if above_cutoff and not labeled:
#                     # we want to flood this value
#                     flood_labels[sni, snj, snk] = label
#                     new_frontier_indices.append((sni, snj, snk))
#                 elif above_cutoff and labeled:
#                     # we border an already labeled point. Get its root
#                     neigh_label = flood_basin_connections[neigh_label]
#                     # if our labels don't match, this is a new connection
#                     if label != neigh_label:
#                         min_label = min(label, neigh_label)
#                         # update our connections
#                         flood_basin_connections[label] = min_label
#                         flood_basin_connections[neigh_label] = min_label
#                         for label_idx, root in enumerate(flood_basin_connections):
#                             flood_basin_connections[label_idx] = flood_basin_connections[root]
                        
#                 elif not above_cutoff:
#                     # The current point is on the edge for the next round
#                     is_edge = True
#                     if value > highest_unavail_neigh:
#                         highest_unavail_neigh = value
                    

#             # If we have an edge, add it to our list for the next round
#             if is_edge:
#                 new_edge_indices.append((i,j,k))
#                 new_edge_neigh_vals.append(highest_unavail_neigh)
#         # update our frontier
#         frontier_indices = new_frontier_indices
    
#     return flood_labels, new_edge_indices, new_edge_neigh_vals, flood_basin_connections



@njit(parallel=True)
def flood_above(
    data,
    flood_labels,
    flood_neigh_vals,
    flood_basin_connections,
    max_value,
    neighbor_transforms,
        ):
    max_label = len(flood_basin_connections)
    # get supercell shape and single cell shape
    snx, sny, snz = flood_labels.shape
    nx, ny, nz = data.shape
    
    # get our initial flood frontier
    flood_frontier = flood_neigh_vals >= max_value
    
    # flood until no change
    while True:
        # get our frontier indices
        frontier_indices = np.argwhere(flood_frontier)
        print(len(frontier_indices))
        new_flood = False
        for i,j,k in frontier_indices:
        # for frontier_idx in prange(len(frontier_indices)):
        #     i,j,k = frontier_indices[frontier_idx]
            # get the label root at this index
            label = flood_labels[i,j,k]
            # create a tracker for if this point is part of our edge
            is_edge = False
            # create a tracker for the highest neigh value that isn't available at
            # the current cutoff
            highest_unavail_neigh = -1.0
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

                # check if this point is above the current cutoff
                above_cutoff = value >= max_value
                # check if this point has been labeled already
                labeled = neigh_label != max_label
                
                if above_cutoff and not labeled:
                    # we want to flood this point and add it to our frontier
                    flood_labels[sni, snj, snk] = label
                    flood_frontier[sni, snj, snk] = True
                    # note we made at least one change
                    new_flood = True
                elif above_cutoff and labeled:
                    # This neighbor is labeled. We want to check if it belongs
                    # to a different group. First, we get the group each point
                    # belongs to.
                    root = flood_basin_connections[label]
                    neigh_root = flood_basin_connections[neigh_label]

                    # if our roots don't match, this is a new connection
                    if root == neigh_root:
                        # This isn't a new connection and we can continue
                        continue
                    # otherwise, we update our connections
                    lowest_root = min(root, neigh_root)
                    for basin_idx in (label, neigh_label, root, neigh_root):
                        flood_basin_connections[basin_idx] = lowest_root
                    
                elif not above_cutoff:
                    # The current point will flood to this neighbor in a future
                    # round. We record the next value where the current point
                    # will flood
                    is_edge = True
                    if value > highest_unavail_neigh:
                        highest_unavail_neigh = value
                    

            # If we have an edge, add it for next round
            if is_edge:
                flood_neigh_vals[i,j,k] = highest_unavail_neigh
            # otherwise, remove it from our edge using an unobtainably low value
            else:
                flood_neigh_vals[i,j,k] = -1.0e12
                
        # if we didn't flood to any new points, cancel
        if not new_flood:
            break
    
    return flood_labels, flood_neigh_vals, flood_basin_connections
    
@njit
def get_dimensionality(
    flood_labels,
    grid_shape,
    seed_coord,
    flood_basin_connections,
        ):
    # max_label = len(flood_basin_connections)
    nx, ny, nz = grid_shape
    transforms = [
            [0, 0, 0],  # -
            [1, 0, 0],  # x
            [0, 1, 0],  # y
            [0, 0, 1],  # z
            [1, 1, 0],  # xy
            [1, 0, 1],  # xz
            [0, 1, 1],  # yz
            [1, 1, 1],  # xyz
        ]
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
    for i, j, k in transforms:
        # shift coords
        x = ox + i*nx
        y = oy + j*ny
        z = oz + k*nz
        # get label
        # NOTE: There should always be a label here due to how we seed labels
        labels.append(flood_basin_connections[flood_labels[x,y,z]])
    
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
        ):
    # create an array to track which points are infinite
    dimensionalities = np.zeros(len(seed_points), dtype=np.int64)
    for seed_idx in prange(len(seed_points)):
        seed_coord = seed_points[seed_idx]
        dimensionalities[seed_idx] = get_dimensionality(
            flood_mask,
            grid_shape,
            seed_coord,
            flood_basin_connections,
            )
    return dimensionalities

@njit
def find_bifurcations(
    connection_pairs,
    connection_elfs,
    basin_maxima_grid,
    data,
    neighbor_transforms,
        ):
    nx, ny, nz = data.shape
    num_basins = len(basin_maxima_grid)
    # create an array to represent where each basin has flooded to
    flood_labels = np.full((nx*2, ny*2, nz*2), num_basins*8, dtype=np.uint16)
    # create a list to store the location of our edge
    edge_indices = []
    edge_neigh_vals = []
    # create a mask to represent which basins have been added to our flood
    flood_basins = np.zeros(len(basin_maxima_grid), dtype=np.bool_)
    # create an array representing which basins are connected to each other in
    # our flood
    flood_basin_connections = np.arange(num_basins*8, dtype=np.uint16)
    # get the transforms to the neighboring unit cells in a 2x2 supercell
    cell_transforms = [np.zeros(3,dtype=np.uint8)]
    for i in (0,1):
        for j in (0,1):
            for k in (0,1):
                if i == 0 and j == 0 and k == 0:
                    # skip 0 transform as we add it above to ensure its our first
                    # index
                    continue
                cell_transforms.append(np.array((i*nx,j*ny,k*nz), dtype=np.uint8))
    
    # get all possible elf values (including 0.0) and flip from high to low
    possible_elf_values = np.unique(connection_elfs)
    possible_elf_values = np.append(np.flip(possible_elf_values), [0.0])

    # create lists to store each bifurcation
    bifurcation_values = []
    bifurcation_features = []
    bifurcation_feature_indices = []
    bifurcation_dimensionalities = []
    
    # create an array representing which basins are connected to one another
    basin_connections = np.empty(len(basin_maxima_grid), dtype=np.int64)
    # and an array pointing each basin group to its corresponding list index
    feature_group_indices = basin_connections.copy()
    
    # create an empty set of feature groups representing the groups above the
    # highest value
    feature_groups = [[-1]]
    feature_ids = [-1]
    # do the same for dimensionalities of the features
    new_dimensionalities = np.array([-1], dtype=np.int64)
    # create a counter for the total number of unique features
    unique_features = 0
    # loop over elf values from high to low
    for val_idx, elf_value in enumerate(possible_elf_values):
        # get the connections that exist at this value
        connection_indices = np.where(connection_elfs>=elf_value)
        current_connections = connection_pairs[connection_indices]

        # reset our connections
        basin_connections[:] = -1
        feature_group_indices[:] = -1

        # loop over our connections
        for i, j in current_connections:
            # i is always lower than j by construction
            # check what group each basin is currently assigned to
            # TODO: Think through this more. Is it correct?
            lower_root = basin_connections[basin_connections[i]]
            higher_root = basin_connections[basin_connections[j]]
            
            # get the lowest possible connection among i, j, or the values they
            # are currently assigned to
            lowest_value = len(basin_connections)
            for val in (i, j, lower_root, higher_root):
                if val == -1:
                    continue
                if val < lowest_value:
                    lowest_value = val
            
            # connect each to the lowest value
            for basin_idx in (i, j, lower_root, higher_root):
                if basin_idx == -1:
                    continue
                basin_connections[basin_idx] = lowest_value
            

        # copy our previous groups
        previous_groups = [i for i in feature_groups]
        # create lists to store features basin groups
        feature_groups = []
        feature_seeds = []
        num_features = 0
        
        # Now loop over our basin connections and get our groups
        for basin_idx, basin_group in enumerate(basin_connections):
            # skip if this basin isn't assigned
            if basin_group == -1:
                continue
            # Our basin group may not be pointing to the true root. Because we
            # are looping from low to high, we can update this once during this
            # loop
            basin_group = basin_connections[basin_group]
            basin_connections[basin_idx] = basin_group
            
            # check if this basin has been seeded already
            seed_point = basin_maxima_grid[basin_idx]
            if not flood_basins[basin_idx]:
                # note that we've seeded this basin
                flood_basins[basin_idx] = True
                # seed this basin in each neighboring cell
                for trans_idx, trans in enumerate(cell_transforms):
                    x, y, z = seed_point + trans
                    # get the adjusted basin index for this cell
                    cell_basin_idx = basin_idx + trans_idx*num_basins
                    # add it to our flood
                    flood_labels[x,y,z] = cell_basin_idx
                    edge_indices.append((x,y,z))
                    edge_neigh_vals.append(1.0e6) # always higher than our cutoff
            
            # check if this basin is the lowest in the group
            if basin_idx == basin_group:
                # create a new feature
                feature_groups.append([basin_idx])
                feature_group_indices[basin_idx] = num_features
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
        previous_group_ids = [i for i in feature_ids]
        feature_ids = []
        same_groups = True
        for group in feature_groups:
            # check to see if this group exists in the previous groups
            group_found = False
            for pgroup, pgroup_idx in zip(previous_groups, previous_group_ids):
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
        
        # if elf_value == 0.17199:
        #     breakpoint()
        # We also want to calculate the dimensionality of each feature, regardless
        # of if we have a new group or not.
        # Flood fill to the current value
        flood_labels, edge_indices, edge_neigh_vals, flood_basin_connections = flood_above(
            data,
            flood_labels,
            edge_indices,
            edge_neigh_vals,
            flood_basin_connections,
            elf_value,
            neighbor_transforms,
            )
        # get the new dimensionalities
        old_dimensionalities = new_dimensionalities.copy()
        new_dimensionalities = get_dimensionality_all(
            flood_labels,
            data.shape,
            feature_seeds,
            flood_basin_connections,
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
            
        # BUG: This is returning only one value for each currently
    return bifurcation_values, bifurcation_features, bifurcation_feature_indices, bifurcation_dimensionalities
        
        
    
    
    
    
    
    
    
    