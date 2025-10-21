# -*- coding: utf-8 -*-

import json
import logging
import time
import itertools

import numpy as np
import plotly.graph_objects as go

from baderkit.core import Bader, Structure

from elf_analyzer.core.bifurcation_graph.nodes import IrreducibleNode, ReducibleNode, NodeBase
from elf_analyzer.core.bifurcation_graph.feature_mappings import LINE_COLOR, FeatureSubtype
# from elf_analyzer.core.utilities import IonicRadiiTools
from elf_analyzer.core.bifurcation_graph.infinite_feature_numba import (
    find_domain_connections,
    find_bifurcations,
    find_potential_saddle_points
    )
from elf_analyzer.core.bifurcation_graph.surrounded_atoms_numba import (
    get_features_surrounding_atoms,
    )
from elf_analyzer.core.utilities.numba_functions import get_min_avg_feat_surface_dists, get_feature_edges

class BifurcationGraph:
    """
    A class for storing the nodes of a bifurcation graph. The nodes themselves
    contain the information on their connectivity.
    """
    
    def __init__(
            self, 
            structure: Structure, 
            labeler: Bader = None, 
            labeler_type = None,
            atomic_radii = None,
            ):
        self._root_nodes = []
        self._nodes = []
        self._node_keys = {}
        
        self.structure = structure
        
        # optional labeler parameter
        self.labeler = labeler
        
        # optional labeler parameter for tracking what method was used to label
        # features
        if labeler is None:
            self.labeler_type = labeler_type
        else:
            self.labeler_type = labeler.__class__.__name__
        
        # optional radii parameter that must be set
        self._atomic_radii = None

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, key):
        return self.nodes[key]

    def __contains__(self, key):
        return key in self.nodes

    def __repr__(self):
        return f"BifurcationGraph(num_nodes={len(self._nodes)})"
    
    @property
    def root_nodes(self):
        return self._root_nodes
    
    @property
    def nodes(self):
        return self._nodes
        
    def node_from_key(self, key):
        return self._node_keys[key]
    
    @property
    def irreducible_nodes(self):
        return [i for i in self if not i.is_reducible]
    
    @property
    def reducible_nodes(self):
        return [i for i in self if i.is_reducible]
    
    @property
    def unassigned_nodes(self):
        return [i for i in self if i.feature_subtype is None or i.feature_subtype == "shallow"]
    
    @property
    def atomic_radii(self):
        if self._atomic_radii is None:
            if self.labeler is not None:
                self._atomic_radii = self.labeler.atomic_radii
        return self._atomic_radii
    
    def nodes_by_type(self, feature_subtype: str):
        return [i for i in self if i.feature_subtype == feature_subtype]
    
    def nodes_by_types(self, feature_subtypes: list[str]):
        return [i for i in self if i.feature_subtype in feature_subtypes]
    
    def to_dict(self) -> dict:
        radii = self.atomic_radii
        if radii is not None:
            radii = [float(i) for i in radii]
        graph_dict =  {
            "nodes": [i.to_dict() for i in self],
            "structure": self.structure.to_json(),
            "labeler_type": self.labeler_type,
            "atomic_radii": radii,
            }

        return graph_dict
    
    @classmethod
    def from_dict(cls, graph_dict: dict):
        nodes = graph_dict["nodes"]
        structure = Structure.from_str(graph_dict["structure"], fmt="json")
        labeler_type = graph_dict["labeler_type"]
        atomic_radii = graph_dict["atomic_radii"]
        if atomic_radii is not None:
            atomic_radii = np.array(atomic_radii, dtype=np.float64)
        
        # create our initial graph object
        graph = cls(structure=structure, labeler_type=labeler_type, atomic_radii=atomic_radii)
        
        # add nodes
        for node_dict in nodes:
            NodeBase.from_dict(graph, node_dict)
        
        return graph

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str):
        graph_dict = json.loads(json_str)
        return cls.from_dict(graph_dict)
    
    @classmethod
    def from_labeler(cls, labeler: Bader):
        reference_grid = labeler.reference_grid
        neighbor_transforms, _ = reference_grid.point_neighbor_transforms
        
        
        #######################################################################
        # Get Bifurcation Values and Corresponding Features
        #######################################################################
        
        logging.info("Locating Bifurcations")
        t0 = time.time()
        
        # get mask where potential saddle points connecting features exist
        bif_mask = find_potential_saddle_points(
            data=reference_grid.total,
            edge_mask=labeler.basin_edges,
            greater=True
            )
        
        # get the basins connected at these points
        lower_points, upper_points, connection_values = find_domain_connections(
            basin_labels=labeler.basin_labels,
            data=reference_grid.total,
            bif_mask=bif_mask,
            neighbor_transforms=neighbor_transforms,
            )

        # clear mask for memory
        bif_mask = None

        # add maxima values as the points each basin "connects" to itself
        basin_maxima = labeler.basin_maxima_ref_values
        basin_indices = np.arange(len(basin_maxima))
        lower_points = np.append(lower_points, basin_indices)
        upper_points = np.append(upper_points, basin_indices)
        connection_values = np.append(connection_values, basin_maxima)
        
        # group and get unique
        connection_array = np.column_stack((lower_points, upper_points, connection_values))
        unique_connections, unique_indices = np.unique(connection_array, return_index=True, axis=0)
        
        # get pairs of connections
        lower_points = lower_points[unique_indices]
        upper_points = upper_points[unique_indices]
        connection_pairs=np.column_stack((lower_points, upper_points))
        
        # get values of connections
        connection_values = connection_values[unique_indices]
        
        # breakpoint()
        basin_maxima_grid = np.round(labeler.reference_grid.frac_to_grid(labeler.basin_maxima_frac)).astype(np.int64)
        basin_maxima_grid %= labeler.reference_grid.shape
        
        basin_maxima_ref_values=labeler.basin_maxima_ref_values
        
        (
            feature_basins,
            feature_min_values,
            feature_max_values,
            feature_dims,
            feature_parents,
            ) = find_bifurcations(
            connection_pairs,
            connection_values,
            basin_maxima_grid,
            basin_maxima_ref_values,
            reference_grid.total,
            neighbor_transforms,
                )
        # convert basins to numpy arrays to avoid Numba reflected list issue
        feature_basins = [np.array(i, dtype=np.int64) for i in feature_basins]
        
        t1 = time.time()
        logging.info(f"Time: {round(t1-t0, 2)}")
        
        #######################################################################
        # Get Atoms Surrounded by Each Feature
        #######################################################################
        logging.info("Finding contained atoms")    
        
        # possible saddle points where voids between features first connect
        bif_mask = find_potential_saddle_points(
            data=reference_grid.total,
            edge_mask=labeler.basin_edges,
            greater=False
            )
        # get the possible values and clear mask
        bif_values = reference_grid.total[bif_mask]
        bif_mask = None

        # add the values from the feature bifurcations and get only the
        # unique options
        bif_values = np.unique(np.append(bif_values, feature_min_values))
        
        # get atom grid coordinates
        atom_grid_coords = reference_grid.frac_to_grid(labeler.structure.frac_coords)
        atom_grid_coords = np.round(atom_grid_coords).astype(np.int64) % reference_grid.shape

        # get the atoms each feature contains
        (
            feature_basins,
            feature_min_values,
            feature_max_values,
            feature_dims,
            feature_parents,
            feature_atoms,
            ) = get_features_surrounding_atoms(
                possible_values=bif_values,
                feature_basins=feature_basins,
                feature_min_values=feature_min_values,
                feature_max_values=feature_max_values,
                feature_dims=feature_dims,
                feature_parents=feature_parents,
                atom_grid_coords=atom_grid_coords,
                neighbor_transforms=neighbor_transforms,
                basin_labels=labeler.basin_labels,
                data=reference_grid.total,
                num_basins=len(labeler.basin_maxima_frac),
                )
        t2 = time.time()
        logging.info(f"Time: {round(t2-t1, 2)}")

        #######################################################################
        # Construct Graph
        #######################################################################    
        graph = cls(structure=labeler.structure, labeler=labeler)
        node_keys = []
        for feat_idx in range(len(feature_basins)):
            if feature_parents[feat_idx] == -1:
                parent = None
            else:
                parent_key = node_keys[feature_parents[feat_idx]]
                parent = graph.node_from_key(parent_key)
            if len(feature_basins[feat_idx]) > 1 or feature_dims[feat_idx] > 0:
                # This is a reducible feature
                node = ReducibleNode(
                    bifurcation_graph=graph,
                    basins=feature_basins[feat_idx], 
                    dimensionality=feature_dims[feat_idx], 
                    contained_atoms=feature_atoms[feat_idx], 
                    min_value=feature_min_values[feat_idx], 
                    max_value=feature_max_values[feat_idx],
                    parent=parent,
                    )
            
            else:
                # this is an irreducible feature. Get additional information
                basin_idx = feature_basins[feat_idx][0]
                frac_coords = labeler.basin_maxima_frac[basin_idx]
                charge = labeler.basin_charges[basin_idx]
                volume = labeler.basin_volumes[basin_idx]
                nearest_atom = labeler.basin_atoms[basin_idx]

                node = IrreducibleNode(
                    bifurcation_graph=graph,
                    basins=feature_basins[feat_idx], 
                    dimensionality=feature_dims[feat_idx], 
                    contained_atoms=feature_atoms[feat_idx], 
                    min_value=feature_min_values[feat_idx], 
                    max_value=feature_max_values[feat_idx],
                    nearest_atom=nearest_atom,
                    frac_coords=frac_coords,
                    charge=charge,
                    volume=volume,
                    parent=parent,
                    )

            node_keys.append(node.key)
                
        # give each reducible node a subtype
        for node in graph.reducible_nodes:
            parent = node.parent
            if parent is None:
                node.feature_subtype = "root"
                continue
            # if the number of basins changes, this is a standard reducible domain
            elif len(parent.basins) != len(node.basins):
                node.feature_subtype = "reducible"
            # check for dimension change
            elif parent.dimensionality != node.dimensionality:
                node.feature_subtype = "dimension reduction"
            elif len(parent.contained_atoms) != len(node.contained_atoms):
                node.feature_subtype = "contained atom reduction"
        
        # sometimes we get extremely shallow reducible features that seem to
        # result from voxelation. Their depth is only one significant figure
        # deep
        # cls._remove_shallow_reducible_nodes(graph)
            
        
        # Now we check for reducible nodes that should really be considered
        # irreducible. These nodes are very deep but their children separate
        # at very low values
        cls._combine_shallow_irreducible_nodes(graph)
        
        return graph
    
    @staticmethod
    def _remove_shallow_reducible_nodes(graph, cutoff=0.05):
        reducible_nodes = graph.reducible_nodes.copy()
        reducible_nodes.reverse()
        for node in reducible_nodes:
            parent = node.parent
            if parent is None:
                continue
            # The parent must only differ in dimensionality
            if not np.all(np.isin(parent.basins, node.basins)):
                continue
            # if the node is exceedingly shallow, it probably is caused
            # by voxelation
            if (node.depth / node.parent.depth) < cutoff:
                node.remove()
    
    @staticmethod
    def _combine_shallow_irreducible_nodes(graph, cutoff=0.05):
        # TODO: Add check that nodes are at relatively similar values
        nodes_to_combine = []
        checked_nodes = []
        for node in graph.reducible_nodes.copy():
            # skip infinite nodes and nodes we've already checked
            if node.is_infinite or node.key in checked_nodes:
                continue
            # get this nodes depth
            depth = node.depth
            is_shallow = True
            # get all children
            for child in node.deep_children:
                # skip other reducible nodes
                if child.is_reducible:
                    continue
                # check if depth is more than the cutoff portion of the parent's depth. If so,
                # we don't consider this feature to be shallow
                if (child.depth / depth) > cutoff:
                    is_shallow = False
                    break
            
            if not is_shallow:
                continue
            
            # note we want to combine this node
            nodes_to_combine.append(node)
            # if the node is shallow, we will combine all of its children later.
            # for now, we note that its children have already been checked
            for child in node.deep_children:
                if child.is_reducible:
                    checked_nodes.append(child.key)

        # now we combine all of the nodes 
        for node in nodes_to_combine:
            node.make_irreducible()
                
    def _calculate_feature_surface_dists(self):
        # Calculate the minimum and average distance from each irreducible features
        # fractional coordinate to its edges. This may be different from the
        # original basins because we may have merged some of them
        if self.labeler is None:
            logging.warning("Surface distances can only be calculated when a graph is created using a labeler (e.g. ElfAnalyzer)")
            return
        
        labeler = self.labeler
        nodes = self.irreducible_nodes
        
        # collect frac coords and map basin labels to features
        frac_coords = [i.frac_coords for i in nodes]
        feature_map = np.empty(len(labeler.basin_maxima_frac), dtype=np.uint32)
        for node_idx, node in enumerate(nodes):
            feature_map[node.basins] = node_idx
        
        # get feature edges
        neighbor_transforms, _ = labeler.reference_grid.point_neighbor_transforms
        edge_mask = get_feature_edges(
            labeled_array=labeler.basin_labels,
            feature_map=feature_map,
            neighbor_transforms = neighbor_transforms,
            vacuum_mask=labeler.vacuum_mask,
            )
        
        # calculate the minimum and average distance to each features surface
        min_dists, avg_dists = get_min_avg_feat_surface_dists(
            labels=labeler.basin_labels,
            feature_map=feature_map,
            frac_coords=np.array(frac_coords, dtype=np.float64),
            edge_mask=edge_mask,
            matrix=labeler.reference_grid.matrix,
            max_value=np.max(self.structure.lattice.abc) * 2,
            )
        
        # set surface distances
        for node, min_dist, avg_dist in zip(nodes, min_dists, avg_dists):
            node.min_surface_dist = min_dist
            node.avg_surface_dist = avg_dist
        
    def get_plot(self) -> go.Figure:
        """
        Returns a plotly figure
        """
        #######################################################################
        # X Values 
        #######################################################################
        indices = [] # The key for each node
        Xn = [] # The X value where the node appears
        Xn1 = []  # The X value where the node disappears
        labels = [] # Strings summarizing node features
        types = [] # The type of feature (reducible, irreducible)
        subtypes = [] # The subtype of feature (reducible, dim change, core, covalent, etc.)
        for i, node in enumerate(self):
            # get info for each node
            labels.append(node.plot_label)
            types.append(node.feature_type)
            subtypes.append(node.feature_subtype)
            indices.append(node.key)
            Xn.append(round(node.min_value, 4))
            Xn1.append(max(node.max_value, node.max_value - node.depth + 0.01))
        
        #######################################################################
        # Y Values 
        #######################################################################
        
        def assign_y_positions(node, y_counter, y_positions):
            # This function iteratively loops starting from the root node and
            # places each parent node at the average position of its children.
            # children are placed when found. The iterative nature results in
            # connecting lines not overlapping.
            if not node.is_reducible: # it's a leaf
                y_positions[node.key] = next(y_counter)
            else: # its a branch
                children = node.children
                for child in children:
                    assign_y_positions(child, y_counter, y_positions)
                child_ys = [y_positions[child.key] for child in children]
                y_positions[node.key] = np.mean(child_ys)
        
        # Create a mapping from node ID to Y position
        y_positions = {}
        y_counter = itertools.count(0)  # This gives 0, 1, 2, ... for leaf placement
        
        # BUGFIX: We may have multiple roots (e.g. molecules separated by vacuum)
        # so we find the y values separately then adjust
        for root_node in self.root_nodes:
            assign_y_positions(root_node, y_counter, y_positions)

        # Then set Yn using our dict
        Yn = [y_positions[i] for i in indices]
        
        # Normalize Y scale
        max_y = 2
        Yn = np.array(Yn, dtype=float)
        Yn -= Yn.min()
        if Yn.max() > 0:
            Yn /= Yn.max()
            Yn *= max_y
        # Get the height of each irreducible node
        y_division = max_y / len(self.irreducible_nodes)

        # Now we need to get the lines that will be used for each edge. These will use
        # a nested lists where each edge has one entry and the sub-lists contain the
        # two x and y entries for each edge.
        Xe = []
        Ye = []
        for node in self.reducible_nodes:
            parent = node.key
            children = node.children
            for child_node in children:
                child = child_node.key
                px = Xn[indices.index(parent)]
                py = Yn[indices.index(parent)]
                cx = Xn[indices.index(child)]
                cy = Yn[indices.index(child)]
        
                # Vertical segment: (px, py) -> (px, cy)
                Xe.extend([px, px, None])
                Ye.extend([py, cy, None])
        
                # Horizontal segment: (px, cy) -> (cx, cy)
                Xe.extend([px, cx, None])
                Ye.extend([cy, cy, None])
        
        # create the figure and add the lines
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=Xe,
                y=Ye,
                mode="lines",
                name="connection",
                line=dict(color=LINE_COLOR, width=3),
                hoverinfo="none",
            )
        )

        # convert lists to numpy arrays for easy querying.
        types = np.array(types)
        labels = np.array(labels)
        Xn = np.array(Xn)
        Xn1 = np.array(Xn1)
        Yn = np.array(Yn)
        Yn0 = Yn - y_division / 3
        Yn1 = Yn + y_division / 3
        already_added_types = set()

        for idx, (feature_type, feature_subtype, label) in enumerate(zip(types, subtypes, labels)):
            
            # only add to legend if this type hasn't been found previously
            showlegend = feature_type not in already_added_types
            already_added_types.add(feature_type)

            if feature_type == "ReducibleNode":
                # add a circle
                fig.add_trace(
                    go.Scatter(
                        x=[Xn[idx]],
                        y=[Yn[idx]],
                        mode="markers",
                        name=f"{feature_subtype}",
                        marker=dict(
                            symbol="circle-dot",
                            size=18,
                            color=feature_subtype.plot_color,
                            line=dict(color="grey", width=1),
                        ),
                        text=label,
                        hoverinfo="text",
                        showlegend=showlegend,
                    )
                )
            else:
                # add a rectangle
                x0 = Xn[idx]
                x1 = Xn1[idx]
                y0 = Yn0[idx]
                y1 = Yn1[idx]
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1, x1, x0, x0],
                        y=[y0, y0, y1, y1, y0],
                        fill="toself",
                        fillcolor=feature_subtype.plot_color,
                        line=dict(color=LINE_COLOR),
                        hoverinfo="text",
                        text=label,
                        name=f"{feature_subtype}",
                        mode="lines",
                        opacity=0.8,
                        showlegend=showlegend,
                    )
                )

        
        min_x = min(Xn)
        max_x = max(Xn1)
        x_range = max_x - min_x
        buffer = x_range * 0.05
        # remove y axis label and add title
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(range=[min_x-buffer, max_x+buffer], title=f"{self.labeler_type} Bifurcations"),
            yaxis=dict(
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
            ),
        )
        return fig
        