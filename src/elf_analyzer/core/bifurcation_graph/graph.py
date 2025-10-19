# -*- coding: utf-8 -*-

import json
import logging
import time
import itertools

import numpy as np
import plotly.graph_objects as go

from baderkit.core import Bader #, Structure

from elf_analyzer.core.bifurcation_graph.nodes import IrreducibleNode, ReducibleNode, NodeBase
from elf_analyzer.core.bifurcation_graph.plot_colors import NODE_COLORS, LINE_COLOR
# from elf_analyzer.core.utilities import IonicRadiiTools
from elf_analyzer.core.bifurcation_graph.infinite_feature_numba import (
    find_connections,
    find_bifurcations
    )
from elf_analyzer.core.bifurcation_graph.surrounded_atoms_numba import get_features_surrounding_atoms

class BifurcationGraph:
    """
    A class for storing the nodes of a bifurcation graph. The nodes themselves
    contain the information on their connectivity.
    """
    
    def __init__(self, bader: Bader = None, labeler = "ElfAnalyzer"):
        self._root_nodes = []
        self._nodes = []
        self._node_keys = {}
        
        # optional bader parameter
        self.bader = bader
        
        # optional labeler parameter for tracking what method was used to label
        # features
        self.labeler = labeler

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
    def unassigned_irreducible_nodes(self):
        return [i for i in self if i.feature_type == "irreducible"]
    
    def to_dict(self) -> dict:
        # NOTE: This could just be a to list method, but I may add other meta
        # data down the line
        # create our initial dict
        graph_dict =  {
            "nodes": [i.to_dict() for i in self],
            }

        return graph_dict
    
    @classmethod
    def from_dict(cls, graph_dict: dict):
        nodes = graph_dict["nodes"]
        
        # create our initial graph object
        graph = cls()
        
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
    def from_bader(cls, bader: Bader, **kwargs):
        reference_grid = bader.reference_grid
        neighbor_transforms, _ = reference_grid.point_neighbor_transforms
        
        
        #######################################################################
        # Get Bifurcation Values and Corresponding Features
        #######################################################################
        
        logging.info("Locating Bifurcations")
        t0 = time.time()
        
        # get connections between neighboring basins
        lower_points, upper_points, connection_values = find_connections(
            bader.basin_labels,
            reference_grid.total,
            bader.basin_edges,
            len(bader.basin_maxima_frac),
            neighbor_transforms,
            )
        # add maxima values as the points each basin "connects" to itself
        basin_maxima = bader.basin_maxima_ref_values
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
        basin_maxima_grid = np.round(bader.reference_grid.frac_to_grid(bader.basin_maxima_frac)).astype(np.int64)
        basin_maxima_grid %= bader.reference_grid.shape
        
        basin_maxima_ref_values=bader.basin_maxima_ref_values
        
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

        # get atom grid coordinates
        atom_grid_coords = reference_grid.frac_to_grid(bader.structure.frac_coords)
        atom_grid_coords = np.round(atom_grid_coords).astype(np.int64) % reference_grid.shape
        
        # get the atoms each feature contains
        feature_atoms = get_features_surrounding_atoms(
                feature_basins=feature_basins,
                feature_min_values=feature_min_values,
                feature_max_values=feature_max_values,
                feature_dims=feature_dims,
                atom_grid_coords=atom_grid_coords,
                neighbor_transforms=neighbor_transforms,
                basin_labels=bader.basin_labels,
                data=reference_grid.total,
                num_basins=len(bader.basin_maxima_frac),
                )
        t2 = time.time()
        logging.info(f"Time: {round(t2-t1, 2)}")

        #######################################################################
        # Construct Graph
        #######################################################################    
        graph = cls(bader=bader, **kwargs)
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
                frac_coords = bader.basin_maxima_frac[basin_idx]
                charge = bader.basin_charges[basin_idx]
                volume = bader.basin_volumes[basin_idx]
                atom_distance = bader.basin_atom_dists[basin_idx]
                nearest_atom = bader.basin_atoms[basin_idx]
                nearest_atom_type = bader.structure[nearest_atom].specie.symbol
                
                node = IrreducibleNode(
                    bifurcation_graph=graph,
                    basins=feature_basins[feat_idx], 
                    dimensionality=feature_dims[feat_idx], 
                    contained_atoms=feature_atoms[feat_idx], 
                    min_value=feature_min_values[feat_idx], 
                    max_value=feature_max_values[feat_idx],
                    nearest_atom=nearest_atom,
                    nearest_atom_type=nearest_atom_type,
                    frac_coords=frac_coords,
                    charge=charge,
                    volume=volume,
                    atom_distance=atom_distance,
                    parent=parent,
                    )

            node_keys.append(node.key)
                
        # TODO: Combine ring/spheres to singular basins. Calculate coord envs
        # for irreducible features?
        # Move plot method to here. Add get_plot_label methods to nodes?
        
        return graph
        
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
        types = [] # The type of feature (reducible, core, covalent, etc.)
        for i, node in enumerate(self):
            # get info for each node
            labels.append(node.plot_label)
            types.append(node.feature_type)
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
        for idx, (feature_type, label) in enumerate(zip(types, labels)):
            # get color
            color = NODE_COLORS.get(feature_type)
            
            # only add to legend if this type hasn't been found previously
            showlegend = feature_type not in already_added_types
            already_added_types.add(feature_type)

            if feature_type == "reducible":
                # add a circle
                fig.add_trace(
                    go.Scatter(
                        x=[Xn[idx]],
                        y=[Yn[idx]],
                        mode="markers",
                        name=f"{feature_type}",
                        marker=dict(
                            symbol="circle-dot",
                            size=18,
                            color=color,
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
                        fillcolor=color,
                        line=dict(color=LINE_COLOR),
                        hoverinfo="text",
                        text=label,
                        name=f"{feature_type}",
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
            xaxis=dict(range=[min_x-buffer, max_x+buffer], title=f"{self.labeler} Bifurcations"),
            yaxis=dict(
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
            ),
        )
        return fig
        