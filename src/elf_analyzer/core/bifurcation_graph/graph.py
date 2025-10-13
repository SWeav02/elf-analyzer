# -*- coding: utf-8 -*-

import json
import logging

import numpy as np

from baderkit.core import Structure, Bader

from elf_analyzer.core.bifurcation_graph.nodes import IrreducibleNode, ReducibleNode, NodeBase

from elf_analyzer.core.utilities import BifurcationGraph, IonicRadiiTools
from elf_analyzer.core.utilities.numba_functions import (
    check_all_covalent,
    find_connections,
    )
from elf_analyzer.core.utilities.solid_dimensionality_test import (
    find_bifurcations
    )
from elf_analyzer.core.utilities.surrounds_atom_test import (
    get_features_surrounding_atoms
    )

class BifurcationGraph:
    """
    A class for storing the nodes of a bifurcation graph. The nodes themselves
    contain the information on their connectivity.
    """
    
    def __init__(self, bader: Bader = None, labeler = "ElfAnalyzer"):
        self._root_node = None
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
    def root_node(self):
        return self._root_node
    
    @property
    def nodes(self):
        return self._nodes
        
    def node_from_key(self, key):
        return self._node_keys[key]
    
    def add_reducible_node(self, **kwargs) -> ReducibleNode:
        node = ReducibleNode(
            bifurcation_graph=self,
            **kwargs,
            )
        return node
    
    def add_irreducible_node(self, **kwargs) -> IrreducibleNode:
        node = IrreducibleNode(
            bifurcation_graph=self,
            **kwargs,
            )
        return node
    
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
        neighbor_transforms = reference_grid.point_neighbor_transforms
        
        
        #######################################################################
        # Get Bifurcation Values and Corresponding Features
        #######################################################################
        
        logging.info("Locating Bifurcations")
        
        # get connections between neighboring basins
        edge_indices = np.argwhere(bader.basin_edges)
        connection_array = find_connections(
            bader.basin_labels,
            reference_grid.total,
            edge_indices,
            len(bader.basin_maxima_frac),
            neighbor_transforms,
            )
        # also add the maximum value of each basin as the point it 'connects' to
        # itself
        basin_maxima = bader.basin_maxima_ref_values
        basin_indices = np.arange(len(basin_maxima))
        connection_array[basin_indices, basin_indices] = basin_maxima
        
        connection_indices = np.nonzero(connection_array)
        connection_pairs = np.column_stack(connection_indices)  # same as argwhere result
        connection_elfs = connection_array[connection_indices]
        
        basin_maxima_grid = np.round(bader.reference_grid.frac_to_grid(bader.basin_maxima_frac)).astype(np.int64)
        basin_maxima_grid %= bader.reference_grid.shape
        
        basin_maxima_ref_values=bader.basin_maxima_ref_values
        
        (
            feature_basins,
            feature_min_elfs,
            feature_max_elfs,
            feature_dims,
            feature_parents,
            ) = find_bifurcations(
            connection_pairs,
            connection_elfs,
            basin_maxima_grid,
            basin_maxima_ref_values,
            reference_grid.total,
            neighbor_transforms,
                )
                               
                
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
                feature_min_elfs=feature_min_elfs,
                feature_max_elfs=feature_max_elfs,
                feature_dims=feature_dims,
                atom_grid_coords=atom_grid_coords,
                neighbor_transforms=neighbor_transforms,
                basin_labels=bader.basin_labels,
                data=reference_grid.total,
                num_basins=len(bader.basin_maxima_frac),
                )

        #######################################################################
        # Construct Graph
        #######################################################################    
        graph = cls(bader=bader, **kwargs)
        for feat_idx in range(len(feature_basins)):
            if len(feature_basins[feat_idx]) > 1:
                # This is a reducible feature
                node = ReducibleNode(
                    bifurcation_graph=graph,
                    basins=feature_basins[feat_idx], 
                    dimensionality=feature_dims[feat_idx], 
                    contained_atoms=feature_atoms[feat_idx], 
                    min_value=feature_min_elfs[feat_idx], 
                    max_value=feature_max_elfs[feat_idx],
                    )
            
            elif len(feature_basins[feat_idx]) == 1:
                # this is an irreducible feature. 
                node = IrreducibleNode(
                    bifurcation_graph=graph,
                    basins=feature_basins[feat_idx], 
                    dimensionality=feature_dims[feat_idx], 
                    contained_atoms=feature_atoms[feat_idx], 
                    min_value=feature_min_elfs[feat_idx], 
                    max_value=feature_max_elfs[feat_idx],
                    )
                # Get additional attributes
                basin_idx = node.basins[0]
                node.frac_coords = bader.basin_maxima_frac[basin_idx]
                node.charge = bader.basin_charges[basin_idx]
                node.volume = bader.basin_volumes[basin_idx]
                node.atom_distance = bader.basin_atom_dists[basin_idx]
                node.nearest_atom = bader.basin_atoms[basin_idx]
                node.nearest_atom_type = bader.structure[node.nearest_atom].specie.symbol
                
        # TODO: Combine ring/spheres to singular basins. Calculate coord envs
        # for irreducible features?
        # Move plot method to here. Add get_plot_label methods to nodes?
        
        return graph
        
        
        