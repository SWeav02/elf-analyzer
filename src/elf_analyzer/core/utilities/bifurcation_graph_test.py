# -*- coding: utf-8 -*-

from abc import ABC, abstractproperty, abstractmethod
from typing import Literal, TypeVar

import numpy as np
from numpy.typing import NDArray
import json

from baderkit.core import Bader

Node = TypeVar("Node", bound="Node")
# Edge = TypeVar("Edge", bound="Edge")
BifurcationGraph = TypeVar("BifurcationGraph", bound="BifurcationGraph")

#TODO: Move graph specific methods to these classes. For example, create the
# graph object from the Bader object here rather than calling a function in
# the ElfAnalyzer class. This should let me make a ChargeAnalyzer class or
# something similar later on.
# Also move the plotting method here as well. And split classes to separate
# files for cleanliness
# Make most of the graph immutable. 
# Maybe combine ring/sphere basins here if possible

def serialize_numpy(value):
    """Convert numpy types into native Python types for JSON serialization."""
    if isinstance(value, np.ndarray):
        return value.tolist()  # arrays → Python lists
    elif isinstance(value, (np.integer,)):
        return int(value)      # numpy int → Python int
    elif isinstance(value, (np.floating,)):
        return float(value)    # numpy float → Python float
    elif isinstance(value, (np.bool_)):
        return bool(value)     # numpy bool → Python bool
    else:
        return value           # leave everything else as is

def list_to_numpy(value):
    """Convert lists of ints/floats to numpy for reading from JSON"""
    if isinstance(value, list):
        if len(value) > 0:
            first_value = value[0]
            if isinstance(first_value, int) or isinstance(first_value, np.integer):
                return np.array(value, dtype=np.int64)
            elif isinstance(first_value, float) or isinstance(first_value, np.floating):
                return np.array(value, dtype=np.float64)
    return value
        
class Node(ABC):
    
    def __init__(
            self, 
            bifurcation_graph: BifurcationGraph,
            key: int, 
            basins: NDArray[int],
            dimensionality: int,
            min_value: float,
            parent: Node | int | None = None,
            **kwargs,
            ):
        # set variables the user can freely change
        self.min_value = min_value
        self.max_value = None
        self.basins = basins
        self.dimensionality = dimensionality
        self.contained_atoms = []
        # set variables that result in updates to other properties when changed
        self._bifurcation_graph = bifurcation_graph
        self._key = key
        # convert integer parents to the corresponding Node object
        if type(parent) == int:
            parent = bifurcation_graph.node_from_key(parent)
        
        # check if our parent is None. If so, we are trying to set a root node
        if parent is None:
            assert bifurcation_graph._root_node is None, "Only one root node allowed per graph. Additional nodes must provide a parent."
            bifurcation_graph._root_node = self
        else:
            parent._children.append(self)
        
        # set parent
        self._parent = parent
        
    def __eq__(
        self,
        other: Node,
            ):
        assert type(other) == Node, "Both objects must by Node types to make equality check"
        assert self.bifurcation_graph is other.bifurcation_graph, "Nodes must belong to the same BifurcationGraph to make equality check"
        
        # node indices are unique. Return the comparison between the two
        return self.key == other.key
    
    ###########################################################################
    # Graph Related Properties
    ###########################################################################
    
    @property
    def bifurcation_graph(self) -> BifurcationGraph:
        return self._bifurcation_graph
    
    @property
    def key(self) -> int:
        return self._key
    
    @abstractproperty
    def is_reducible(self):
        pass
    
    @property
    def parent(self) -> Node:
        return self._parent
    
    @parent.setter
    def parent(self, new_parent: int | Node):
        # make sure this isn't the parent node
        assert self._parent is not None, "Root node cannot be assigned a new parent"
        if type(new_parent) == int:
            new_parent = self.bifurcation_graph.get(new_parent, None)
        # remove this node from the current parent's children
        self._parent._children = [i for i in self._parent._children if i is not self]
        # update this node's parent
        self._parent = new_parent
        # add this node to the new parent's children
        new_parent._children.append(self)
    
    @property
    def siblings(self) -> list[Node]:
        if self.parent is not None:
            return [i for i in self.parent.children if i is not self]
        else:
            return []
        
    @property
    def ancestors(self) -> list[Node]:
        """

        Returns
        -------
        list[Node]
            The parent, grandparent, great grandparent, etc. of this node.

        """
        all_parents = []
        current_parent = self.parent
        while current_parent is not None:
            all_parents.append(current_parent)
            current_parent = current_parent.parent
        return all_parents    
    
    @abstractmethod
    def remove(self) -> None:
        pass
        
    ###########################################################################
    # Feature Related Properties
    ###########################################################################
    
    @property
    def depth(self):
        assert self.max_value is not None, "Max value must be set to calculate depth"
        return self.max_value - self.min_value
    
    @property
    def depth_to_infinite(self):
        if self.is_infinite:
            return 0.0 # this node itself is infinite

        # loop from most recent ancestor to oldest
        # NOTE: There will always be at least one ancestor that is infinite, because
        # the root must be infinite
        for ancestor in self.ancestors:
            if ancestor.is_infinite:
                break
        return self.max_value - ancestor.max_value
    
    @property
    def is_infinite(self):
        return self.dimensionality > 0
    
    @property
    def basin_mask(self):
        basin_labels = self.bifurcation_graph.bader.basin_labels
        return np.isin(basin_labels, self.basins)
    
    @property
    def feature_mask(self):
        data = self.bifurcation_graph.bader.reference_grid.total
        return self.basin_mask & (data >= self.min_value)
    
        
class ReducibleNode(Node):
    def __init__(self):
        super().__init__()
        self._children = []
    
    @property
    def is_reducible(self):
        return True
    
    @property
    def children(self) -> list[Node]:
        return self._children
    
    @property
    def deep_children(self) -> list[Node]:
        all_children = []
        current_children = self.children
        while True:
            new_children = []
            for child in current_children:
                new_children.extend(child.children)
                all_children.append(child)
            current_children = new_children
            if len(current_children) == 0:
                break
        return all_children
    
    def remove(self) -> None: 
        # if this is the root, try to assigning child as the new root.
        if self.parent is None:
            assert len(self.children) == 1, "Root can only be deleted if it has a single child"
            self.bifurcation_graph._root_node = self.children[0].index
            self.children[0]._parent = None
        else:
            # assign all children to parent
            for child in self.children:
                child.parent = self.parent
            # remove this node from the current parent's children
            self.parent._children = [i for i in self.parent._children if i is not self]
        # delete this node
        graph = self.bifurcation_graph
        graph._nodes = [i for i in graph._nodes if i is not self]
        del(graph._node_keys[self.key])

class IrreducibleNode(Node):
    def __init__(self):
        super().__init__()
        self.basin_type = None
        self.basin_subtype = None
    
    @property
    def is_reducible(self):
        return False
    
    @property
    def _basin_idx(self):
        assert len(self.basins) == 1, "This irreducible node has more than 1 basin. This is a BUG!!!"
        return self.basins[0]
    
    @property
    def frac_coords(self):
        return self.bifurcation_graph.bader.basin_maxima_frac[self._basin_idx]
    
    @property
    def charge(self):
        return self.bifurcation_graph.bader.basin_charges[self._basin_idx]
    
    @property
    def volume(self):
        return self.bifurcation_graph.bader.basin_volumes[self._basin_idx]
    
    @property
    def atom_distance(self):
        return self.bifurcation_graph.bader.basin_atom_dists[self._basin_idx]
    
    @property
    def nearest_atom(self):
        return self.bifurcation_graph.bader.basin_atoms[self._basin_idx]
    
    @property
    def nearest_atom_type(self):
        return self.bifurcation_graph.bader.structure[self.nearest_atom].specie.symbol
    
    def remove(self) -> None: 
        # remove this node from the current parent's children
        self.parent._children = [i for i in self.parent._children if i is not self]
        # delete this node
        graph = self.bifurcation_graph
        graph._nodes = [i for i in graph._nodes if i is not self]
        del(graph._node_keys[self.key])
    
class ReducibleShellNode(IrreducibleNode):
    
    @property
    def is_reducible(self):
        return True
    
    @property
    def _basin_idx(self):
        distances = self.bifurcation_graph.bader.basin_atom_dists[self.basins]
        min_dist = distances.min()
        min_idx = np.argmax(distances==min_dist)
        return min_idx
    
    @property
    def charge(self):
        charges = self.bifurcation_graph.bader.basin_charges[self.basins]
        return charges.sum()
    
    @property
    def volume(self):
        volumes = self.bifurcation_graph.bader.basin_volumes[self.basins]
        return volumes.sum()
        
    
class BifurcationGraph:
    """
    A class for storing the nodes of a bifurcation graph. The nodes themselves
    contain the information on their connectivity.
    """
    
    def __init__(
            self,
            bader: Bader,
            ):
        self._bader = bader
        self._root_node = None
        self._nodes = []
        self._node_keys = {}

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
    def bader(self) -> Bader:
        return self._bader
    
    @property
    def root_node(self) -> Node:
        return self._root_node
    
    @property
    def nodes(self) -> list[Node]:
        return self._nodes
    
    @nodes.setter
    def nodes(self, new_nodes):
        raise Exception(
            "Nodes cannot be set directly. Use the `add_node` or `remove_node` methods instead."
            )
        
    def node_from_key(self, key):
        return self._node_keys[key]
    
    def add_node(self, **kwargs) -> Node:
        # get an index for this node if there isn't one already
        key = kwargs.get("index", None)
        if key is None:
            if len(self) > 0:
                key = max(self._node_keys.keys()) + 1
            else:
                key = 0
        assert type(key) == int, "Index must be an integer value"
        assert key not in self._node_keys.keys(), "Index must not already exist"
        # create the new node
        node = Node(key=key, bifurcation_graph=self, **kwargs)
        # add the node to the graph
        self._nodes.append(node)
        self._node_keys[key] = node
        # return the node
        return node
        
    def update_node_attributes(self, key: int, **kwargs):
        assert type(key) == int, "Index must be an integer"
        node = self._node_keys[key]
        for key, value in kwargs.items():
            setattr(node, key, value)
    
    def remove_node(self, node: int | Node):
        # convert int to Node item
        if type(node) == int:
            node = self._node_keys[node]
        node.remove()
    
    def to_dict(self) -> dict:
        # NOTE: This could just be a to list method, but I may add other meta
        # data down the line
        # create our initial dict
        graph_dict = {
            "nodes": [],
            }
        for node in self:
            # create a dict to store node properties
            parent = node.parent
            if parent:
                parent_key = parent.key
            else:
                parent_key = None
            node_dict = {
                "parent": parent_key,
                "key": node.key,
                }
            # add all properties
            node_props = {}
            for attr_name, value in node.__dict__.items():
                # skip hidden variables
                if attr_name[0] == "_":
                    continue
                node_props[attr_name] = serialize_numpy(list_to_numpy(value))
            node_dict["properties"] = node_props
            # add node
            graph_dict["nodes"].append(node_dict)
        return graph_dict
    
    @classmethod
    def from_dict(cls, graph_dict: dict):
        nodes = graph_dict["nodes"]
        
        # create our initial graph object
        graph = cls()
        
        # add our nodes
        for node_dict in nodes:
            key = node_dict["key"]
            parent = node_dict["parent"]
            properties = node_dict["properties"]
            # create node
            node = graph.add_node(index=key, parent=parent)
            # add attributes
            for attr_name, value in properties.items():
                setattr(node, attr_name, list_to_numpy(value))
        return graph

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str):
        graph_dict = json.loads(json_str)
        return cls.from_dict(graph_dict)