# -*- coding: utf-8 -*-

from abc import ABC, abstractproperty, abstractmethod
from typing import Literal, TypeVar, Dict, Type
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
import json

from baderkit.core import Bader

Node = TypeVar("Node", bound="NodeBase")    

class NodeBase(ABC):
    
    _registry: Dict[str, Type["NodeBase"]] = {}
    node_type = None
    feature_type = None
    is_reducible = False
    
    def __init__(
            self, 
            bifurcation_graph,
            basins: NDArray[int],
            dimensionality: int,
            contained_atoms: list[int],
            min_value: float,
            max_value: float,
            key: int | None = None,
            parent: Node | int | None = None,
            ):
        
        # set properties that all nodes have
        self.bifurcation_graph = bifurcation_graph
        self.basins = np.array(basins)
        self.dimensionality = int(dimensionality)
        self.contained_atoms = np.array(contained_atoms)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        

        # convert integer parents to the corresponding Node object
        if type(parent) == int:
            parent = bifurcation_graph.node_from_key(parent)
        
        # check if our parent is None. If so, we add a new root node
        if parent is None:
            bifurcation_graph._root_nodes.append(self)
        else:
            parent._children.append(self)
        
        # set parent
        self._parent = parent
        
        # create a key for this node
        if key is None:
            if len(bifurcation_graph) > 0:
                key = max(bifurcation_graph._node_keys.keys()) + 1
            else:
                key = 0
        self.key = key

        # add this node to the corresponding graph
        bifurcation_graph._nodes.append(self)
        bifurcation_graph._node_keys[key] = self
        
    def __init_subclass__(cls, **kwargs):
        # automatically registers subclasses. Used for convenient from_dict method
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls  # Register subclass automatically
        
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
        
    ###########################################################################
    # Feature Related Properties
    ###########################################################################
    
    @property
    def depth(self):
        return self.max_value - self.min_value
    
    @property
    def depth_to_infinite(self):
        if self.is_infinite:
            return 0.0 # this node itself is infinite

        # loop from most recent ancestor to oldest and find the first that is infinite
        for ancestor in self.ancestors:
            if ancestor.is_infinite:
                break
        try:
            return self.max_value - ancestor.max_value
        except:
            breakpoint()
    
    @property
    def is_infinite(self):
        return self.dimensionality > 0
    
    @property
    def _bader(self):
        return self.bifurcation_graph.bader
    
    @property
    def basin_mask(self):
        assert self._bader is not None, "Masks can only be generated for graphs connected to Bader objects"
        # We don't cache this as if it was called for many nodes in a large system
        # there could be some pretty major issues
        basin_labels = self._bader.basin_labels
        return np.isin(basin_labels, self.basins)
    
    @property
    def feature_mask(self):
        assert self._bader is not None, "Masks can only be generated for graphs connected to Bader objects"
        data = self._bader.reference_grid.total
        return self.basin_mask & (data >= self.min_value)
    
    ###########################################################################
    # Properties that must be set by children or require additional steps
    ###########################################################################
    @abstractproperty
    def plot_label(self) -> None:
        pass
    
    @abstractmethod
    def remove(self) -> None:
        pass
    
    def to_dict(self) -> dict:

        if self.parent is None:
            parent_key = None
        else:
            parent_key = self.parent.key
        
        return {
        "key": self.key,
        "basins": [int(i) for i in self.basins], # convert to python for json serialize
        "dimensionality": int(self.dimensionality),
        "contained_atoms": [int(i) for i in self.contained_atoms],
        "min_value": float(self.min_value),
        "max_value": float(self.max_value),
        "parent": parent_key,
        "node_type": self.node_type,
        "feature_type": self.feature_type,
            }
    
    @classmethod
    def from_dict(cls, bifurcation_graph, node_dict: dict) -> Node:
        # automatic node creation for all inheriting Nodes
        node_type = node_dict.pop("node_type")
        subclass = cls._registry[node_type]
        return subclass(bifurcation_graph=bifurcation_graph, **node_dict)
    

class ReducibleNode(NodeBase):
    
    node_type = "ReducibleNode"
    feature_type = "reducible"
    is_reducible = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._children = []
    
    @property
    def children(self):
        return self._children
    
    @property
    def deep_children(self):
        all_children = []
        current_children = self.children
        while True:
            new_children = []
            for child in current_children:
                all_children.append(child)
                if not child.is_reducible:
                    continue
                new_children.extend(child.children)
                
            current_children = new_children
            if len(current_children) == 0:
                break
        return all_children
    
    def remove(self) -> None: 
        # if this is a root, each child will be a new root
        if self.parent is None:
            # remove this node as a root
            self.bifurcation_graph._root_nodes = [i for i in self.bifurcation_graph._root_nodes if i is not self]
            # add each child as a root
            for child in self.children:
                self.bifurcation_graph._root_nodes.append(child)
                child._parent = None
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
        
    def make_irreducible(self):
        # we want to combine all of the irreducible nodes in this node into one
        # then delete all children.
        frac_coords = None
        charge = 0
        volume = 0
        nearest_atom = None
        nearest_atom_type = None
        atom_distance = 1e300
        max_value = -1e300
        for child in self.deep_children:
            if child.is_reducible:
                continue
            charge += child.charge
            volume += child.volume
            if child.atom_distance < atom_distance:
                atom_distance = child.atom_distance
                frac_coords = child.frac_coords
                nearest_atom = child.nearest_atom
                nearest_atom_type = child.nearest_atom_type
            if child.max_value > max_value:
                max_value = child.max_value
        # delete all nodes below this one
        nodes = self.deep_children.copy()
        nodes.reverse()
        for node in nodes:
            node.remove()
        # delete self
        self.remove()
        
        # create a new irreducible node connected to parent
        node = IrreducibleNode(
            key=self.key,
            bifurcation_graph=self.bifurcation_graph,
            basins=self.basins,
            dimensionality=self.dimensionality,
            contained_atoms=self.contained_atoms,
            min_value=self.min_value,
            max_value=max_value,
            parent=self.parent,
            frac_coords=frac_coords, 
            charge=charge, 
            volume=volume, 
            nearest_atom=nearest_atom, 
            nearest_atom_type=nearest_atom_type, 
            atom_distance=atom_distance,
            feature_type="shallow"
            )
        node.feature_type = "shallow"
        
    @property
    def plot_label(self) -> str:
        lines = [
            f"type: {self.feature_type}",
            f"max value: {round(self.max_value, 4)}",
            f"min value: {round(self.min_value, 4)}",
            f"depth: {round(self.depth, 4)}",
            f"contained atoms: {self.contained_atoms}",
            f"dimensionality: {self.dimensionality}",
        ]
    
        label = "<br>".join(lines)
        return label



class IrreducibleNode(NodeBase):
    
    is_reducible = False
    node_type = "IrreducibleNode"
    feature_type = "irreducible" # Should get updated by analyzer eventually
    
    def __init__(
            self,
            frac_coords: NDArray[float],
            charge: float,
            volume: float,
            nearest_atom: int,
            nearest_atom_type: str,
            atom_distance: float,
            feature_type: str | None = "irreducible",
            # irreducible_type: str | None = "point",
            **kwargs,
        ):
        super().__init__(**kwargs)
        
        # set each instance variable with typing
        self.feature_type = feature_type
        self.frac_coords = np.array(frac_coords, dtype=np.float64)
        # self.irreducible_type = irreducible_type
        self.nearest_atom_type = nearest_atom_type

        self.charge = float(charge)
        self.volume = float(volume)
        self.nearest_atom = int(nearest_atom)
        self.atom_distance = float(atom_distance)
        
    @property
    def plot_label(self) -> str:
        lines = [
            f"type: {self.feature_type}",
            f"depth: {round(self.depth, 4)}",
            f"depth to infinite feature: {round(self.depth_to_infinite, 4)}",
            f"max value: {round(self.max_value, 4)}",
            f"min value: {round(self.min_value, 4)}",
            f"charge: {round(self.charge, 4)}",
            f"volume: {round(self.volume, 4)}",
            f"atom distance: {round(self.atom_distance, 4)}",
            f"nearest atom index: {self.nearest_atom}",
            f"nearest atom type: {self.nearest_atom_type}",
        ]
    
        label = "<br>".join(lines)
        return label

    # TODO: Add information calculated later in the process. How should I treat
    # these variables?
        # if getattr(self, "bare_electron_indicator", None) is not None:
        #     label += f'\nfeature radius: {round(self.feature_radius, 4)}'
        #     label += f'\ndistance beyond atom: {round(self.dist_beyond_atom, 4)}'
        #     label += f'\ncoord number: {round(self.coord_num, 4)}'
        #     label += f'\ncoord atoms: {self.coord_atoms}'
        #     label += f"\nBEI array: {self.bare_electron_scores.round(4)}"
    
    def remove(self) -> None: 
        # remove this node from the current parent's children
        self.parent._children = [i for i in self.parent._children if i is not self]
        # delete this node
        graph = self.bifurcation_graph
        graph._nodes = [i for i in graph._nodes if i is not self]
        del(graph._node_keys[self.key])
        
    def to_dict(self) -> dict:
        node_dict = super().to_dict()
        # convert some attributes to serializable versions
        node_dict["frac_coords"] = [float(i) for i in self.frac_coords]
        # add other attributes
        for attr in [
            "feature_type",
            "charge",
            "volume",
            "nearest_atom",
            "nearest_atom_type",
            "atom_distance",
                ]:
            node_dict[attr] = getattr(self, attr)
        return node_dict