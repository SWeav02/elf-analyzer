# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Literal, TypeVar, Dict, Type

import numpy as np
from numpy.typing import NDArray
from pymatgen.analysis.local_env import CrystalNN

from elf_analyzer.core.bifurcation_graph.feature_mappings import FeatureSubtype

Node = TypeVar("Node", bound="NodeBase")    

class NodeBase(ABC):
    
    _registry: Dict[str, Type["NodeBase"]] = {}
    feature_type = None
    is_reducible = False
    
    label_map = {
        "type" : "feature_subtype",
        "basins" : "basins",
        "dimensionality" : "dimensionality",
        "contained atoms" : "contained_atoms",
        "min value" : "min_value",
        "max value" : "max_value",
        "depth" : "depth",
        }
    
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
            feature_subtype: str = None,
            ):
        
        # set properties that all nodes have
        self.bifurcation_graph = bifurcation_graph
        self.basins = np.array(basins)
        self.dimensionality = int(dimensionality)
        self.contained_atoms = np.array(contained_atoms)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self._feature_subtype = feature_subtype
        

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
    def feature_subtype(self) -> FeatureSubtype:
        return self._feature_subtype
    
    @feature_subtype.setter
    def feature_subtype(self, value: FeatureSubtype | str):
        if not value in FeatureSubtype.subtypes[self.feature_type]:
            raise ValueError(f"Invalid feature subtype for {self.feature_type} node. Options are {[i for i in FeatureSubtype.subtypes[self.feature_type]]}")
        self._feature_subtype = FeatureSubtype(value)
    
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
        return self.max_value - ancestor.max_value

    
    @property
    def is_infinite(self):
        return self.dimensionality > 0
    
    @property
    def _bader(self):
        return self.bifurcation_graph.labeler
    
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
    @property
    def plot_label(self) -> None:
        lines = []
        for tag, attr in self.label_map.items():
            value = getattr(self, attr, None)
            if value is None:
                continue
            if tag == "type":
                lines.append(f"{tag}: {value.value}".title())
                continue
            if isinstance(value, (float, np.floating)):
                value = round(value, 4)
            lines.append(f"{tag}: {value}".title())
    
        label = "<br>".join(lines)
        return label
    
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
        "feature_type": self.feature_type,
        "feature_subtype": self.feature_subtype,
            }
    
    @classmethod
    def from_dict(cls, bifurcation_graph, node_dict: dict) -> Node:
        # automatic node creation for all inheriting Nodes
        node_type = node_dict.pop("feature_type")
        subclass = cls._registry[node_type]
        return subclass(bifurcation_graph=bifurcation_graph, **node_dict)
    

class ReducibleNode(NodeBase):
    
    feature_type = "ReducibleNode"
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
            )
        node.feature_subtype = "shallow"
        return node

class IrreducibleNode(NodeBase):
    
    is_reducible = False
    feature_type = "IrreducibleNode"
    
    label_map = NodeBase.label_map
    label_map.update(
        {
        "charge" : "charge",
        "volume" : "volume",
        "depth to infinite feature" : "depth_to_infinite",
        "atom distance" : "atom_distance",
        "nearest atom index" : "nearest_atom",
        "nearest atom species" : "nearest_atom_species",
        "minimum surface dist" : "min_surface_dist",
        "average surface dist" : "avg_surface_dist",
        "distance beyond atom" : "dist_beyond_atom",
        "coord number" : "coord_num",
        "coord atom indices" : "coord_atom_indices",
        "coord atom species" : "coord_atom_species"
            }
        )
    
    def __init__(
            self,
            frac_coords: NDArray[float],
            charge: float,
            volume: float,
            nearest_atom: int,
            coord_atom_indices: list[int] = None,
            **kwargs,
        ):
        super().__init__(**kwargs)
        
        self.frac_coords = np.array(frac_coords, dtype=np.float64)
        self.charge = float(charge)
        self.volume = float(volume)
        self.nearest_atom = int(nearest_atom)
        
        self._min_surface_dist = None
        self._avg_surface_dist = None
        self._coord_atom_indices = coord_atom_indices
        
    @property
    def nearest_atom_species(self) -> str:
        return self.bifurcation_graph.structure[self.nearest_atom].species_string
    
    def _calc_atom_dist(self, atom_idx):
        structure = self.bifurcation_graph.structure
        atom_frac_coords = structure[atom_idx].frac_coords
        lattice = structure.lattice
        dist, _ = lattice.get_distance_and_image(atom_frac_coords, self.frac_coords)
        return dist
    
    @property
    def atom_distance(self) -> float:
        return self._calc_atom_dist(self.nearest_atom)
    
    @property
    def dist_beyond_atom(self) -> float:
        if self.bifurcation_graph.atomic_radii is not None:
            radius = self.bifurcation_graph.atomic_radii[self.nearest_atom]
            return self.atom_distance - radius
        
    @property
    def min_surface_dist(self) -> float:
        if self._min_surface_dist is None:
            self.bifurcation_graph._calculate_feature_surface_dists()
        return self._min_surface_dist
    
    @property
    def avg_surface_dist(self) -> float:
        if self._avg_surface_dist is None:
            self.bifurcation_graph._calculate_feature_surface_dists()
        return self._avg_surface_dist
    
    @property
    def coord_num(self) -> int:
        return len(self.coord_atom_indices)
    
    @property
    def coord_atom_indices(self) -> list[int]:
        if self._coord_atom_indices is None:
            if self.feature_subtype in ["core", "shell", "deep shell"]:
                self._coord_atom_indices = [self.nearest_atom]
            else:
                # TODO: I would really like a better method of doing this as
                # CrystalNN is very slow in this situation
                feature_structure = self.bifurcation_graph.structure.copy()
                feature_structure.append("H-", self.frac_coords)
                cnn = CrystalNN(distance_cutoffs=None)
                coordination = cnn.get_nn_info(feature_structure, -1)
                self._coord_atom_indices = [int(i["site_index"]) for i in coordination]
        
        return self._coord_atom_indices
    
    @coord_atom_indices.setter
    def coord_atom_indices(self, value: list[int]):
        try:
            value = [int(i) for i in value]
        except:
            raise TypeError("Atom indices must be a list of integers")
        
        self._coord_atom_indices = value
    
    @property
    def coord_atom_species(self) -> list[str]:
        structure = self.bifurcation_graph.structure
        return [structure[i].species_string for i in self.coord_atom_indices]
    
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
            "charge",
            "volume",
            "nearest_atom",
            "nearest_atom_species",
            "atom_distance",
                ]:
            node_dict[attr] = getattr(self, attr)
        return node_dict