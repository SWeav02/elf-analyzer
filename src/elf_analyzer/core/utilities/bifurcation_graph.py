# -*- coding: utf-8 -*-

from typing import Literal, TypeVar

import numpy as np
from numpy.typing import NDArray
import json

Node = TypeVar("Node", bound="Node")
# Edge = TypeVar("Edge", bound="Edge")
BifurcationGraph = TypeVar("BifurcationGraph", bound="BifurcationGraph")

# TODO: Make a to/from json

class Node:
    
    def __init__(
            self, 
            bifurcation_graph: BifurcationGraph,
            key: int, 
            parent: Node | int | None = None,
            **kwargs,
            ):
        # set variables the user can freely change
        # self.attributes = attributes
        # set variables that result in updates to other properties when changed
        self._bifurcation_graph = bifurcation_graph
        self._key = key
        self._children = []
        # convert integer parents to the corresponding Node object
        if type(parent) == int:
            parent = bifurcation_graph[parent]
        
        # check if our parent is None. If so, we are trying to set a root node
        if parent is None:
            assert bifurcation_graph._root_node is None, "Only one root node allowed per graph. Additional nodes must provide a parent."
            bifurcation_graph._root_node = self
        else:
            parent._children.append(self)
        
        # set parent
        self._parent = parent
        
        # set any other provided attributes
        for key , value in kwargs.items():
            setattr(self, key, value)
        
    def __eq__(
        self,
        other: Node,
            ):
        assert type(other) == Node, "Both objects must by Node types to make equality check"
        assert self.bifurcation_graph is other.bifurcation_graph, "Nodes must belong to the same BifurcationGraph to make equality check"
        
        # node indices are unique. Return the comparison between the two
        return self.key == other.key
    
    @property
    def bifurcation_graph(self) -> BifurcationGraph:
        return self._bifurcation_graph
    
    @property
    def key(self) -> int:
        return self._key
    
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
    def children(self) -> list[Node]:
        return self._children
    
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
    
class BifurcationGraph:
    """
    A class for storing the nodes of a bifurcation graph. The nodes themselves
    contain the information on their connectivity.
    """
    
    def __init__(self):
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
    
    