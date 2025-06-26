# -*- coding: utf-8 -*-

from networkx import DiGraph
import numpy as np
from numpy.typing import NDArray
import networkx
import json
    
class BifurcationGraph(DiGraph):
    """
    This is an expansion of networkx's Graph class specifically with
    additional methods related to bifurcation plots.
    """

    def parent_index(self, n: int) -> int:
        """
        Returns the node index for the parent of the provided node index
        """
        predecessor_list = list(self.predecessors(n))
        if len(predecessor_list) > 0:
            return predecessor_list[0]
        else:
            return None

    def parent_dict(self, n: int) -> dict:
        """
        Returns the dictionary of attributes assigned to the parent of
        the provided node index
        """
        parent_index = self.parent_index(n)
        if parent_index is not None:
            return self.nodes[parent_index]
        else:
            return None

    def deep_parent_indices(self, n: int) -> NDArray[np.int64]:
        """
        Returns the indices of all nodes connected to this node by
        parents.
        """
        predecessor_list = []

        current_predecessor = n
        while current_predecessor is not None:
            current_predecessor = self.parent_index(current_predecessor)
            if current_predecessor is not None:
                predecessor_list.append(current_predecessor)
        return predecessor_list

    def child_indices(self, n: int) -> NDArray[np.int64]:
        """
        Returns the indices of the children of this node if they exist
        """
        child_indices_list = list(self.successors(n))
        return np.array(child_indices_list)

    def child_dicts(self, n: int) -> dict:
        """
        Returns the dictionaries of attributes assigned to the children of
        the provided node index. Returns a nested dict with child indices
        as keys and dicts as values.
        """
        children = {}
        for i in self.child_indices(n):
            children[i] = self.nodes[i]
        return children

    def deep_child_indices(self, n: int) -> NDArray[np.int64]:
        """
        Returns the indices of all subsequent nodes after this node.
        """
        all_found = False
        child_indices = self.child_indices(n)
        while not all_found:
            new_child_indices = child_indices.copy()
            for i in child_indices:
                new_child_indices = np.concatenate(
                    [new_child_indices, self.child_indices(i)]
                )
            new_child_indices = np.unique(new_child_indices)
            if len(child_indices) == len(new_child_indices):
                all_found = True
            child_indices = new_child_indices
        return child_indices.astype(int)

    def deep_child_dicts(self, n: int) -> dict:
        """
        Returns the dictionaries of attributes assigned subsequent nodes
        after this node. Returns a nested dict with child indices
        as keys and dicts as values.
        """
        children = {}
        for i in self.deep_child_indices(n):
            children[i] = self.nodes[i]
        return children

    def sibling_indices(self, n: int) -> NDArray[np.int64]:
        """
        Returns the indices of the siblings of this node if they exist
        """
        parent_idx = self.parent_index(n)
        if parent_idx is None:
            return
        siblings = self.child_indices(parent_idx)
        # remove self
        siblings = siblings[siblings != n]
        return siblings

    def sibling_dicts(self, n: int) -> dict:
        """
        Returns the dictionaries of attributes assigned to the siblings of
        the provided node index. Returns a nested dict with child indices
        as keys and dicts as values.
        """
        siblings = {}
        for i in self.sibling_indices(n):
            siblings[i] = self.nodes[i]
        return siblings

    def to_dict(self) -> dict:
        """
        Converts graph into two dicts for the nodes and edges
        """
        graph_dict = {}
        node_dict = {}
        for node in self.nodes:
            node_dict[node] = self.nodes[node]
        edge_list = [edge for edge in self.edges]
        graph_dict["nodes"] = node_dict
        graph_dict["edges"] = edge_list
        return graph_dict

    @classmethod
    def from_dict(cls, graph_dict: dict):
        """
        Converts from a dict to a bifurcation graph
        """
        new_graph = BifurcationGraph()

        for node_idx in graph_dict["nodes"].keys():
            new_graph.add_node(node_idx)

        networkx.set_node_attributes(new_graph, graph_dict["nodes"])

        for edge0, edge1 in graph_dict["edges"]:
            new_graph.add_edge(edge0, edge1)

        return new_graph

    def to_json(self):
        """
        Converts graph to a jsonable object
        """
        graph_dict = self.to_dict()
        # convert all numpy objects to python
        for node, attributes in graph_dict["nodes"].items():
            for key, attribute in attributes.items():
                if isinstance(attribute, np.integer):
                    attributes[key] = int(attribute)
                if isinstance(attribute, np.floating):
                    attributes[key] = float(attribute)
                if isinstance(attribute, np.ndarray) or isinstance(attribute, list):
                    new_attribute = list(attribute)
                    for i, value in enumerate(new_attribute):
                        if isinstance(value, np.integer):
                            new_attribute[i] = int(value)
                        if isinstance(value, np.floating):
                            new_attribute[i] = float(value)
                    attributes[key] = new_attribute

        cleaned_edges = []
        for edge in graph_dict["edges"]:
            new_edge = [int(edge[0]), int(edge[1])]
            cleaned_edges.append(new_edge)

        graph_dict["edges"] = cleaned_edges
        graph_json = json.dumps(graph_dict)
        return graph_json

    @classmethod
    def from_json_string(cls, graph_string: str):
        """
        Converts from a json string to a BifurcationGraph
        """
        graph_dict = json.loads(graph_string)
        new_graph = cls.from_dict(graph_dict)
        return new_graph