# -*- coding: utf-8 -*-

"""
Contains the main class for running elf topology analysis.
"""

# -*- coding: utf-8 -*-

import logging
import math
from functools import cached_property
from pathlib import Path
import itertools
from tqdm import tqdm

import networkx
import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure

from baderkit.core import Grid, Bader

from elf_analyzer.core.utilities import UnionFind, BifurcationGraph, IonicRadiiTools, find_connections


class ElfAnalyzer:
    """
    A class for finding electride sites from an ELFCAR.
    """
    
    _spin_polarized = False

    def __init__(
        self,
        elf_grid: Grid,
        charge_grid: Grid,
        ignore_low_pseudopotentials: bool = False,
        downscale_resolution: int = 200,
        bader_kwargs: dict = {},
        method_arguments: dict = {}
    ):
        self.elf_grid = elf_grid.copy()
        self.charge_grid = charge_grid.copy()
        self.ignore_low_pseudopotentials = ignore_low_pseudopotentials
        self.bader_kwargs = bader_kwargs
        self.method_arguments = method_arguments
        self.downscale_resolution = downscale_resolution
            
        # define properties that will be updated by running the method
        self._bader = None
        self._bifurcations = None
        self._bifurcation_graph = None
        self._bifurcation_plot = None
        self._downscaled_elf_grid = None
        self._downscaled_labels = None
        self._atomic_radii = None
        

    ###########################################################################
    # Main Properties
    ###########################################################################
    
    @property
    def structure(self) -> Structure:
        """
        Shortcut to grid's structure object
        """
        structure = self.elf_grid.structure.copy()
        structure.add_oxidation_state_by_guess()
        return structure
    
    @cached_property
    def labeled_structure(self) -> Structure:
        return self.get_labeled_structure(self.bifurcation_graph)
    
    @property
    def bader(self) -> Bader:
        """
        Returns the Bader object associated with this system
        """
        if self._bader is None:
            self._bader = Bader(
                charge_grid=self.charge_grid, 
                reference_grid=self.elf_grid,
                **self.bader_kwargs,
                )
        return self._bader

    @cached_property
    def atom_coordination_envs(self) -> list:
        """
        Gets the coordination environment for the atoms in the system
        using CrystalNN
        """
        # TODO: Add crystalNN kwargs?
        cnn = CrystalNN(distance_cutoffs=None)
        neighbors = cnn.get_all_nn_info(self.structure)
        return neighbors
    
    @property
    def bifurcations(self) -> dict:
        if self._bifurcations is None:
            self._bifurcations = self._get_bifurcations()
        return self._bifurcations
    
    @property
    def bifurcation_graph(self) -> BifurcationGraph:
        if self._bifurcation_graph is None:
            self._get_bifurcation_graph(
                self.bader, 
                self.elf_grid, 
                self.charge_grid, 
                **self.method_arguments)
        return self._bifurcation_graph
    
    @property
    def bifurcation_plot(self) -> go.Figure:
        if self._bifurcation_plot is None:
            self._get_bifurcation_plot()
        return self._bifurcation_plot
    
    @property
    def downscaled_elf_grid(self) -> Grid:
        if self._downscaled_elf_grid is None:
            self._get_downscale_grids()
        return self._downscaled_elf_grid
    
    @property
    def downscaled_labels(self) -> NDArray[np.int64]:
        if self._downscaled_labels is None:
            self._get_downscale_grids()
        return self._downscaled_labels
    
    @property
    def atomic_radii(self) -> NDArray[np.float64]:
        # TODO: Figure out a way to calculate this on the fly with only covalent/lone-pairs assigned
        return self._atomic_radii
    
    ###########################################################################
    # Core Graph Construction
    ###########################################################################
    
    def _get_bifurcation_graph(
        self,
        bader: Bader,
        elf_grid: Grid,
        charge_grid: Grid,
        shell_depth: float = 0.05,
        combine_shells: bool = True,
        min_covalent_charge: float = 0.6,
        min_covalent_angle: float = 135,
        min_covalent_bond_ratio: float = 0.4,
        radius_refine_method: str = "cubic",
        electride_elf_min: float = 0.5,
        electride_depth_min: float = 0.2,
        electride_charge_min: float = 0.5,
        electride_volume_min: float = 10,
        electride_radius_min: float = 0.3,
        **kwargs,
    ):
        """
        This will construct a BifurcationGraph class.
        Each node will contain information on whether it is
        reducible/irreducible, atomic/valent, etc.

        This method is largely meant to be called through the get_bifurcation_graphs
        method.
        """
        # get an initial graph connecting bifurcations and final basins
        logging.info(
            "Generating initial bifurcation graph."
            )
        self._initialize_bifurcation_graph()
        
        logging.info("Labeling elf domains")
        # assign node properties
        # TODO: Move a lot of this to the actual Node class (e.g. depth, 3d depth, charge etc)
        self._assign_node_properties()
        
        # First, we clean up the graph in case we removed a node earlier due
        # to incorrect labeling and this resulted in a fake split (e.g. Dy2C)
        self._clean_reducible_nodes()
        
        # Now we have a graph with information associated with each basin. We want
        # to label each node.
        self._mark_atomic(shell_depth)
        
        # Now we want to label our valence features as Covalent, Metallic, or bare electron.
        # Many covalent and metallic features are easy to find. Covalent bonds
        # are typically exactly along a bond between an atom and its nearest
        # neighbors. Metallic features have a low depth. We mark these first
        self._mark_covalent_lonepair(
            min_covalent_charge=min_covalent_charge,
            min_covalent_angle=min_covalent_angle,
            min_covalent_bond_ratio=min_covalent_bond_ratio,
        )
        
        # Now that we have a sense of which features are covalent/lone-pairs
        # we want to correct for a few possible errors in our assignments.
        # Sometimes if we've set our shell depth too low we will end up with only
        # "lone-pairs" surrounding an atom. We relabel these as shells.
        self._correct_for_high_depth_shells()
        
        # Sometimes a metallic/bare electron will detatch from an atomic basin
        # rather than a valence domain. These will be misassigned as shells. We
        # correct for these here by looking for shell basins outside the atoms
        # radius. We need to run a new bader with the labeled structure for this
        # so we also take advantage of the moment to assign radii to the features
        # Note we don't use downscaled grid here
        self._get_atomic_radii(radius_refine_method=radius_refine_method)
        
        self._correct_far_shell_features()
        
        # Reduce any related shell basins to a single basin for clarity
        if combine_shells:
            self._reduce_atomic_shells()
        breakpoint()
        # Now we want to mark the radius of each feature. We don't use the
        # downscaled grid here to get the best chance at a reasonable radius
        graph = self._mark_feature_radii(
            graph=graph,
            bader=bader,
            )
        
        # Now we calculate a bare electron indicator for each valence basin. This
        # is used just to give a sense of how bare an electron is vs. a more common
        # metallic feature.
        graph = self._mark_bare_electron_indicator(
            graph=graph, 
            radii=radii
        )
        
        # Sometimes a bare electron or metal feature will be mislabeled due to it
        # being nearly between two atoms. In these cases, the features are very
        # far outside the atoms radius, while a covalent bond never is. We relabel them
        # here.
        graph = self._correct_far_covalent_features(graph)
        
        # Finally, we want to distinguish between a metal and a bare electron.
        # This is currently very arbitrary and based on a series of cutoffs.
        graph = self._mark_metallic_or_electride(
            graph,
            electride_elf_min=electride_elf_min,
            electride_depth_min=electride_depth_min,
            electride_charge_min=electride_charge_min,
            electride_volume_min=electride_volume_min,
            electride_radius_min=electride_radius_min,
            )
        
        # In some cases, the user may not have used a pseudopotential with enough core electrons.
        # This can result in an atom having no assigned core/shell, which will
        # result in nonsense later. We check for this here and throw an error
        assigned_atoms = []
        for i in graph.nodes:
            node = graph.nodes[i]
            # We only want to consider basins that are core or shell, so we check
            # here and skip otherwise
            basin_subtype = node.get("subtype", None)
            if not basin_subtype in ["core", "shell"]:
                continue
            atom = node.get("nearest_atom", None)
            if atom is not None:
                assigned_atoms.append(atom)
        if (
            len(np.unique(assigned_atoms)) != len(self.structure)
            and not self.ignore_low_pseudopotentials
        ):
            
            raise Exception(
                "At least one atom was not assigned a zero-flux basin. This typically results"
                "from pseudo-potentials (PPs) with only valence electrons (e.g. the defaults for Al, Si, B in VASP 5.X.X)."
                "Try using PPs with more valence electrons such as VASP's GW potentials"
            )
        # Finally, we add a label to each node with a summary of information
        # for plotting
        for i in graph.nodes:
            node = graph.nodes[i]
            if not "split" in node.keys():
                try:
                    subtype = node["subtype"]
                except:
                    raise Exception(
                        "At least one ELF feature was not assigned. This is a bug. Please report to our github:"
                        "https://github.com/jacksund/simmate/issues"
                    )

        return graph
    
    def _get_bifurcations(self):
        """
        Scans through each bader basin and determines when they connect
        to the basins bordering them. Then determines the ELF values
        at which there are topological changes to the ELF isosurface.
        Returns a dictionary of ELF values and the basins in shared
        domains at that value.
        """
        logging.info("Finding all basin connections")
        bader = self.bader
        
        elf_grid = bader.reference_grid
        neighbor_shifts, _ = elf_grid.voxel_26_neighbors
        # get connections between neighboring basins
        edge_indices = np.argwhere(bader.basin_edges)
        connection_array = find_connections(
            bader.basin_labels,
            elf_grid.total,
            edge_indices,
            len(bader.basin_maxima_frac),
            neighbor_shifts,
            )
        
        # also add the maximum value of each basin as the point it 'connects' to
        # itself
        basin_maxima = bader.basin_maxima_ref_values
        basin_indices = np.arange(len(basin_maxima))
        connection_array[basin_indices, basin_indices] = basin_maxima
        
        connection_indices = np.nonzero(connection_array)
        connection_pairs = np.column_stack(connection_indices)  # same as argwhere result
        connection_elfs = connection_array[connection_indices]

        # Now we want to compile which domains exist at each elf value and the
        # basins they contain. We will scan over each maximum and connection
        # point and see if there is a change in domains
        possible_elf_values = list(np.unique(connection_elfs))
        possible_elf_values.reverse()
        # For each elf value, starting from the highest, we check which basins are connected
        # to each other to form a domain.
        important_values = {}
        connected_components = []
        for elf_value in tqdm(possible_elf_values, desc="Finding bifurcation elf values"):
            # Find the indices where connections are above the current value
            # and get the connected basins
            connected_basins = connection_pairs[connection_elfs>=elf_value]
            uf = UnionFind()
            uf.bulk_union(connected_basins[:,0], connected_basins[:,1])
            # Get the previous and current groups
            previous_connected = connected_components.copy()
            connected_components = uf.groups_sets()
            # Check that this list of sets is different from the previous one. If
            # it is, we add this as an important elf value
            if connected_components != previous_connected:
                important_values[elf_value] = connected_components
        # Our basins are in sets currently, but we want them to be arrays for operations
        # down the line. We convert them here
        important_values_new = {}
        for key, value in important_values.items():
            important_values_new[key] = [np.array(list(i)).astype(int) for i in value]
        important_values = important_values_new
        return important_values
    
    def _initialize_bifurcation_graph(self):
        # Now that we have our elf values where changes occur, we want to generate our
        # initial graph
        graph = BifurcationGraph()
        # The elf values where topological changes happen are noted by the keys
        # of our dictionary
        keys = np.unique([i for i in self.bifurcations.keys()])
        # Our initial domain contains all of the basins and is stored in the
        # lowest key. We add this to our graph to avoid issues later in processing
        # due to it being the root.
        current_basin_groups = self.bifurcations[keys[0]]
        graph.add_node(
            basins=current_basin_groups[0],
            appears_at=0.0
            )

        # Now we loop over the ELF values at which bifurcations occur or maxima
        # exist
        for key in tqdm(keys[1:], desc="Constructing graph"):
            # Get the current and previous groups for comparison
            previous_basin_groups = current_basin_groups.copy()
            current_basin_groups = self.bifurcations[key]
            
            for basin_group in current_basin_groups:
                # we check if this basin group exists in the previous one. If it
                # does, we've already added a node for this group and continue
                old_group = any(np.array_equal(basin_group, other_group) for other_group in previous_basin_groups)
                if old_group:
                    continue
                # otherwise, this is a new group and we want to add a node representing it.
                # We also want to find the node that should be the parent of this one
                # so that we can assign an edge and label the value at which the
                # parent split. This corresponds to the most recent node that
                # had a group containing all of this group.
                nodes = graph.nodes.copy()
                nodes.reverse()
                parent_found = False
                for node in nodes:
                    if np.all(np.isin(basin_group, node.basins)):
                        parent_node = node
                        parent_found = True
                        break
                if not parent_found:
                    breakpoint()
                # We've now found our parent and we want to update it's split value
                parent_node.disappears_at = key
                parent_node.reducible = True
                # Now we update our node count and add the current node
                graph.add_node(
                    parent=parent_node,
                    basins=basin_group,
                    appears_at=key,
                    reducible=False,
                    )
        
        # we also want to add the values at which irreducible features disappear
        for node in graph:
            if not node.reducible:
                maxima_values = self.bader.basin_maxima_ref_values[node.basins]
                node.disappears_at = maxima_values.max()
        
        self._bifurcation_graph = graph
    
    def _get_downscale_grids(self):
        # get the elf and label grids
        elf_grid = self.elf_grid
        labels = self.bader.basin_labels.copy()
        label_grid = elf_grid.copy()
        label_grid.total = labels
        
        # We will use a downscaled version of our ELF for speed in some cases
        downscale_resolution = self.downscale_resolution
        if downscale_resolution is not None and self.elf_grid.voxel_resolution > downscale_resolution:
            self._downscaled_elf_grid = elf_grid.regrid(downscale_resolution)
            downscaled_label_grid = label_grid.regrid(downscale_resolution, order=0)
        else:
            # NOTE: We don't copy the grids to avoid unneccessary reallocation
            self._downscaled_elf_grid = elf_grid
            downscaled_label_grid = label_grid
        self._downscaled_labels = downscaled_label_grid.total
    
    
    def _assign_node_properties(self):
        # get bifurcation graph and bader object
        graph = self.bifurcation_graph
        bader = self.bader
        labels = bader.basin_labels
        # get downscaled graphs
        downscaled_labels = self.downscaled_labels
        downscaled_elf_grid = self.downscaled_elf_grid
        checked_nodes = []
        # Loop over this graph and label each node with important information
        for node in tqdm(graph, desc="Calculating node properties"):
            checked_nodes.append(node.key)
            # get parent and included basins
            parent = node.parent
            basins = node.basins
            if node.reducible:
                # this is a reducible domain. We want to get the atoms contained in
                # this domain when it first appeared, as well as whether it was
                # an infinite connection right before it split
                if parent is not None:
                    parent_split = parent.disappears_at - 0.01
                    low_elf_mask = np.isin(downscaled_labels, basins) & np.where(
                        downscaled_elf_grid.total > parent_split, True, False
                    )
                    high_elf_mask = np.isin(
                        downscaled_labels, basins
                    ) & np.where(
                        downscaled_elf_grid.total > (node.disappears_at - 2 * 0.01), True, False
                    )
                    # TODO:
                        # If I rework the methods for checking for surrounded
                        # atoms or infinite features, I can probably move the
                        # downscaled data to just be an array rather than a Grid object.
                        # I can also probably fix the issues I had that require slight
                        # buffering of the mask
                    atoms = downscaled_elf_grid.get_atoms_surrounded_by_volume(low_elf_mask)
                    # BUG-FIX we check if this feature is infinite right
                    # before it split. This should fix issues with atomic
                    # features in small cells that connect to themselves
                    # by wrapping around the cell. In a larger cell, the
                    # split would be noted, but it's not for these.
                    is_infinite = downscaled_elf_grid.check_if_infinite_feature(high_elf_mask)
                else:
                    # if we have no parent this is our first node and
                    # we have as many atoms as there are in the structure
                    atoms = [i for i in range(len(self.structure))]
                    # This is always infinite, so we note that by adding -1
                    # to the front of our list
                    is_infinite = True
                
                # set new attributes for this node
                node.atoms = atoms
                node.is_infinite = is_infinite

            else:
                # This is an irreducible domain.
                # We want to store data relavent to the type of domain it might
                # be.
                # First we get a mask representing where this feature is
                basin_mask = np.isin(labels, basins)
                max_elf = np.max(self.elf_grid.total[basin_mask])
                depth = max_elf - parent.disappears_at
                # We also want to mark a type of depth corresponding to the
                # point where this feature connected with an infinite domain.
                # NOTE: There will always be an ancestor that is infinite, because
                # the root is infinite
                ancestors = node.ancestors
                for ancestor in ancestors:
                    try:
                        if ancestor.is_infinite:
                            break
                    except:
                        breakpoint()
                depth_3d = max_elf - ancestor.disappears_at
                # Using this, we can find the average frac coords of the attractors
                # in this basin
                # TODO: Check if this is necessary. With the updated Bader package
                # no basin maxima should ever border each other, and all of them
                # should eventually reduce to a distinct basin.
                empty_structure = self.structure.copy()
                empty_structure.remove_oxidation_states()
                empty_structure.remove_species(empty_structure.symbol_set)
                frac_coords = bader.basin_maxima_frac[basins]
                if len(frac_coords) == 1:
                    frac_coord = frac_coords[0]
                else:
                    # We append these to an empty structure and use pymatgen's
                    # merge method to get their average position
                    for frac_coord in frac_coords:
                        empty_structure.append("He", frac_coord)
                    if len(empty_structure) > 1:
                        empty_structure.merge_sites(tol=1, mode="average")
                    frac_coord = empty_structure.frac_coords[0]

                # We can also get the charge from the bader analysis
                charges = bader.basin_charges[basins]
                charge = charges.sum()
                # and the volumes
                volumes = bader.basin_volumes[basins]
                volume = volumes.sum()
                # We can also get the distance of this feature to the nearest
                # atom and what that atom is. We have to assume we have several
                # basins, so we use the shortest distance and corresponding ato
                distances = bader.basin_atom_dists[basins]
                distance = distances.min()
                nearest_atom = bader.basin_atoms[basins][
                    np.where(distances == distance)[0][0]
                ]

                # Now we update this node with the information we gathered
                try:
                    node.max_elf = round(max_elf, 4)
                    node.depth = round(depth, 4)
                    node.depth_3d = round(depth_3d, 4)
                    node.charge = round(charge, 4)
                    node.volume = round(volume, 4)
                    node.atom_distance = round(distance, 4)
                    node.nearest_atom = nearest_atom
                    node.nearest_atom_type = self.structure[nearest_atom].specie.symbol
                    node.frac_coords = frac_coords
                except:
                    breakpoint()
        return graph
    
    def _clean_reducible_nodes(self):
        # TODO: Is this still necessary?
        graph = self.bifurcation_graph
        # nodes_to_remove = []
        for node in graph:
            if not node.reducible:
                continue

            children = node.children
            # check if we only have one child
            if len(children) != 1:
                continue
            # check if this child is reducible
            if not children[0].reducible:
                continue
            # If we made it to this point, the current node is reducible, but only
            # has one child. We want to delete this node.
            node.remove()
    
    def _mark_atomic(self, shell_depth: float = 0.05):
        elf_data = self.elf_grid.total
        graph = self.bifurcation_graph
        # create a variable to track the number of atoms left to assign
        remaining_atoms = len(self.structure)
        # BUG: The remaining atom count is broken currently. Sometimes atoms are
        # double counted, e.g. when a core feature breaks off before another feature
        # that fully surround the atom.
        for node in tqdm(graph, desc="Marking atomic nodes"):
            # We are going to use attributes of each irreducible feature to
            # assign its children, so if this node isn't irreducible we skip it
            if not node.reducible:
                continue
            # There are three situations for our reducible feature. First, if
            # it surrounds 0 atoms then all of its children must be valence. We
            # skip in this case
            if len(node.atoms) == 0:
                # Label all children as valence
                for child in node.children:
                    if child.reducible:
                        continue
                    # We sometimes label the nodes of reducible features as covalent.
                    # We don't want to overwrite these so we check that the subtype
                    # doesn't exist
                    elif not hasattr(child, "basin_subtype"):
                        child.basin_type = "val"
                        child.basin_subtype = None
                continue
            # Second, it can contain more than one atom. In a full core model,
            # The atoms that split off of this type of feature would themselves
            # be reducible and always fit into the next category. However, with
            # a pseudopotential model, this is not the case. Instead, an atom
            # may only have a single irreducible feature. We check for this by
            # noting if the child features fully surround an atom at the ELF they separate at
            # TODO: It may be that this loop should just be for when the number
            # of atoms is infinite. Basically, any finite number suggests a
            # molecular feature and all basins would be core/shell/covalent/lone-pair.
            elif node.is_infinite:
                for child in node.children:
                    # skip any children that are reducible
                    if child.reducible:
                        continue
                    # Using these basins, and the value the basin split at, we
                    # get a mask for the location of the basin
                    low_elf_mask = np.isin(self.bader.basin_labels, child.basins) & np.where(
                        elf_data > node.disappears_at, True, False
                    )
                    atoms_in_basin = self.elf_grid.get_atoms_in_volume(low_elf_mask)
                    basin_type = "val"
                    basin_subtype = None
                    if len(atoms_in_basin) > 0:
                        basin_type = "atom"
                        basin_subtype = "core"
           
                        # Note that we found a new atom
                        remaining_atoms -= 1
                    # label this basin
                    child.basin_type = basin_type
                    child.basin_subtype = basin_subtype

            # The final option is that our reducible region surrounds a finite
            # number of atoms. Most of the subregions of this
            # environment will be atomic, but they can be of several types including
            # atom shells/cores, unshared electrons, lone-pairs. The one exception
            # is heterogenous covalent bonds, which should be shared.
            elif len(node.atoms) > 0:                    
                # Otherwise, these features are atomic, shells, or covalent/lone-pairs
                # Now we loop over all of the children of this feature, including
                # deeper children. We label these children based on their depth
                # and whether they surround the atom. We label features as:
                # core, shell, or other.
                # The "others" will be assigned later on as lone-pairs or covalent
                # depending on if they are along an atomic bond
                for child in node.deep_children:
                    # define our default types
                    basin_type = "atom"
                    basin_subtype = None
                    # If we have a split, we don't want to label this node so
                    # we continue.
                    if child.reducible:
                        continue
                    # If we have many shell basins that form a sphere around the
                    # atom they may separate at a low depth. However, lone-pairs
                    # that are highly symmetric may also separate in a similar way.
                    # We actually want the depth to the point where the basin connects
                    # to a reducible domain surrounding the atom of interest. This is
                    # the point where this node split.
                    basin_shell_depth = child.max_elf - node.disappears_at

                    if basin_shell_depth < shell_depth:
                        basin_subtype = "shell"
                    else:
                        # otherwise, we check if the feature surrounds an atom
                        # Using the childs basins, and the value the basin split at, we
                        # get a mask for the location of the basin
                        low_elf_mask = np.isin(self.bader.basin_labels, child.basins) & np.where(
                            elf_data > child.parent.disappears_at, True, False
                        )
                        atoms_in_basin = self.elf_grid.get_atoms_in_volume(low_elf_mask)
                        
                        if len(atoms_in_basin) > 0:
                            # We have an core region
                            basin_subtype = "core"
                        else:
                            # otherwise its an other
                            basin_type = "val"
                            basin_subtype = "other"
                    # Now we assign our types to the child node.
                    child.basin_type = basin_type
                    child.basin_subtype = basin_subtype

        return graph
    
    def _mark_covalent_lonepair(
        self,
        min_covalent_charge: float = 0.6,
        min_covalent_angle: float = 135,
        min_covalent_bond_ratio: float = 0.4,
    ):
        """
        Takes in a bifurcation graph and labels valence features that
        are obviously metallic or covalent
        """
        graph = self.bifurcation_graph
        
        for node in graph:
            if node.reducible or getattr(node, "basin_type") != "val":
                continue
            previous_subtype = getattr(node, "basin_subtype")
            # Default to bare electron
            basin_type = "val"
            subtype = "bare electron"

            # Check for covalent character based on position relative to bonds.
            # We create a temporary structure to calculate distances to neighboring
            # atoms. This is just to utilize pymatgen's distance method which
            # takes periodic boundaries into account.
            # TODO: This may be slow for larger structures. This could probably
            # be done using numpy arrays and the structure.distance_matrix
            # We assume there is only one basin, as this is the typical case for
            # covalent bonds
            frac_coords = node.frac_coords
            temp_structure = self.structure.copy()
            temp_structure.append("X", frac_coords)
            nearest_atom = node.nearest_atom
            atom_dist = round(temp_structure.get_distance(nearest_atom, -1), 2)
            atom_neighs = self.atom_coordination_envs[nearest_atom]
            # We want to see if our feature lies directly between our atom and
            # any of its neighbors.
            covalent = False
            # If we're above our charge cutoff, we check if we are along a bond
            if node.charge > min_covalent_charge:
                for neigh_dict in atom_neighs:
                    # We use the temp structure to calculate distance between the
                    # feature and neighbors. This automatically acounts for wrapping
                    # in the unit cell
                    neigh_idx = neigh_dict["site_index"]
                    neigh_dist = round(temp_structure.get_distance(neigh_idx, -1), 2)
                    # We use the distance calculated by cnn for the atom/neigh dist
                    atom_neigh_dist = round(neigh_dict["site"].nn_distance, 2)
                    # Sometimes we have a lone-pair that appears to be within our
                    # angle cutoff (e.g. CaC2), but is much closer to one atom than
                    # a covalent bond would be. We check for this here with a ratio.
                    atom_dist_ratio = atom_dist / atom_neigh_dist
                    if atom_dist_ratio < min_covalent_bond_ratio:
                        continue
                    # We want to apply the law of cosines to get angle with feature
                    # at center, then convert to degrees. This won't work if our feature
                    # is exactly along the bond, so we first check for that case.
                    # we check within a small tolerance for rounding errors
                    test_dist = round(atom_dist + neigh_dist, 2)
                    tolerance = 0.01
                    if (
                        (test_dist - tolerance)
                        <= atom_neigh_dist
                        <= (test_dist + tolerance)
                    ):
                        covalent = True
                        break
                    try:
                        feature_angle = math.acos(
                            (atom_dist**2 + neigh_dist**2 - atom_neigh_dist**2)
                            / (2 * atom_dist * neigh_dist)
                        )
                        feature_angle = feature_angle * 180 / math.pi
                    except:
                        # We don't have a valid triange. This can happen if the feature
                        # is along the bond but not between the atoms (lone-pairs)
                        # or if we are comparing atoms not near the lone pair. In
                        # either case we don't have a covalent bond and continue
                        continue
    
                    # check that we're above the cutoff
                    if feature_angle > min_covalent_angle:
                        covalent = True
                        break
            # Now we've noted if our feature is covalent. If it is, we label it
            # as such
            if covalent:
                subtype = "covalent"
            # We also noted in our atomic assignment which features were part
            # of the atomic branch, but weren't shells or cores. The remaining
            # options were covalent or lone-pairs and we've just assigned the
            # covalent ones. So, if our previous subtype was "other" and the
            # feature isn't covalent it must be a lone-pair
            if previous_subtype == "other" and not covalent:
                subtype = "lone-pair"
                # BUG: In some rare cases, this may misassign basins that should
                # be bare electrons (e.g. Sr6CrN6) if the basin doesn't bifurcate
                # before the atomic basins. This could potentially be corrected
                # for with a distance cutoff.

            # We've now checked for metallic character, covalent bonds and most
            # lone-pairs. We update our subtype accordingly
            node.basin_type = basin_type
            node.basin_subtype = subtype

        # There is an exception to the lone-pair rule that can result in missing
        # a lone-pair assignment. If a covalent/lone-pair feature surrounds two atoms
        # these features won't be assigned as "other".
        # This happens in CaC2 around the C2 molecules for example. The covalent
        # bonds are labeled in the loop above, but the lone-pair will
        # still be labeled as a bare electron. We correct for this in an
        # additional loop by checking for bare electrons that are siblings with
        # covalent bonds.
        # BUG-FIX rather than exact siblings, we want all of the features that
        # are children of the parent domain that fully surrounds the molecule
        # TODO: This could be moved to a Node property
        def get_molecule_parent(node):
            # get parent that fully surrounds at least one atom
            molecule_parent = None
            parent = node.parent
            while molecule_parent is None:
                if len(parent.atoms) != 0:
                    molecule_parent = parent
                else:
                    parent = parent.parent
            return molecule_parent
        
        # keep track of nodes to reassign as lone-pairs. We can't reassign them
        # in this loop because we check that at least one sibling is a covalent
        # bond, and we don't want to accidentally relabel them.
        nodes_to_relabel = []
        for node in graph:
            if node.reducible or getattr(node, "basin_type") != "val":
                continue
            if node.basin_subtype == "bare electron":

                all_cov_lp_be = True
                at_least_one_cov = False
                molecule_parent = get_molecule_parent(node)

                # Check if all siblings are covalent, bare electrons, or lone-pairs. If so,
                # this is a lone-pair
                for sibling in molecule_parent.deep_children:
                    # skip reducible siblings
                    if sibling.reducible:
                        continue
                    # make sure this sibling isn't the child of a different submolecule
                    direct_parent = get_molecule_parent(sibling)
                    if len(direct_parent.atoms) != 0 and direct_parent != molecule_parent:
                        continue

                    # We need to make sure there's at least one covalent bond as well
                    if sibling.basin_subtype == "covalent":
                        at_least_one_cov = True
                    elif sibling.basin_subtype not in [
                        "bare electron",
                        "covalent",
                        "lone-pair",
                    ]:
                        all_cov_lp_be = False
                if all_cov_lp_be and at_least_one_cov:
                    nodes_to_relabel.append(node)
        for node in nodes_to_relabel:
            node.basin_subtype = "lone-pair"
            
    def _correct_for_high_depth_shells(self):
        """
        Sometimes atomic shells have particularly deep separations, for
        example when they are heavily polarized (e.g. Er2C). In these
        cases, the shell will split into one irreducible domain and
        one or more reducible domains. This is similar to a covalent bond/
        lone-pair shell. However, none of the domains will fit the criteria
        for a covalent bond, so all of them will be marked as shells or
        lone-pairs. We change all of them to be marked as shells here.
        """
        graph = self.bifurcation_graph
        
        for node in graph:
            if not node.reducible:
                continue
            # We check only for situations where we have a finite number of
            # atoms in a reducible region
            if len(node.atoms) > 0 and not node.is_infinite:
                all_lone_pairs_or_shells = True
                for child in node.deep_children:
                    # skip reducible domains
                    if child.reducible:
                        continue
                    # check if the child is not a lone pair or shell
                    if child.basin_subtype not in ["lone-pair", "shell"]:
                        all_lone_pairs_or_shells = False
                        break
                if not all_lone_pairs_or_shells:
                    # This reducible domain isn't a shell. Continue
                    continue
                for child in node.deep_children:
                    # skip reducible domains
                    if child.reducible:
                        continue
                    child.basin_type = "atom"
                    child.basin_subtype = "shell"

        return graph
    
    def _correct_far_shell_features(self):
        """
        Corrects any shell nodes that are outside the radius of the atom
        to be considered bare electrons instead
        """
        
        for node in self.bifurcation_graph:
            if node.reducible or getattr(node, "basin_subtype") != "shell":
                continue
            atom = node.nearest_atom
            atom_radius = self.atomic_radii[atom]
            distance = node.atom_distance
            tolerance = 0.0
            if distance > atom_radius + tolerance:
                # This shouldn't be considered a shell basin and we relabel it
                # We also need to find the radius of this feature to match what we
                # had before
                node.basin_type = "val"
                node.basin_subtype = "bare electron"

    def _combine_shells(self, nodes: list) -> BifurcationGraph():
        """
        Combines a list of nodes into one
        """
        # Get the new values for each feature of this node
        basins = []
        atom_distance = 50
        volume = 0
        charge = 0
        max_elf = 0
        nearest_atom = -1
        nearest_atom_type = None
        frac_coords = None
        depth = 0
        depth_3d = 0
        # update all of our shell characteristics
        for child in nodes:
            nearest_atom = child.nearest_atom
            nearest_atom_type = child.nearest_atom_type
            basins.extend(child.basins)
            atom_distance = min(atom_distance, child.atom_distance)
            volume += child.volume
            charge += child.charge
            max_elf = max(max_elf, child.max_elf)
            frac_coords = child.frac_coords
            depth = max(depth, child.depth)
            depth_3d = max(depth_3d, child.depth_3d)

        # Add the attributes
        node = nodes[0]
        node.basin_type = "atom"
        node.basin_subtype = "shell"
        node.basins = np.array(basins)
        node.atom_distance = atom_distance
        node.volume = volume
        node.charge = charge
        node.max_elf = max_elf
        node.nearest_atom = nearest_atom
        node.nearest_atom_type = nearest_atom_type
        node.depth = depth
        node.depth_3d = depth_3d
        node.frac_coords = frac_coords

        children_to_remove = nodes[1:]
        # delete all of the unused nodes
        for child in children_to_remove:
            child.remove()

    def _reduce_atomic_shells(self):
        """
        Reduces shell nodes to a single node
        """
        graph = self.bifurcation_graph
        # We want to combine any nodes that belong to the same atomic shell. We
        # can do this by confirming that they share 2 aspects: The same closest
        # atom and a similar distance to the atom. To do this, we create a dictionary
        # to store these two attributes and the associated shells
        shell_groups = {}
        reducible_nodes = []
        group_num = 0
        for node in graph:
            if node.reducible:
                reducible_nodes.append(node)
                continue
            if getattr(node, "basin_subtype") != "shell":
                continue
            # First we get the shells nearest atom and distance
            atom = node.nearest_atom
            dist = node.atom_distance
            # Now we compare to all of our dictionary items
            assigned_group = None
            for shell_group, values in shell_groups.items():
                group_atom = values["atom"]
                dist_diff = values["dists"] - dist
                # We calculate a percent difference since shells close to the
                # core can be very close together. This likely doesn't matter
                # for a PP model anyways.
                percent_dist_diff = dist_diff / dist
                if group_atom == atom and percent_dist_diff.max() < 0.2:
                    assigned_group = shell_group
                    break
            if assigned_group is not None:
                dists = shell_groups[assigned_group]["dists"]
                dists = np.insert(dists,len(dists),dist)
                shell_groups[assigned_group]["dists"] = dists
                shell_groups[assigned_group]["nodes"].append(node)
            else:
                shell_groups[group_num] = {
                    "atom" : atom,
                    "dists" : np.array([dist]),
                    "nodes" : [node],
                    }
                group_num += 1
        
        # Now we want to go through and combine all of the shells we just grouped
        for group, values in shell_groups.items():
            nodes = values["nodes"]
            self._combine_shells(nodes)
        
        # Now that we've done that, there may be some nodes that were parents
        # of these shells that are either empty or a parent to one newly grouped
        # shell. We loop over the potential parents backwards, deleting any that
        # have no children and replacing any that have only one child
        reducible_nodes.reverse()
        for parent in reducible_nodes:
            children = parent.children
            child_num = len(children)
            if child_num == 0:
                # This is an empty node and we just delete it.
                parent.remove()
            elif child_num == 1:
                # This node is now the parent of a single shell feature. We replace
                # it.
                child = children[0]
                # recalculate depth and assign it to the child
                parent_elf = parent.disappears_at
                child.depth = child.max_elf - parent_elf
                child.reducible = True
                # delete the parent
                parent.remove()

        return graph
    
    def _mark_feature_radii(
            self,
            graph: BifurcationGraph(),
            bader: Bader,
            ):
        basin_radii = bader.basin_surface_distances
        valence_summary = self._get_valence_summary(graph)
        for feature_idx, attributes in valence_summary.items():
            basins = attributes["basins"]
            feature_radius = basin_radii[basins].min()
            networkx.set_node_attributes(
                graph, {feature_idx: {"feature_radius": feature_radius}}
            )
        return graph
        
    def _mark_bare_electron_indicator(
        self,
        graph: BifurcationGraph(),
        radii,
    ) -> BifurcationGraph():
        """
        Takes in a bifurcation graph and calculates an electride character
        score for each valence feature. Electride character ranges from
        0 to 1 and is the combination of several different metrics:
        ELF value, charge, depth, volume, and atom distance.
        """
        valence_summary = self._get_valence_summary(graph)

        for feature_idx, attributes in tqdm(valence_summary.items(), desc="Calculating bare electron character"):
            # We want to get a metric of how "bare" each feature is. To do this,
            # we need a value that ranges from 0 to 1 for each attribute we have
            # available. We can combine these later with or without weighting to
            # get a final value from 0 to 1.
            # First, the ELF value already ranges from 0 to 1, with 1 being more
            # localized. We don't need to alter this in any way.
            elf_contribution = attributes["max_elf"]

            # next, we look at the charge. If we are using a spin polarized result
            # the maximum amount should be 1. Otherwise, the value could be up
            # to 2. We make a guess at what the value should be here
            charge = attributes["charge"]
            if self._spin_polarized:
                max_value = 1
            else:
                if 0 < charge <= 1.1:
                    max_value = 1
                else:
                    max_value = 2
            # Anything significantly below our indicates metallic character and
            # anything above indicates a feature like a covalent bond with pi contribution.
            # we use a symmetric linear equation around our max value that maxes out at 1
            # where the charge exactly matches and decreases moving away.
            if charge <= max_value:
                charge_contribution = charge / max_value
            else:
                # If somehow our charge is much greater than the value, we will
                # get a negative value, so we use a max function to prevent this
                charge_contribution = max(-charge / max_value + 2, 0)

            # Now we look at the depth of our feature. Like the ELF value, this
            # can only be from 0 to 1, and bare electrons tend to take on higher
            # values. Therefore, we leave this as is.
            # NOTE: The depth here is the depth to the first irreducible feature
            # that extends infinitely in at least one direction. This is different
            # from the technical "depth" used in ELF topology analysis, but is
            # more related to how isolated a feature is.
            depth_contribution = attributes["3d_depth"]

            # Next is the volume. Bare electrons are usually thought of as being
            # similar to a free s-orbital with a similar size to a hydride. Therefore
            # we use the hydride crystal radius to calculate an ideal volume and set
            # this contribution as a fraction of this, capping at 1.
            hydride_radius = 1.34  # Taken from wikipedia and subject to change
            hydride_volume = 4 / 3 * 3.14159 * (hydride_radius**3)
            volume_contribution = min(attributes["volume"] / hydride_volume, 1)

            # Next is the distance from the atom. Ideally this should be scaled
            # relative to the radius of the atom, but which radius to use is a
            # difficult question. We use CrystalNN to get the neighbors around
            # the nearest atom and get the EN difference. We use this to guess
            # whether covalent or ionic radii should be used, then pull the appropriate one.
            # First, we also want to get the coordination environment of this
            # feature, even though this doesnt feed into our BEI.
            frac_coords = attributes["frac_coords"]
            temp_structure = self.structure.copy()
            temp_structure.append("H-", frac_coords)
            cnn = CrystalNN(distance_cutoffs=None)
            coordination = cnn.get_nn_info(temp_structure, -1)
            coord_num = len(coordination)
            coord_indices = [i["site_index"] for i in coordination]
            coord_atoms = [temp_structure[i].specie.symbol for i in coord_indices]
            # Now that we have the nearby atoms, we want to get the smallest radius
            # of this basin
            atom_indices = np.unique(coord_indices)
            atom_radius = 10
            atom_distance = 10
            dist_minus_radius = 10
            nearest_atom_idx = -1
            nearest_atom_species = None
            for atom_idx in atom_indices:
                atom_radius_new = radii[atom_idx]
                dist = temp_structure.get_distance(atom_idx, -1)
                dist_minus_radius_new = dist-atom_radius_new
                if dist_minus_radius_new < dist_minus_radius:
                    dist_minus_radius = dist_minus_radius_new
                    atom_radius = atom_radius_new
                    atom_distance = dist
                    nearest_atom_idx = atom_idx
                    nearest_atom_species = temp_structure[atom_idx].specie.symbol
                    
            # Now that we have a radius, we need to get a metric of 0-1. We need
            # to set an ideal distance corresponding to 1 and a minimum distance
            # corresponding to 0. The ideal distance is the sum of the atoms radius
            # plus the radius of a true bare electron (approx the H- radius). The
            # minimum radius should be 0, corresponding to the radius of the atom.
            # Thus covalent bonds should have a value of 0 and lone-pairs may
            # be slightly within this radius, also recieving a value of 0.
            radius = dist - atom_radius
            dist_contribution = radius / hydride_radius
            # limit to a range of 0 to 1
            dist_contribution = min(max(dist_contribution, 0), 1)

            # We want to keep track of the full values in a convenient way
            unnormalized_contributors = np.array(
                [
                    elf_contribution,
                    charge,
                    depth_contribution,
                    attributes["volume"],
                    dist_minus_radius,
                ]
            )
            # Finally, our bare electron indicator is a linear combination of
            # the indicator above. The contributions are somewhat arbitrary, but
            # are based on chemical intuition. The ELF and charge contributions
            contributers = np.array(
                [
                    elf_contribution,
                    charge_contribution,
                    depth_contribution,
                    volume_contribution,
                    dist_contribution,
                ]
            )
            weights = np.array(
                [
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                ]
            )
            bare_electron_indicator = np.sum(contributers * weights)

            
            # we update our node to include this information
            networkx.set_node_attributes(
                graph,
                {
                    feature_idx: {
                        "unnormalized_bare_electron_indicator": unnormalized_contributors,
                        "bare_electron_indicator": bare_electron_indicator,
                        "bare_electron_scores": contributers,
                        "dist_beyond_atom": round(dist_minus_radius,4),
                        "coord_num": coord_num,
                        "coord_indices": coord_indices,
                        "coord_atoms": coord_atoms,
                        "atom_distance": atom_distance,
                        "nearest_atom": nearest_atom_idx,
                        "nearest_atom_type": nearest_atom_species,
                    }
                },
            )
            
        return graph
    
    def _correct_far_covalent_features(self, graph: BifurcationGraph()) -> BifurcationGraph():
        # BUG-FIX On occasion, a metal feature will sit very close to being along
        # an atom-atom bond, but will sit well outside that atoms ELF radius. In
        # these cases they will be mislabeled as covalent. We correct for that here
        valence_summary = self._get_valence_summary(graph)
        for feature_idx, attributes in valence_summary.items():
            dist_beyond_atom = attributes["dist_beyond_atom"]
            feature_subtype = attributes["subtype"]
            if dist_beyond_atom > 0.2 and feature_subtype in ["covalent", "lone-pair"]:
                networkx.set_node_attributes(graph,{feature_idx: {"subtype": "bare electron"}},)
        return graph
    
    def _mark_metallic_or_electride(
            self,
            graph: BifurcationGraph(),
            electride_elf_min: float = 0.5,
            electride_depth_min: float = 0.2,
            electride_charge_min: float = 0.5,
            electride_volume_min: float = 10,
            electride_radius_min: float = 0.3,
                                    ) -> BifurcationGraph():
        valence_summary = self._get_valence_summary(graph)
        # create an array of our conditions to check against
        conditions = np.array(
            [
                electride_elf_min,
                electride_depth_min,
                electride_charge_min,
                electride_volume_min,
                electride_radius_min,
            ]
        )
        for feature_idx, attributes in tqdm(valence_summary.items(), desc="Marking metallic and bare electron nodes"):
            if not attributes["subtype"] == "bare electron":
                # skip any covalent/lone-pair features
                continue
            # we have a bare electron. We check each condition
            condition_test = np.array(
                [
                    attributes["max_elf"],
                    attributes[
                        "3d_depth"
                    ],  # Note we use the depth to an infinite connection rather than true depth
                    attributes["charge"],
                    attributes["volume"],
                    # attributes["feature_radius"],
                    attributes["dist_beyond_atom"]
                ]
            )
            # check if we meet all conditions. If so we have a bare electron/electride
            if np.all(condition_test > conditions):
                subtype = "bare electron"
            else:
                # We don't meet our conditions so we consider this some form
                # of metallic feature
                subtype = "metallic"
            networkx.set_node_attributes(graph,{feature_idx: {"subtype": subtype}},)
        
        return graph

    ###########################################################################
    # Post Graph Construction
    ###########################################################################
    
    def _get_bifurcation_plot(
        self,
        graph: BifurcationGraph(),
    ):
        """
        Returns a plotly figure
        """
        indices = []
        end_indices = []
        # X position is determined by the ELF value at which the feature appears.
        Xn = []
        Xn1 = []  # Used for depth
        labels = []
        types = []
        for i in graph.nodes():
            indices.append(i)
            node = graph.nodes[i]
            if node.get("split", None) is None:
                if node["depth"] > 0.01:
                    Xn1.append(node["max_elf"])
                else:
                    Xn1.append(node["max_elf"] - node["depth"] + 0.01)
                end_indices.append(i)
                # Get label
                label = f"""type: {node["subtype"]}\ndepth: {node["depth"]}\ndepth to inf connection: {node["3d_depth"]}\nmax elf: {node["max_elf"]}\ncharge: {node["charge"]}\nvolume: {node["volume"]}\natom distance: {round(node["atom_distance"],4)}\nnearest atom index: {node["nearest_atom"]}\nnearest atom type: {node["nearest_atom_type"]}"""
                if node.get("bare_electron_indicator", None) is not None:
                    label += f'\nfeature radius: {round(node["feature_radius"],4)}\ndistance beyond atom: {node["dist_beyond_atom"]}'
                    label += f'\ncoord number: {node["coord_num"]}\ncoord atoms: {node["coord_atoms"]}'
                    label += f"\nBEI array: {node['bare_electron_scores'].round(4)}"
                types.append(node["subtype"])
            else:
                Xn1.append(-1)
                atom_num = node["atom_num"]
                if atom_num == -1:
                    atom_num = "infinite"
                label = f"""type: reducible\ncontained atoms: {node["atoms"]}\ntotal contained atoms: {atom_num}"""
                types.append("reducible")
            # change to html line break
            label = label.replace("\n", "<br>")
            labels.append(label)
            parent = graph.parent_dict(i)
            if parent is not None:
                Xn.append(parent["split"])

            else:
                Xn.append(0)
        
        def assign_y_positions(graph, node_idx, y_counter, y_positions, indices):
            # This function iteratively loops starting from the root node and
            # places each parent node at the average position of its children.
            # children are placed when found. The iterative nature results in
            # connecting lines not overlapping.
            children = graph.child_indices(node_idx)
            if len(children) == 0:  # it's a leaf
                y_positions[node_idx] = next(y_counter)
            else:
                for child in children:
                    assign_y_positions(graph, child, y_counter, y_positions, indices)
                child_ys = [y_positions[child] for child in children]
                y_positions[node_idx] = np.mean(child_ys)
        # Create a mapping from node ID to Y position
        y_positions = {}
        y_counter = itertools.count(0)  # This gives 0, 1, 2, ... for leaf placement
        
        # for root in root_nodes:
        assign_y_positions(graph, 1, y_counter, y_positions, indices)
        
        # Then set Yn using this
        Yn = [y_positions[i] for i in indices]
        
        # Normalize Y scale
        max_y = 2
        Yn = np.array(Yn, dtype=float)
        Yn -= Yn.min()
        if Yn.max() > 0:
            Yn /= Yn.max()
            Yn *= max_y
        # Get how spread out each node is
        y_division = max_y / len(end_indices)

        # Now we need to get the lines that will be used for each edge. These will use
        # a nested lists where each edge has one entry and the sub-lists contain the
        # two x and y entries for each edge.
        Xe = []
        Ye = []
        for edge in graph.edges():
            parent = edge[0]
            child = edge[1]
            Xe.extend([Xn[indices.index(parent)], Xn[indices.index(child)], None])
            Ye.extend([Yn[indices.index(parent)], Yn[indices.index(child)], None])

        # create the figure and add the lines and nodes
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=Xe,
                y=Ye,
                mode="lines",
                name="connection",
                line=dict(color="rgb(210,210,210)", width=3),
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
        for idx in range(len(Xn)):
            # get color
            basin_type = types[idx]
            # add nodes for each type of point
            # for basin_type in np.unique(types):
            # Color code by type
            if basin_type == "reducible":
                color = "rgba(128, 128, 128, 1)"  # grey
            elif basin_type == "shell" or basin_type == "core":
                color = "rgba(0, 0, 0, 1)"  # black
            elif basin_type == "covalent":
                color = "rgba(0, 255, 255, 1)"  # aqua
            elif basin_type == "metallic":
                color = "rgba(192, 192, 192, 1)"  # silver
            elif basin_type == "lone-pair":
                color = "rgba(128, 0, 128, 1)"  # purple
            elif basin_type == "bare electron":
                color = "rgba(128, 0, 0, 1)"  # maroon

            showlegend = basin_type not in already_added_types
            already_added_types.add(basin_type)
            sub_label = labels[idx]
            if Xn1[idx] == -1:
                fig.add_trace(
                    go.Scatter(
                        # x=xs,
                        # y=ys,
                        x=[Xn[idx]],
                        y=[Yn[idx]],
                        mode="markers",
                        name=f"{basin_type}",
                        marker=dict(
                            symbol="circle-dot",
                            size=18,
                            color=color,  #'#DB4551',
                            line=dict(color="grey", width=1),
                        ),
                        text=sub_label,
                        hoverinfo="text",
                        showlegend=showlegend,
                    )
                )
            else:
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
                        line=dict(color="grey"),
                        hoverinfo="text",
                        text=sub_label,
                        name=f"{basin_type}",
                        mode="lines",
                        opacity=0.8,
                        showlegend=showlegend,
                    )
                )

        # remove y axis label
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(range=[-0.1, 1], title="ELF"),
            yaxis=dict(
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
            ),
        )
        return fig

    def get_labeled_structure(
        self,
        graph: BifurcationGraph(),
        include_lone_pairs: bool = False,
        include_shared_features: bool = True,
        **kwargs,
    ):
        # First, we get the valence features for this graph and create a
        # structure that we will add features to
        valence_features = self._get_valence_summary(graph)
        structure = self.structure.copy()
        structure.remove_oxidation_states()
        structure_index_to_node = {}
        for feat_idx, attributes in valence_features.items():
            # get our subtype
            subtype = attributes["subtype"]
            if subtype == "bare electron":
                species = "e"
            if subtype == "covalent" and include_shared_features:
                species = "z"
            elif subtype == "metallic" and include_shared_features:
                species = "m"
            elif subtype == "lone-pair" and include_shared_features:
                species = "lp"

            # Now that we have the type of feature, we want to add it to our
            # structure.
            frac_coords = attributes["frac_coords"]
            structure.append(species, frac_coords)
            structure_index_to_node[len(structure)-1] = feat_idx

        # To find the atoms/electrides surrounding a covalent/metallic bond,
        # we need the structure to be organized with atoms first, then electrides,
        # then whatever else. We organize everything here.
        electride_indices = structure.indices_from_symbol("E")
        other_indices = []
        node_to_index = {}
        for symbol in ["M", "Z", "Lp"]:
            other_indices.extend(structure.indices_from_symbol(symbol))
        sorted_structure = self.structure.copy()
        sorted_structure.remove_oxidation_states()
        for i in electride_indices:
            frac_coords = structure[i].frac_coords
            sorted_structure.append("E", frac_coords)
            node=structure_index_to_node[i]
            node_to_index[node] = len(sorted_structure)-1
        for i in other_indices:
            symbol = structure.species[i].symbol
            frac_coords = structure[i].frac_coords
            sorted_structure.append(symbol, frac_coords)
            node=structure_index_to_node[i]
            node_to_index[node] = len(sorted_structure)-1
        
        # Now we want to add the nodes index to our graph
        for node, index in node_to_index.items():
            networkx.set_node_attributes(graph, {node: {"feature_structure_index":index}})

        logging.info(f"{len(electride_indices)} bare electrons found")
        if len(other_indices) > 0:
            f"{len(other_indices)} shared sites found"

        return sorted_structure
    
    ###########################################################################
    # Utilities
    ###########################################################################
    @property
    def _site_voxel_coords(self) -> np.array:
        frac_coords = self.structure.frac_coords
        vox_coords = self.elf_grid.get_voxel_coords_from_frac(frac_coords)
        return vox_coords.astype(int)
    
    @cached_property
    def _site_sphere_voxel_coords(self) -> list:
        site_sphere_coords = []
        for vox_coord in self._site_voxel_coords:
            nearby_voxels = self.elf_grid.get_voxels_in_radius(0.05, vox_coord)
            site_sphere_coords.append(nearby_voxels)
        return site_sphere_coords
    
    def _get_atomic_radii(self, radius_refine_method: str):
        bader = self.bader
        # We will need to get radii from the ELF. To do this, we need a labeled
        # bader result to pass to our PartitioningToolkit
        frac_coords = bader.basin_maxima_frac
        temp_structure = self.structure.copy()
        for node in self.bifurcation_graph:
            # skip nodes that aren't valence
            if node.reducible or getattr(node, "basin_type") != "val":
                continue
            
            subtype = node.basin_subtype
            
            if subtype == "covalent":
                species = "Z"
            elif subtype == "lone-pair":
                # species = "Lp"
                # We want to consider lone-pairs as part of the atom so we continue
                continue
            else:
                species = "X"
                
            for basin_idx in node.basins:
                frac_coord = frac_coords[basin_idx]
                temp_structure.append(species, frac_coord)
        
        # recalculate the atoms for our bader object
        bader_labeled = bader.copy()
        bader_labeled.run_atom_assignment(structure=temp_structure)

        partitioning = IonicRadiiTools(self.elf_grid, bader_labeled)
        # TODO Ideally, these radii are stored at a class level so that they
        # can be passed to the BadElfToolkit class for summary. However, this
        # requires knowledge of if this is spin-up/spin-down which I currently
        # don't have stored at this level
        self._atomic_radii = partitioning.get_elf_ionic_radii(
            refine_method=radius_refine_method, labeled_structure=temp_structure
        )
    
    @staticmethod
    def _get_shared_feature_neighbors(structure: Structure) -> NDArray:
        """
        For each covalent bond or metallic feature in a dummy atom labeled
        structure, returns a list of nearest atom neighbors.
        """
        # We want to get the atoms and electride sites that are closest to each
        # shared feature. However, we don't want to find any nearby shared features
        # as neighbors.
        # To do this we will remove all of the shared dummy atoms, and create
        # temporary structures with only one of the shared dummy atoms at a time.
        shared_feature_indices = []
        cleaned_structure = structure.copy()
        for symbol in ["Z", "M", "Le", "Lp"]:
            if not symbol in cleaned_structure.symbol_set:
                continue
            cleaned_structure.remove_species([symbol])
            shared_feature_indices.extend(structure.indices_from_symbol(symbol))
        shared_feature_indices = np.array(shared_feature_indices)
        shared_feature_indices.sort()
        # We will be using the indices of the cleaned structure to note neighbors,
        # so these must match the original structure. We assert that here
        assert all(
            cleaned_structure[i].species == structure[i].species
            for i in range(len(cleaned_structure))
        ), "Provided structure must list atoms and electride dummy atoms first"

        # Replace any electrides with "He" so that CrystalNN doesn't throw an error
        if "E" in cleaned_structure.symbol_set:
            cleaned_structure.replace_species({"E": "He"})
        # for each index, we append a dummy atom ("He" because its relatively small)
        # then get the nearest neighbors
        cnn = CrystalNN(distance_cutoffs=None)
        all_neighbors = []
        for idx in shared_feature_indices:
            neigh_indices = []
            # Add this dummy atom to the temporary structure
            frac_coords = structure[idx].frac_coords
            temp_structure = cleaned_structure.copy()
            temp_structure.append("He", frac_coords)
            # Get the nearest neighbors to this dummy atom
            nn = cnn.get_nn(temp_structure, -1)
            # Get the index for each neighboras a list, then append this list
            # to our full list. Note that it is important that these indices be
            # the same as in the original structure, so atoms and electrides must
            # come before shared electrons in the provided structure.
            for n in nn:
                neigh_indices.append(n.index)
            all_neighbors.append(neigh_indices)
        return all_neighbors

    def _get_atom_en_diff_and_cn(self, site: int) -> list([float, int]):
        """
        Uses the coordination environment of an atom to get the EN diff
        between it and it's neighbors as well as its coordination number.
        This is useful for guessing which radius to use.
        """
        # get the neighbors for this site and its electronegativity
        neigh_list = self.atom_coordination_envs[site]
        site_en = self.structure.species[site].X
        # create a variable for storing the largest EN difference
        max_en_diff = 0
        for neigh_dict in neigh_list:
            # get the EN for each neighbor and calculate the difference
            neigh_site = neigh_dict["site_index"]
            neigh_en = self.structure.species[neigh_site].X
            en_diff = site_en - neigh_en
            # if the difference is larger than the current stored one, replace
            # it.
            if abs(en_diff) > max_en_diff:
                max_en_diff = en_diff
        # return the en difference and number of neighbors
        return max_en_diff, len(neigh_list)
    
    def _get_valence_summary(self, graph: BifurcationGraph()) -> dict:
        """
        Takes in a bifurcation graph and summarizes any valence basin
        information as a nested dictionary where each key is the node
        index and each value is a dictionary of useful information
        """
        summary = {}
        for i in graph.nodes:
            node = graph.nodes[i]
            basin_type = node.get("type", None)
            if basin_type == "val":
                summary[i] = node
        return summary

    @classmethod
    def from_vasp(
        cls,
        elf_file: str | Path = "ELFCAR",
        charge_file: str | Path = "CHGCAR",
        **kwargs,
    ):
        """
        Creates a BadElfToolkit instance from the requested partitioning file
        and charge file in VASP format.
        """

        elf_grid = Grid.from_vasp(elf_file)
        charge_grid = Grid.from_vasp(charge_file)
        return cls(
            elf_grid=elf_grid,
            charge_grid=charge_grid,
            **kwargs,
        )
    
    @classmethod
    def from_cube(
        cls,
        elf_file: str | Path,
        charge_file: str | Path,
        **kwargs,
    ):
        """
        Creates a BadElfToolkit instance from the requested partitioning file
        and charge file in .cube format.
        """

        elf_grid = Grid.from_cube(elf_file)
        charge_grid = Grid.from_cube(charge_file)
        return cls(
            elf_grid=elf_grid,
            charge_grid=charge_grid,
            **kwargs,
        )
    
    @classmethod
    def from_dynamic(
        cls,
        elf_file: str | Path,
        charge_file: str | Path,
        **kwargs,
    ):
        """
        Creates a BadElfToolkit instance from the requested partitioning file
        and charge file. Attempts to guess the file format from the name of the
        files.
        """

        elf_grid = Grid.from_cube(elf_file)
        charge_grid = Grid.from_cube(charge_file)
        return cls(
            elf_grid=elf_grid,
            charge_grid=charge_grid,
            **kwargs,
        )
    
    ###########################################################################
    # Methods for writing results
    ###########################################################################
    def write_bifurcation_plot(
            self,
            filename: str | Path,
            ):
        plot = self.bifurcation_plot
        # make sure path is a Path object
        filename = Path(filename)
        # add .html if filename doesn't include it
        filename_html = filename.with_suffix(".html")
        plot.write_html(filename_html)
    
    def write_feature_basins(
            self, 
            nodes: list, 
            file_pre:str = "ELFCAR"
            ):
        """
        For a give list of nodes, writes the bader basins associated with
        each.
        """
        bader = self.bader
        graph = self.bifurcation_graph
        for node in nodes:
            basins = graph.nodes[node]["basins"]
            basin_labeled_voxels = bader.basin_labels.copy()
            charge_mask = np.isin(basin_labeled_voxels, basins)
            charge = bader.charge
            empty_grid = np.zeros(charge.shape)
            empty_grid[charge_mask] = charge[charge_mask]
            grid = Grid(self.structure, data={"total":empty_grid})
            grid.write_file(f"{file_pre}_{node}")
    
    def write_valence_basins(self):
        graph = self.bifurcation_graph
        bader = self.bader
        nodes = self._get_valence_summary(graph)
        self.write_feature_basins(bader, graph, nodes, file_pre="ELFCAR")