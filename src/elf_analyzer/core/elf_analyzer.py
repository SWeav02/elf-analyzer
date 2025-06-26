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

from elf_analyzer.core.utilities import UnionFind, BifurcationGraph, IonicRadiiTools#, find_connections



class ElfAnalyzer:
    """
    A class for finding electride sites from an ELFCAR.

    Args:
        grid : Grid
            A BadELF app Grid instance made from an ELFCAR.

        directory : Path
            Path the the directory to write files from
    """

    def __init__(
        self,
        elf_grid: Grid,
        charge_grid: Grid,
        directory: Path,
        separate_spin: bool = False,
        ignore_low_pseudopotentials: bool = False,
        downscale_resolution: int = 200,
        bader_method: str = None,
    ):
        self.elf_grid = elf_grid.copy()
        self.charge_grid = charge_grid.copy()
        self.directory = directory
        self.ignore_low_pseudopotentials = ignore_low_pseudopotentials
        self.bader_method = bader_method
        if downscale_resolution is not None:
            self.downscale_resolution = downscale_resolution
        else:
            self.downscale_resolution = elf_grid.voxel_resolution
        # check if this is a spin polarized calculation and if the user wants
        # to pay attention to this.
        if elf_grid.is_spin_polarized and separate_spin:
            self.spin_polarized = True
            self._elf_grid_up, self._elf_grid_down = elf_grid.split_to_spin()
            self._charge_grid_up, self._charge_grid_down = charge_grid.split_to_spin(
                "charge"
            )
        else:
            self.spin_polarized = False
            self._elf_grid_up, self._elf_grid_down = None, None
            self._charge_grid_up, self._charge_grid_down = None, None

    @property
    def structure(self) -> Structure:
        """
        Shortcut to grid's structure object
        """
        structure = self.elf_grid.structure.copy()
        structure.add_oxidation_state_by_guess()
        return structure

    @cached_property
    def atom_coordination_envs(self) -> list:
        """
        Gets the coordination environment for the atoms in the system
        using CrystalNN
        """
        cnn = CrystalNN(distance_cutoffs=None)
        neighbors = cnn.get_all_nn_info(self.structure)
        return neighbors

    @staticmethod
    def get_shared_feature_neighbors(structure: Structure) -> NDArray:
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

    def get_atom_en_diff_and_cn(self, site: int) -> list([float, int]):
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

    @property
    def site_voxel_coords(self) -> np.array:
        frac_coords = self.structure.frac_coords
        vox_coords = self.elf_grid.get_voxel_coords_from_frac(frac_coords)
        return vox_coords.astype(int)

    @cached_property
    def site_sphere_voxel_coords(self) -> list:
        site_sphere_coords = []
        for vox_coord in self.site_voxel_coords:
            nearby_voxels = self.elf_grid.get_voxels_in_radius(0.05, vox_coord)
            site_sphere_coords.append(nearby_voxels)
        return site_sphere_coords

    @cached_property
    def bader_up(self) -> Bader:
        """
        Returns a Bader object
        """
        if self.spin_polarized:
            return Bader(
                charge_grid=self._charge_grid_up, 
                reference_grid=self._elf_grid_up, 
                method=self.bader_method
                )
        else:
            return Bader(
                charge_grid=self.charge_grid, 
                reference_grid=self.elf_grid,
                method=self.bader_method
                )

    @cached_property
    def bader_down(self) -> Bader:
        """
        Returns a Bader object
        """
        if self.spin_polarized:
            return Bader(
                charge_grid=self._charge_grid_down, 
                reference_grid=self._elf_grid_down, 
                method=self.bader_method
                )
        else:
            return None
        
    def get_bifurcation_graphs(
        self,
        **cutoff_kwargs,
    ):
        """
        This will construct a bifurcation graph using a networkx
        DiGraph. Each node will contain information on whether it is
        reducible/irreducible, atomic/valent, etc.

        If the calculation is spin polarized, two graphs will be returned,
        one for each spin
        """
        if self.spin_polarized:
            elf_grid = self._elf_grid_up
            charge_grid = self._charge_grid_up
        else:
            elf_grid = self.elf_grid
            charge_grid = self.charge_grid
        # Get either the spin up graph or combined spin graph
        graph_up = self._get_bifurcation_graph(
            self.bader_up,
            elf_grid,
            charge_grid,
            **cutoff_kwargs,
        )
        if self.spin_polarized:
            # Check if there's any difference in each spin. If not, we only need
            # to run this once. We set the tolerance such that all ELF values must
            # be within 0.0001 of each other
            if np.allclose(
                self._elf_grid_up.total, self._elf_grid_down.total, rtol=0, atol=1e-4
            ):
                logging.info("Spin grids are found to be equal. Using spin up only.")
                graph_down = graph_up.copy()
            else:
                # Get the spin down graph
                graph_down = self._get_bifurcation_graph(
                    self.bader_down,
                    self._elf_grid_down,
                    self._charge_grid_down,
                    **cutoff_kwargs,
                )
            return graph_up, graph_down
        else:
            # We don't use spin polarized, so return the one graph.
            return graph_up
        
    def _get_important_elf_domains(
        self,
        elf_grid: Grid,
        bader: Bader,
        edge_mask: NDArray,
            ):
        """
        Scans through each bader basin and determines when they connect
        to the basins bordering them. Then determines the ELF values
        at which there are topological changes to the ELF isosurface.
        Returns a dictionary of ELF values and the basins in shared
        domains at that value.
        """
        # Extract Bader volumes and pad them to avoid edge issues
        basin_labels = bader.basin_labels
        elf_data = elf_grid.total

        # Get fractional coordinates of Bader maxima and convert to voxel indices
        maxima_frac_coords = bader.basin_maxima_frac
        maxima_voxel_coords = np.round(
            elf_grid.get_voxel_coords_from_frac(maxima_frac_coords)
        ).astype(int)

        # Get ELF values at the Bader maxima
        maxima_elf_values = elf_data[
            maxima_voxel_coords[:, 0],
            maxima_voxel_coords[:, 1],
            maxima_voxel_coords[:, 2]
        ]
        
        # # get the connection points for each basin
        # connections_array = find_connections(
        #     labeled_array=basin_labels,
        #     data=elf_data,
        #     basin_num=len(maxima_voxel_coords),
        #     neighbor_transforms=self.elf_grid.voxel_26_neighbors[0],
        #     )
        # # Find all connection pairs
        # connection_mask = connections_array > 0
        # connection_pairs = np.argwhere(connections_array)
        # connection_elfs = connections_array[connection_mask]
        # breakpoint()
        # Generate all 26 neighboring voxel shifts
        neighbor_shifts = list(itertools.product([-1, 0, 1], repeat=3))
        neighbor_shifts.remove((0, 0, 0))  # Remove the (0, 0, 0) self-shift

        # Get labels and elf values surrounding each voxel
        shifted_labels_list = []
        shifted_elf_list = []

        for dx, dy, dz in neighbor_shifts:
            rolled_labels = np.roll(basin_labels, shift=(dx, dy, dz), axis=(0, 1, 2))
            rolled_elf = np.roll(elf_data, shift=(dx, dy, dz), axis=(0, 1, 2))

            shifted_labels_list.append(rolled_labels)
            shifted_elf_list.append(rolled_elf)

        # Data for edge voxels
        edge_voxels = np.where(edge_mask)
        edge_basins = basin_labels[edge_voxels]
        edge_elf_values = elf_data[edge_voxels]

        # Gather neighbor labels and ELF values at the edge voxels
        neighbor_labels = [
            rolled_labels[edge_voxels] for rolled_labels in shifted_labels_list
        ]
        neighbor_elfs = [
            rolled_elf[edge_voxels] for rolled_elf in shifted_elf_list
        ]

        # Combine into 2D arrays where each row is an edge voxel
        neighbor_labels = np.column_stack(neighbor_labels)
        neighbor_elfs = np.column_stack(neighbor_elfs)

        # Analyze basin-basin connections through edges
        processed_basins = []
        connection_elfs = []
        connection_pairs = []

        for basin_idx in tqdm(range(len(maxima_frac_coords)), desc="Finding basin neighbors"):
            processed_basins.append(basin_idx)
            # Add connection to self
            connection_elfs.append(maxima_elf_values[basin_idx])
            connection_pairs.append([basin_idx, basin_idx])
            # Find edge voxels belonging to this basin
            is_edge_of_basin = edge_basins == basin_idx
            basin_edge_elfs = edge_elf_values[is_edge_of_basin]
            basin_neighbors = neighbor_labels[is_edge_of_basin]
            neighbor_edge_elfs = neighbor_elfs[is_edge_of_basin]

            # Identify unique neighbor basins, excluding self and previously processed ones
            mask_other_basins = basin_neighbors != basin_idx
            potential_neighbors = basin_neighbors[mask_other_basins]
            unique_neighbors = np.unique(potential_neighbors)
            new_neighbors = unique_neighbors[~np.isin(unique_neighbors, processed_basins)]

            for neighbor_idx in new_neighbors:
                # Mask to select only edge voxels neighboring this specific basin
                mask_neighbor = basin_neighbors == neighbor_idx

                # Extract ELF values at those edges
                neighbor_elfs_masked = np.zeros_like(neighbor_edge_elfs)
                neighbor_elfs_masked[mask_neighbor] = neighbor_edge_elfs[mask_neighbor]

                # Max ELF per voxel across 26 directions
                max_neighbor_elf = np.max(neighbor_elfs_masked, axis=1)

                # Compare with this basin's edge ELF and take the lower (bottleneck)
                bottleneck_elf = np.minimum(max_neighbor_elf, basin_edge_elfs)

                # Store the maximum bottleneck ELF as the connection value
                connection_elfs.append(bottleneck_elf.max())
                connection_pairs.append([basin_idx, neighbor_idx])
        # convert the connections to arrays for easy iteration
        connection_pairs = np.array(connection_pairs)
        connection_elfs = np.array(connection_elfs)
        # breakpoint()
        # Now we want to compile which domains exist at each elf value and the
        # basins they contain. We will scan over each maximum and connection
        # point and see if there is a change in domains
        possible_elf_values = list(np.unique(connection_elfs))
        possible_elf_values.reverse()
        # For each elf value, starting from the highest, we check which basins are connected
        # to each other to form a domain.
        important_values = {}
        # Add the value at the ELF slightly above where the last basin disappears
        connected_components = []
        for elf_value in tqdm(possible_elf_values, desc="Finding bifurcation elf values"):
            # Find the indices where connections are above the current value
            elf_indices = np.where(connection_elfs>=elf_value)[0]
            # get the connected basins
            connected_basins = connection_pairs[elf_indices]
            uf = UnionFind()
            for a, b in connected_basins:
                uf.union(a,b)
            # Get the previous and current groups
            previous_connected = connected_components.copy()
            connected_components = uf.groups()
            # Check that this list of sets is different from the previous one
            if not {frozenset(s) for s in previous_connected} == {frozenset(s) for s in connected_components}:
                important_values[elf_value] = connected_components
        # Our basins are in sets currently, but we want them to be arrays for operations
        # down the line. We convert them here
        important_values_new = {}
        for key, value in important_values.items():
            important_values_new[key] = [np.array(list(i)).astype(int) for i in value]
        important_values = important_values_new
        return important_values
    
    def _get_initial_graph(
            self,
            important_values: dict
            ):
        # Now that we have our elf values where changes occur, we want to generate our
        # initial graph
        graph = BifurcationGraph()
        # The elf values where topological changes happen are noted by the keys
        # of our dictionary
        keys = np.unique([i for i in important_values.keys()])
        # Our initial domain contains all of the basins and is stored in the
        # lowest key. We add this to our graph to avoid issues later in processing
        # due to it being the root.
        current_basin_groups = important_values[keys[0]]
        graph.add_node(1)
        networkx.set_node_attributes(
            graph,
            {1: {"basins": current_basin_groups[0],}}
            )
        # Now we loop over the ELF values at which bifurcations occur or maxima
        # exist
        node_count = 1
        for key in tqdm(keys[1:], desc="Constructing graph"):
            # Get the current and previous groups for comparison
            previous_basin_groups = current_basin_groups.copy()
            current_basin_groups = important_values[key]
            for basin_group in current_basin_groups:
                basin_group = basin_group
                # we check if this basin group exists in the previous one. If it
                # does, we've already added a node for this group and continue
                old_group = any(np.array_equal(basin_group, other_group) for other_group in previous_basin_groups)
                if old_group:
                    continue
                # otherwise, this is a new group and we want to add a node representing it.
                # We also want to find node that should be the parent of this one
                # so that we can assign an edge and label the value at which the
                # parent split. This corresponds to the most recent node that
                # had a group containing all of this group.
                nodes = list(graph.nodes)
                nodes.reverse()
                parent_found = False
                for node in nodes:
                    if np.all(np.isin(basin_group, graph.nodes[node]["basins"])):
                        parent_node = node
                        parent_found = True
                        break
                if not parent_found:
                    breakpoint()
                # We've now found our parent and we want to update it's split value
                networkx.set_node_attributes(graph,{parent_node: {"split": key}})
                # Now we update our node count and add the current node
                node_count += 1
                graph.add_node(node_count)
                graph.add_edge(parent_node, node_count)
                networkx.set_node_attributes(graph,{node_count: {"basins": basin_group}})
        return graph
    
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
        logging.info(
            "Generating initial bifurcation graph."
            )
        # Also get an array labeling which bader basin each voxel is assigned to
        basin_labeled_voxels = bader.basin_labels.copy()

        # We will use a downscaled version of our ELF for speed in some cases
        downscale_resolution = self.downscale_resolution
        if elf_grid.voxel_resolution > downscale_resolution:
            downscaled_elf_grid = elf_grid.regrid(downscale_resolution)
            label_grid = elf_grid.copy()
            label_grid.total = basin_labeled_voxels
            downscaled_label_grid = label_grid.regrid(downscale_resolution, order=0)
        else:
            downscaled_elf_grid = elf_grid.copy()
            downscaled_label_grid = elf_grid.copy()
            downscaled_label_grid.total = basin_labeled_voxels
        downscaled_basin_labeled_voxels = downscaled_label_grid.total
        
        edge_mask = bader.basin_edges
        # We don't use a downscaled ELF here to ensure that we find all possible
        # maxima
        important_elf_values = self._get_important_elf_domains(
            elf_grid=elf_grid,
            bader=bader,
            edge_mask=edge_mask,
            )
        
        # get an initial graph connecting bifurcations and final basins
        graph = self._get_initial_graph(important_values=important_elf_values)
        
        logging.info("Labeling elf domains")
        # Get properties for each domain
        graph = self._get_node_properties(
            graph = graph,
            elf_grid = elf_grid,
            downscaled_elf_grid = downscaled_elf_grid,
            charge_grid = charge_grid,
            bader = bader,
            basin_labeled_voxels = basin_labeled_voxels,
            downscaled_basin_labeled_voxels = downscaled_basin_labeled_voxels,
            )
        
        # First, we clean up the graph in case we removed a node earlier due
        # to incorrect labeling and this resulted in a fake split (e.g. Dy2C)
        graph = self._clean_reducible_nodes(graph)
        # Now we have a graph with information associated with each basin. We want
        # to label each node.
        # graph = self._mark_atomic(graph, downscaled_basin_labeled_voxels, downscaled_elf_grid, shell_depth)
        graph = self._mark_atomic(graph, basin_labeled_voxels, elf_grid, shell_depth)
        # Now we want to label our valence features as Covalent, Metallic, or bare electron.
        # Many covalent and metallic features are easy to find. Covalent bonds
        # are typically exactly along a bond between an atom and its nearest
        # neighbors. Metallic features have a low depth. We mark these first
        graph = self._mark_covalent_lonepair(
            graph,
            min_covalent_charge=min_covalent_charge,
            min_covalent_angle=min_covalent_angle,
            min_covalent_bond_ratio=min_covalent_bond_ratio,
        )
        
        # Now that we have a sense of which features are covalent/lone-pairs
        # we want to correct for a few possible errors in our assignments.
        # Sometimes if we've set our shell depth too low we will end up with only
        # "lone-pairs" surrounding an atom. We relabel these as shells.
        graph = self._correct_for_high_depth_shells(graph)
        
        # Sometimes a metallic/bare electron will detatch from an atomic basin
        # rather than a valence domain. These will be misassigned as shells. We
        # correct for these here by looking for shell basins outside the atoms
        # radius. We need to run a new bader with the labeled structure for this
        # so we also take advantage of the moment to assign radii to the features
        # Note we don't use downscaled grid here
        radii = self._get_atomic_radii(
            graph, 
            bader, 
            elf_grid, 
            radius_refine_method
            )
        
        graph = self._correct_far_shell_features(
            graph=graph,
            radii=radii,
            )
        
        # Reduce any related shell basins to a single basin for clarity
        if combine_shells:
            graph = self._reduce_atomic_shells(graph)
        
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
    
    def _get_node_properties(
            self, 
            graph: BifurcationGraph(),
            elf_grid: Grid,
            downscaled_elf_grid: Grid,
            charge_grid: Grid,
            bader: Bader,
            basin_labeled_voxels: NDArray,
            downscaled_basin_labeled_voxels: NDArray,
            ) -> BifurcationGraph():
        # Now loop over this graph and label each node with important information
        for node_idx in tqdm(graph.nodes, desc="Calculating node properties"):
            attributes = graph.nodes[node_idx]
            parent_attributes = graph.parent_dict(node_idx)
            if "split" in attributes.keys():
                # this is a reducible domain. We want to get the atoms contained in
                # this domain when it first appeared, as well as whether it was
                # an infinite connection right before it split
                if parent_attributes is not None:
                    parent_split = parent_attributes["split"] - 0.01
                    basins = graph.nodes[node_idx]["basins"]
                    low_elf_mask = np.isin(downscaled_basin_labeled_voxels, basins) & np.where(
                        downscaled_elf_grid.total > parent_split, True, False
                    )
                    high_elf_mask = np.isin(
                        downscaled_basin_labeled_voxels, basins
                    ) & np.where(
                        downscaled_elf_grid.total > (attributes["split"] - 2 * 0.01), True, False
                    )
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
                
                atom_num = len(atoms)
                if is_infinite:
                    atom_num = -1
                
                networkx.set_node_attributes(
                    graph,
                    {
                        node_idx: {
                            "split": attributes["split"],
                            "num": len(graph.child_indices(node_idx)),
                            "atoms": atoms,
                            "atom_num": atom_num,
                        }
                    },
                )
            else:
                # This is an irreducible domain.
                # We want to store data relavent to the type of domain it might
                # be.
                # First we get a mask representing where this feature is
                basins = attributes["basins"]
                basin_mask = np.isin(basin_labeled_voxels, basins)
                max_elf = np.max(elf_grid.total[basin_mask])
                split = parent_attributes["split"]
                depth = round(max_elf - split, 4)
                # We also want to mark a type of depth corresponding to the
                # point where this feature connected with an infinite domain.
                all_parent_indices = graph.deep_parent_indices(node_idx)
                for idx in all_parent_indices:
                    current_parent = graph.nodes[idx]
                    if current_parent["atom_num"] == -1:
                        infinite_split = current_parent["split"]
                        break
                depth_3d = round(max_elf - infinite_split, 4)
                # Using this, we can find the average frac coords of the attractors
                # in this basin
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
                    networkx.set_node_attributes(
                        graph,
                        {
                            node_idx: {
                                "max_elf": round(max_elf, 4),
                                "depth": depth,
                                "3d_depth": depth_3d,
                                "charge": charge,
                                "volume": volume,
                                "atom_distance": round(distance, 4),
                                "nearest_atom": nearest_atom,
                                "nearest_atom_type": self.structure[nearest_atom].specie.symbol,
                                "frac_coords": frac_coord,
                            }
                        },
                    )
                except:
                    breakpoint()
        return graph
    
    def get_valence_summary(self, graph: BifurcationGraph()) -> dict:
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

    def _mark_atomic(
        self,
        graph: BifurcationGraph(),
        basin_labeled_voxels,
        elf_grid,
        shell_depth: float = 0.05,
    ) -> BifurcationGraph():
        elf_data = elf_grid.total
        # create a variable to track the number of atoms left to assign
        remaining_atoms = len(self.structure)
        # BUG: The remaining atom count is broken currently. Sometimes atoms are
        # double counted, e.g. when a core feature breaks off before another feature
        # that fully surround the atom.
        for i in tqdm(graph.nodes, desc="Marking atomic nodes"):
            # Get the dict of information for our node and the parent of our node
            node = graph.nodes[i]
            # We are going to use attributes of each irreducible feature to
            # assign its children, so if this node isn't irreducible we skip it
            if not "split" in node.keys():
                continue
            # We also label this node with how many atoms are left to assign
            networkx.set_node_attributes(
                graph, {i: {"remaining_atoms": remaining_atoms}}
            )
            # There are three situations for our reducible feature. First, if
            # it surrounds 0 atoms then all of its children must be valence. We
            # skip in this case
            num_atoms = node["atom_num"]
            if num_atoms == 0:  # or remaining_atoms == 0:
                # Label all children as valence
                for child_idx, child in graph.child_dicts(i).items():
                    # skip an reducible features
                    if "split" in child.keys():
                        continue
                    # We sometimes label the nodes of reducible features as covalent.
                    # We don't want to overwrite these so we check that the subtype
                    # doesn't exist
                    elif child.get("subtype") is None:
                        networkx.set_node_attributes(
                            graph, {child_idx: {"type": "val", "subtype": None}}
                        )
                continue
            # Second, it can contain more than one atom. In a full core model,
            # The atoms that split off of this type of feature would themselves
            # be reducible and always fit into the next category. However, with
            # a pseudopotential model, this is not the case. Instead, an atom
            # may only have a single irreducible feature. We check for this by
            # noting if the child features fully surround an atom at the ELF they separate at
            # NOTE: -1 atoms really indicates infinite
            # TODO: It may be that this loop should just be for when the number
            # of atoms is infinite. Basically, any finite number suggests a
            # molecular feature and all basins would be core/shell/covalent/lone-pair.
            elif num_atoms == -1:
                for child_idx, child in graph.child_dicts(i).items():
                    # skip any nodes that are reducible
                    if "split" in child.keys():
                        continue
                    # Get the basins that belong to this child
                    basins = child["basins"]
                    # Using these basins, and the value the basin split at, we
                    # get a mask for the location of the basin
                    parent_split = node["split"]
                    low_elf_mask = np.isin(basin_labeled_voxels, basins) & np.where(
                        elf_data > parent_split, True, False
                    )
                    atoms_in_basin = elf_grid.get_atoms_in_volume(low_elf_mask)
                    basin_type = "val"
                    basin_subtype = None
                    if len(atoms_in_basin) > 0:
                        basin_type = "atom"
                        basin_subtype = "core"
           
                        # Note that we found a new atom
                        remaining_atoms -= 1
                    # label this basin
                    networkx.set_node_attributes(
                        graph,
                        {child_idx: {"type": basin_type, "subtype": basin_subtype}},
                    )
            # The final option is that our reducible region surrounds a finite
            # number of atoms. Most of the subregions of this
            # environment will be atomic, but they can be of several types including
            # atom shells/cores, unshared electrons, lone-pairs. The one exception
            # is heterogenous covalent bonds, which should be shared.
            elif num_atoms > 0:                    
                # Otherwise, these features are atomic, shells, or covalent/lone-pairs
                # Now we loop over all of the children of this feature, including
                # deeper children. We label these children based on their depth
                # and whether they surround the atom. We label features as:
                # core, shell, or other.
                # The "others" will be assigned later on as lone-pairs or covalent
                # depending on if they are along an atomic bond
                for child_idx, child in graph.deep_child_dicts(i).items():
                # for child_idx in important_children:
                    child = graph.nodes[child_idx]
                    # define our default types
                    basin_type = "atom"
                    basin_subtype = None
                    # If we have a split, we don't want to label this node so
                    # we continue.
                    if "split" in child.keys():
                        continue
                    # If we have many shell basins that form a sphere around the
                    # atom they may separate at a low depth. However, lone-pairs
                    # that are highly symmetric may also separate in a similar way.
                    # We actually want the depth to the point where the basin connects
                    # to a reducible domain surrounding the atom of interest. This is
                    # the point where this node split.
                    basin_shell_depth = child["max_elf"] - node["split"]

                    # if child["depth"] < shell_depth:
                    if basin_shell_depth < shell_depth:
                        basin_subtype = "shell"
                    else:
                        # otherwise, we check if the feature surrounds an atom
                        # Get the basins that belong to this child
                        basins = child["basins"]
                        # Using these basins, and the value the basin split at, we
                        # get a mask for the location of the basin
                        child_parent = graph.parent_dict(child_idx)
                        parent_split = child_parent["split"]
                        low_elf_mask = np.isin(basin_labeled_voxels, basins) & np.where(
                            elf_data > parent_split, True, False
                        )
                        atoms_in_basin = elf_grid.get_atoms_in_volume(low_elf_mask)
                        
                        if len(atoms_in_basin) > 0:
                            # We have an core region
                            basin_subtype = "core"
                        else:
                            # otherwise its an other
                            basin_type = "val"
                            basin_subtype = "other"
                    # Now we assign our types to the child node.
                    networkx.set_node_attributes(
                        graph,
                        {child_idx: {"type": basin_type, "subtype": basin_subtype}},
                    )

        return graph
    
    def _get_atomic_radii(
            self, 
            graph: BifurcationGraph(), 
            bader: Bader ,
            elf_grid: Grid,
            radius_refine_method: str,
            ):
        valence_summary = self.get_valence_summary(graph)
        # We will need to get radii from the ELF. To do this, we need a labeled
        # pybader result to pass to our PartitioningToolkit
        frac_coords = bader.basin_maxima_frac
        temp_structure = self.structure.copy()
        for feature_idx, attributes in valence_summary.items():
            if attributes["subtype"] == "covalent":
                species = "Z"
            elif attributes["subtype"] == "lone-pair":
                # species = "Lp"
                # We want to consider lone-pairs as part of the atom so we continue
                continue
            else:
                species = "X"
            for basin_idx in attributes["basins"]:
                frac_coord = frac_coords[basin_idx]
                temp_structure.append(species, frac_coord)
        
        # recalculate the atoms for our bader object
        bader_labeled = bader.copy()
        bader_labeled.run_atom_assignment(structure=temp_structure)

        partitioning = IonicRadiiTools(elf_grid, bader_labeled)
        # TODO Ideally, these radii are stored at a class level so that they
        # can be passed to the BadElfToolkit class for summary. However, this
        # requires knowledge of if this is spin-up/spin-down which I currently
        # don't have stored at this level
        radii = partitioning.get_elf_ionic_radii(
            refine_method=radius_refine_method, labeled_structure=temp_structure
        )
        return radii
    
    def _correct_far_shell_features(
            self,
            graph: BifurcationGraph(),
            radii,
            ) -> BifurcationGraph():
        """
        Corrects any shell nodes that are outside the radius of the atom
        to be considered bare electrons instead
        """
        
        for node in graph.nodes:
            attributes = graph.nodes[node]
            if attributes.get("subtype",None) != "shell":
                continue
            atom = attributes.get("nearest_atom")
            atom_radius = radii[atom]
            distance = attributes.get("atom_distance")
            tolerance = 0.0
            if distance > atom_radius + tolerance:
                # This shouldn't be considered a shell basin and we relabel it
                # We also need to find the radius of this feature to match what we
                # had before

                networkx.set_node_attributes(
                    graph,
                    {node: {"type": "val", "subtype": "bare electron"}},
                )   
        
        return graph
            
        
    
    def _correct_for_high_depth_shells(
        self,
        graph: BifurcationGraph(),
    ) -> BifurcationGraph():
        """
        Sometimes atomic shells have particularly deep separations, for
        example when they are heavily polarized (e.g. Er2C). In these
        cases, the shell will split into one irreducible domain and
        one or more reducible domains. This is similar to a covalent bond/
        lone-pair shell. However, none of the domains will fit the criteria
        for a covalent bond, so all of them will be marked as shells or
        lone-pairs. We change all of them to be marked as shells here.
        """
        for i in graph.nodes:
            # Get the dict of information for our node and the parent of our node
            node = graph.nodes[i]
            # skip irreducible domains
            if not "split" in node.keys():
                continue
            num_atoms = node["atom_num"]
            # We check only for situations where we have a finite number of
            # atoms in a reducible region
            if num_atoms > 0:
                all_lone_pairs_or_shells = True
                for child_idx, child in graph.deep_child_dicts(i).items():
                    # skip reducible domains
                    if "split" in child.keys():
                        continue
                    if child["subtype"] not in ["lone-pair", "shell"]:
                        all_lone_pairs_or_shells = False
                        break
                if not all_lone_pairs_or_shells:
                    # This reducible domain isn't a shell. Continue
                    continue
                for child_idx, child in graph.deep_child_dicts(i).items():
                    # skip reducible domains
                    if "split" in child.keys():
                        continue
                    networkx.set_node_attributes(
                        graph,
                        {child_idx: {"type": "atom", "subtype": "shell"}},
                    )

        return graph

    def _combine_shells(
        self, graph: BifurcationGraph(), nodes: list[int]
    ) -> BifurcationGraph():
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
        for child_idx in nodes:
            child = graph.nodes[child_idx]
            nearest_atom = child["nearest_atom"]
            nearest_atom_type = child["nearest_atom_type"]
            basins.extend(child["basins"])
            atom_distance = min(atom_distance, child["atom_distance"])
            volume += child["volume"]
            charge += child["charge"]
            max_elf = max(max_elf, child["max_elf"])
            frac_coords = child["frac_coords"]
            depth = max(depth, child["depth"])
            depth_3d = max(depth_3d, child["3d_depth"])

        # clear the attributes from the first node
        graph.nodes[nodes[0]].clear()
        # Add the attributes
        networkx.set_node_attributes(
            graph,
            {
                nodes[0]: {
                    "type": "atom",
                    "subtype": "shell",
                    "basins": basins,
                    "atom_distance": round(atom_distance, 4),
                    "volume": round(volume, 4),
                    "charge": round(charge, 4),
                    "max_elf": round(max_elf, 4),
                    "nearest_atom": nearest_atom,
                    "nearest_atom_type": nearest_atom_type,
                    "depth": round(depth, 4),
                    "3d_depth": depth_3d,
                    "frac_coords": frac_coords,
                }
            },
        )
        children_to_remove = nodes[1:]
        # delete all of the unused nodes
        for j in children_to_remove:
            graph.remove_node(j)
        return graph

    def _reduce_atomic_shells(
        self,
        graph: BifurcationGraph(),
    ) -> BifurcationGraph():
        """
        Reduces shell nodes to a single node
        """
        # We want to combine any nodes that belong to the same atomic shell. We
        # can do this by confirming that they share 2 aspects: The same closest
        # atom and a similar distance to the atom. To do this, we create a dictionary
        # to store these two attributes and the associated shells
        shell_groups = {}
        reducible_nodes = []
        group_num = 0
        for i in graph.nodes:
            node = graph.nodes[i]
            if "split" in node.keys():
                reducible_nodes.append(i)
            if node.get("subtype", None) != "shell":
                continue
            # First we get the shells nearest atom and distance
            atom = node["nearest_atom"]
            dist = node["atom_distance"]
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
                shell_groups[assigned_group]["nodes"].append(i)
            else:
                shell_groups[group_num] = {
                    "atom" : atom,
                    "dists" : np.array([dist]),
                    "nodes" : [i],
                    }
                group_num += 1
        
        # Now we want to go through and combine all of the shells we just grouped
        for group, values in shell_groups.items():
            nodes = values["nodes"]
            graph = self._combine_shells(graph, nodes)
        
        # Now that we've done that, there may be some nodes that were parents
        # of these shells that are either empty or a parent to one newly grouped
        # shell. We loop over the potential parents backwards, deleting any that
        # have no children and replacing any that have only one child
        reducible_nodes = list(np.unique(reducible_nodes))
        reducible_nodes.reverse()
        for parent in reducible_nodes:
            children = graph.child_indices(parent)
            child_num = len(children)
            if child_num == 0:
                # This is an empty node and we just delete it.
                graph.remove_node(parent)
            elif child_num == 1:
                # This node is now the parent of a single shell feature. We replace
                # it.
                child_dict = graph.nodes[children[0]]
                # recalculate depth
                parent_dict = graph.nodes[parent]
                parent_elf = parent_dict["split"]
                depth = child_dict["max_elf"] - parent_elf

                # clear the attributes from the first node
                graph.nodes[parent].clear()
                # Add the attributes
                networkx.set_node_attributes(
                    graph,
                    {
                        parent: {
                            "type": child_dict["type"],
                            "subtype": child_dict["subtype"],
                            "basins": child_dict["basins"],
                            "atom_distance": child_dict["atom_distance"],
                            "volume": child_dict["volume"],
                            "charge": child_dict["charge"],
                            "max_elf": child_dict["max_elf"],
                            "nearest_atom": child_dict["nearest_atom"],
                            "nearest_atom_type": child_dict["nearest_atom_type"],
                            "depth": round(depth, 4),
                            "3d_depth": child_dict["3d_depth"],
                            "frac_coords": child_dict["frac_coords"],
                            "reducible": True,
                        }
                    },
                )
                # delete the child node
                graph.remove_node(children[0])

        return graph

    def _mark_covalent_lonepair(
        self,
        graph: BifurcationGraph(),
        min_covalent_charge: float = 0.6,
        min_covalent_angle: float = 135,
        min_covalent_bond_ratio: float = 0.4,
    ) -> BifurcationGraph():
        """
        Takes in a bifurcation graph and labels valence features that
        are obviously metallic or covalent
        """
        valence_summary = self.get_valence_summary(graph)
        # TODO: Many of these features could be symmetric. I should only perform
        # each action for one of these symmetric features and assign the result
        # to all of them.
        for feature_idx, attributes in tqdm(valence_summary.items(), desc="Marking covalent and lone-pair nodes"):
            previous_subtype = attributes.get("subtype")
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
            frac_coords = attributes["frac_coords"]
            temp_structure = self.structure.copy()
            temp_structure.append("X", frac_coords)
            nearest_atom = attributes["nearest_atom"]
            atom_dist = round(temp_structure.get_distance(nearest_atom, -1), 2)
            atom_neighs = self.atom_coordination_envs[nearest_atom]
            # We want to see if our feature lies directly between our atom and
            # any of its neighbors.
            covalent = False
            # If we're above our charge cutoff, we check if we are along a bond
            if attributes["charge"] > min_covalent_charge:
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
            networkx.set_node_attributes(
                graph, {feature_idx: {"type": basin_type, "subtype": subtype}}
            )

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
        def get_molecule_parent(idx):
            # get parent that fully surrounds at least one atom
            molecule_parent_idx = -1
            parent_idx = graph.parent_index(idx)
            while molecule_parent_idx == -1:
                current_parent = graph.nodes[parent_idx]
                if current_parent["atom_num"] != 0:
                    molecule_parent_idx = parent_idx
                else:
                    parent_idx = graph.parent_index(parent_idx)
            return molecule_parent_idx

        features_to_relabel = []
        for feature_idx, attributes in valence_summary.items():
            if attributes.get("subtype") == "bare electron":

                all_cov_lp_be = True
                at_least_one_cov = False
                molecule_parent_idx = get_molecule_parent(feature_idx)
                # for sibling_idx, sibling in graph.sibling_dicts(feature_idx).items():
                # Check if all siblings are covalent, bare electrons, or lone-pairs. If so,
                # this is a lone-pair
                for sibling_idx, sibling in graph.deep_child_dicts(
                    molecule_parent_idx
                ).items():
                    # make sure this sibling isn't the child of a different submolecule
                    direct_parent_idx = get_molecule_parent(sibling_idx)
                    direct_parent = graph.nodes[direct_parent_idx]
                    if (
                        direct_parent["atom_num"] != 0
                        and direct_parent_idx != molecule_parent_idx
                    ):
                        continue
                    if "split" in sibling.keys():
                        continue
                    # We need to make sure there's at least one covalent bond as well
                    if sibling["subtype"] == "covalent":
                        at_least_one_cov = True
                    elif sibling["subtype"] not in [
                        "bare electron",
                        "covalent",
                        "lone-pair",
                    ]:
                        all_cov_lp_be = False
                if all_cov_lp_be and at_least_one_cov:
                    features_to_relabel.append(feature_idx)
        for feature_idx in features_to_relabel:
            networkx.set_node_attributes(
                graph, {feature_idx: {"type": "val", "subtype": "lone-pair"}}
            )

        return graph
    
    def _mark_feature_radii(
            self,
            graph: BifurcationGraph(),
            bader: Bader,
            ):
        basin_radii = bader.basin_surface_distances
        valence_summary = self.get_valence_summary(graph)
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
        valence_summary = self.get_valence_summary(graph)

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
            if self.spin_polarized:
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
        valence_summary = self.get_valence_summary(graph)
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
        valence_summary = self.get_valence_summary(graph)
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
            

    def _clean_reducible_nodes(self, graph: BifurcationGraph()) -> BifurcationGraph():

        nodes_to_remove = []
        for i in graph.nodes:
            node = graph.nodes[i]
            # skip irreducible nodes
            if not "split" in node.keys():
                continue
            children = graph.child_indices(i)
            # check if we only have one child
            if len(children) != 1:
                continue
            # check if this child is reducible
            child = graph.nodes[children[0]]
            if not "split" in child.keys():
                continue
            # If we made it to this point, we have a single reducible child under
            # this reducible node. We want to remove the child and change the
            # connections
            nodes_to_remove.append(children[0])

        # now remove each child
        nodes_to_remove.reverse()
        for child_idx in nodes_to_remove:
            child = graph.nodes[child_idx]
            i = graph.parent_index(child_idx)
            edge_companions = []
            for edge in graph.edges:
                if child_idx == edge[0]:
                    edge_companions.append(edge[1])
            # get the features to update on this node
            split = child["split"]
            num = child["num"]
            networkx.set_node_attributes(
                graph,
                {i: {"split": split, "num": num}},
            )
            # delete the child node
            graph.remove_node(child_idx)
            # add back connections
            for edge_companion in edge_companions:
                graph.add_edge(i, edge_companion)
        return graph

    def get_bifurcation_plots(
        self,
        write_plot: bool = False,
        plot_name="bifurcation_plot.html",
        **cutoff_kwargs,
    ):
        """
        Plots bifurcation plots automatically. Graphs will be generated
        using the provided resolution. If the provided
        ELF and Charge Density are spin polarized, two plots will be
        generated.
        """
        # remove .html if its at the end of the plot name
        plot_name = plot_name.replace(".html", "")

        if self.spin_polarized:
            graph_up, graph_down = self.get_bifurcation_graphs(**cutoff_kwargs)
            plot_up = self.get_bifurcation_plot(
                graph_up, write_plot, f"{plot_name}_up.html"
            )
            plot_down = self.get_bifurcation_plot(
                graph_down, write_plot, f"{plot_name}_down.html"
            )
            return plot_up, plot_down
        else:
            graph = self.get_bifurcation_graphs(**cutoff_kwargs)
            return self.get_bifurcation_plot(graph, write_plot, plot_name)

    def get_bifurcation_plot(
        self,
        graph: BifurcationGraph(),
        write_plot=False,
        plot_name="bifurcation_plot.html",
    ):
        """
        Returns a plotly figure
        """
        # remove .html if its at the end of the plot name
        plot_name = plot_name.replace(".html", "")
        # then add .html to ensure its there
        plot_name += ".html"

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

        if write_plot:
            fig.write_html(self.directory / plot_name)
        return fig

    def get_labeled_structures(
        self,
        include_lone_pairs: bool = False,
        include_shared_features: bool = True,
        **cutoff_kwargs,
    ):
        """
        Returns a structure with dummy atoms at electride and shared
        electron sites. Off atom basins are assigned using the bifurcation
        graph method. See the .get_bifurcation_plot method for more info.

        Dummy atoms will have the following labels:
        e: electride, le: bare electron, m: metallic feature, z: covalent bond,
        lp: lone-pair

        If spin porized files are provided, returns two structures.
        """
        if self.spin_polarized:
            graph_up, graph_down = self.get_bifurcation_graphs(
                **cutoff_kwargs,
            )
            structure_up = self._get_labeled_structure(
                graph_up,
                include_lone_pairs,
                include_shared_features,
            )
            structure_down = self._get_labeled_structure(
                graph_down,
                include_lone_pairs,
                include_shared_features,
            )
            return structure_up, structure_down
        else:
            graph = self.get_bifurcation_graphs(
                **cutoff_kwargs,
            )
            return self._get_labeled_structure(
                graph,
                include_lone_pairs,
                include_shared_features,
            )

    def _get_labeled_structure(
        self,
        graph: BifurcationGraph(),
        include_lone_pairs: bool = False,
        include_shared_features: bool = True,
        **kwargs,
    ):
        # First, we get the valence features for this graph and create a
        # structure that we will add features to
        valence_features = self.get_valence_summary(graph)
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

        logging.info(f"{len(electride_indices)} electride sites found")
        if len(other_indices) > 0:
            f"{len(other_indices)} shared sites found"

        return sorted_structure

    @classmethod
    def from_vasp(
        cls,
        directory: Path = Path("."),
        elf_file: str = "ELFCAR",
        charge_file: str = "CHGCAR",
        **kwargs,
    ):
        """
        Creates a BadElfToolkit instance from the requested partitioning file
        and charge file.

        Args:
            directory (Path):
                The path to the directory that the badelf analysis
                will be located in.
            elf_file (str):
                The filename of the file containing the ELF. Must be a VASP
                ELFCAR type file.
            charge_file (str):
                The filename of the file containing the charge information. Must
                be a VASP CHGCAR file.
            **kwargs:
                Any other keyword arguments to pass to the ElfAnalysisToolkit

        Returns:
            A ElfAnalyzerToolkit instance.
        """

        elf_grid = Grid.from_vasp(directory / elf_file)
        charge_grid = Grid.from_vasp(directory / charge_file)
        return cls(
            elf_grid=elf_grid,
            charge_grid=charge_grid,
            directory=directory,
            **kwargs,
        )

    def get_full_analysis(self, write_results: bool = True, **kwargs):
        """
        Gets the BifurcationGraphs, plots, and labeled structures for
        the entire analysis and returns them as a dict object.
        """
        if self.spin_polarized:
            graph_up, graph_down = self.get_bifurcation_graphs(**kwargs)
            # bader_up, bader_down = self.bader_up, self.bader_down
            plot_up = self.get_bifurcation_plot(
                graph_up, write_plot=write_results, plot_name="bifurcation_plot_up"
            )
            plot_down = self.get_bifurcation_plot(
                graph_down, write_plot=write_results, plot_name="bifurcation_plot_down"
            )
            structure_up = self._get_labeled_structure(graph_up, **kwargs)
            structure_down = self._get_labeled_structure(graph_down, **kwargs)
            if write_results:
                # write structures
                structure_up.to(self.directory / "labeled_up.cif", "cif")
                structure_down.to(self.directory / "labeled_down.cif", "cif")

            return {
                "graph_up": graph_up,
                "graph_down": graph_down,
                "plot_up": plot_up,
                "plot_down": plot_down,
                "structure_up": structure_up,
                "structure_down": structure_down,
            }

        else:
            graph = self.get_bifurcation_graphs(**kwargs)
            # bader = self.bader_up
            plot_name = "bifurcation_plot"
            if "plot_name" in kwargs.keys():
                plot_name = kwargs["plot_name"]
            plot = self.get_bifurcation_plot(
                graph, write_plot=write_results, plot_name=plot_name
            )
            structure = self._get_labeled_structure(graph, **kwargs)
            if write_results:
                # write structures
                structure.to(self.directory / "labeled.cif", fmt="cif")
            return {
                "graph": graph,
                "plot": plot,
                "structure": structure,
            }
    
    def write_feature_basins(
            self, 
            bader: Bader, 
            graph: BifurcationGraph(), 
            nodes: list, 
            file_pre:str = "ELFCAR"
            ):
        """
        For a give list of nodes, writes the bader basins associated with
        each.
        """
        for node in nodes:
            basins = graph.nodes[node]["basins"]
            basin_labeled_voxels = bader.basin_labels.copy()
            charge_mask = np.isin(basin_labeled_voxels, basins)
            charge = bader.charge
            empty_grid = np.zeros(charge.shape)
            empty_grid[charge_mask] = charge[charge_mask]
            grid = Grid(self.structure, data={"total":empty_grid})
            grid.write_file(f"{file_pre}_{node}")
    
    def write_valence_basins(self, results: dict):
        if self.spin_polarized:
            graph_down = results["graph_down"]
            bader_down = self.bader_down
            nodes_down = self.get_valence_summary(graph_down)
            self.write_feature_basins(bader_down, graph_down, nodes_down, file_pre="ELFCAR_down")
            # get graph for spin up
            graph_up = results["graph_up"]
            bader_up = self.bader_up
            nodes_up = self.get_valence_summary(graph_up)
            self.write_feature_basins(bader_up, graph_up, nodes_up, file_pre="ELFCAR_up")
        
        else:
            graph = results["graph"]
            bader = self.bader_up
            nodes = self.get_valence_summary(graph)
            self.write_feature_basins(bader, graph, nodes, file_pre="ELFCAR")