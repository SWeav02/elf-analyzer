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
from typing import TypeVar
from rich.progress import track
import time

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure

from baderkit.core import Grid, Bader
from baderkit.core.toolkit import Format

# from elf_analyzer.core.utilities import BifurcationGraph, IonicRadiiTools
from elf_analyzer.core.utilities.numba_functions import check_all_covalent

from elf_analyzer.core.bifurcation_graph import BifurcationGraph

Self = TypeVar("Self", bound="ElfAnalyzer")
# TODO:
    # - Update graph class to be more useful to specifically bifurcation plot
    # - Make updates to selection rules:
        # 1. Core features should have maxima within one voxel of an atom
        # 2. Shells should have a low depth to a node containing EXACTLY 1 atom.
        # That node should be the same for all shells, so shallow metal features
        # that split at an earlier stage are excluded. Also check up on the shell
        # depth correction I currently have. Maybe those really should be assigned
        # as something else
        # 3. Covalent/lone-pair features don't need to be in a finite molecule.
        # e.g. they may be in a polymer. Instead, covalent bonds should be found
        # using charge, angle, and bond ratio cutoffs. (is charge even necessary if we've already found shells?)
        # Lone pairs are then any features at approximately the same radius as the
        # covalent bond.
    # - For now I shouldn't label electrides by default. Instead I'll add a
    # convenience method for getting an electride structure. That way the Simmate
    # workflows can still function 
    # - create method to get all radii around each atom using new radii method:
        # 1. get atom neighbors (up to distance or dynamically)
        # 2. find radii for closest and add to official neighbors
        # 3. go to next closest and get radii. See if they fit under planes made
        # by other radii
        # 4. Continue until all neighbors found. (Whats the best stop condition?)
    # - rework BadELF voxel assignment using Numba. That should just be so much
    # faster. I could probably even reimplement exact voxel splitting
    # - Update simmate workflows (and badelf -_- )

    
# TODO: add convenience properties/methods for getting information about different
# assignments

class ElfAnalyzer(Bader):
    """
    A class for finding electride sites from an ELFCAR.
    """
    
    _spin_polarized = False

    def __init__(
        self,
        ignore_low_pseudopotentials: bool = False,
        shared_shell_ratio: float = 1/3,
        combine_shells: bool = True,
        min_covalent_charge: float = 0.6,
        min_covalent_angle: float = 135,
        electride_elf_min: float = 0.5,
        electride_depth_min: float = 0.2,
        electride_charge_min: float = 0.5,
        electride_volume_min: float = 10,
        electride_radius_min: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        # ensure the reference file is ELF
        if self.reference_grid.data_type != "elf":
            logging.warning("A non-ELF reference file has been detected. Results may not be valid.")
        
        self.ignore_low_pseudopotentials = ignore_low_pseudopotentials
        
        # define cutoff variables
        # TODO: These should be hidden variables to allow for setter methods
        self.shared_shell_ratio = shared_shell_ratio
        self.combine_shells = combine_shells
        self.min_covalent_charge = min_covalent_charge
        self.min_covalent_angle = min_covalent_angle
        self.electride_elf_min = electride_elf_min
        self.electride_depth_min = electride_depth_min
        self.electride_charge_min = electride_charge_min
        self.electride_volume_min = electride_volume_min
        self.electride_radius_min = electride_radius_min
            
        # define properties that will be updated by running the method
        self._bifurcations = None
        self._bifurcation_graph = None
        self._bifurcation_plot = None
        self._downscaled_reference_grid = None
        self._downscaled_labels = None
        self._atomic_radii = None
        

    ###########################################################################
    # Calculated Properties
    ###########################################################################
    
    @cached_property
    def feature_structure(self) -> Structure:
        return self.get_feature_structure()
    
    @property
    def bifurcations(self) -> dict:
        if self._bifurcations is None:
            self._bifurcations = self._get_bifurcations()
        return self._bifurcations
    
    @property
    def bifurcation_graph(self) -> BifurcationGraph:
        if self._bifurcation_graph is None:
            self._get_bifurcation_graph()
        return self._bifurcation_graph
    
    @property
    def bifurcation_plot(self) -> go.Figure:
        if self._bifurcation_plot is None:
            self._bifurcation_plot = self._get_bifurcation_plot()
        return self._bifurcation_plot
    
    @property
    def atomic_radii(self) -> NDArray[np.float64]:
        if self._atomic_radii is None:
            self._get_atomic_radii()
        # TODO: Figure out a way to calculate this on the fly with only covalent/lone-pairs assigned
        return self._atomic_radii
    
    ###########################################################################
    # Core Graph Construction
    ###########################################################################
    
    def _get_bifurcation_graph(self):
        """
        This will construct a BifurcationGraph class.
        Each node will contain information on whether it is
        reducible/irreducible, atomic/valent, etc.

        This method is largely meant to be called through the get_bifurcation_graphs
        method.
        """
        # run bader for nice looking logging purposes
        # NOTE: I call a property rather than run_bader/run_atom_assignment to
        # avoid repeat calcs if we've already run
        self.atom_labels 
        
        logging.info("Beginning ELF Analysis")

        # get an initial graph connecting bifurcations and final basins
        self._initialize_bifurcation_graph()
        
        # Now we have a graph with information associated with each basin. We want
        # to label each node. First, we label cores as they are the simplest
        self._mark_cores()
        
        # Next, we mark shells. This step distinguishes ionic and covalent
        # features, the most ambiguous step
        self._mark_shells()
        
        # Next we label covalent bonds. These must lie along an atomic bond and
        # have a reasonably large charge
        self._mark_covalent()
        
        # Next we mark lone pairs. These split off from covalent bonds or rarely
        # from atomic shells (e.g. SnO)
        self._mark_lonepairs()
        
        plot = self.bifurcation_graph.get_plot()
        plot.write_html("test1.html")
        breakpoint()
        
        
        
        # Now we want to label our valence features as Covalent, Metallic, or bare electron.
        # Many covalent and metallic features are easy to find. Covalent bonds
        # are typically exactly along a bond between an atom and its nearest
        # neighbors. Metallic features have a low depth. We mark these first
        self._mark_covalent_lonepair()
        
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
        self._get_atomic_radii()
        self._correct_far_shell_features()
        
        # Reduce any related shell basins to a single basin for clarity
        if self.combine_shells:
            self._reduce_atomic_shells()
        
        # Now we want to mark the radius of each feature. We don't use the
        # downscaled grid here to get the best chance at a reasonable radius
        self._mark_feature_radii()
        
        # Now we calculate a bare electron indicator for each valence basin. This
        # is used just to give a sense of how bare an electron is vs. a more common
        # metallic feature.
        self._mark_bare_electron_indicator()
        
        # Sometimes a bare electron or metal feature will be mislabeled due to it
        # being nearly between two atoms. In these cases, the features are very
        # far outside the atoms radius, while a covalent bond never is. We relabel them
        # here.
        self._correct_far_covalent_features()
        
        # Finally, we want to distinguish between a metal and a bare electron.
        # This is currently very arbitrary and based on a series of cutoffs.
        self._mark_metallic_or_electride()
        
        # In some cases, the user may not have used a pseudopotential with enough core electrons.
        # This can result in an atom having no assigned core/shell, which will
        # result in nonsense later. We check for this here and throw an error
        assigned_atoms = []
        for node in self.bifurcation_graph:
            # We only want to consider basins that are core or shell, so we check
            # here and skip otherwise
            basin_subtype = getattr(node, "basin_subtype", None)

            if not basin_subtype in ["core", "shell"]:
                continue
            atom = getattr(node, "nearest_atom", None)
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
        # Finally, we ensure that all nodes have an assignment
        for node in self.bifurcation_graph:
            if node.reducible:
                continue
            if getattr(node, "basin_subtype", None) is None:
                raise Exception(
                    "At least one ELF feature was not assigned. This is a bug. Please report to our github:"
                    "https://github.com/jacksund/simmate/issues"
                )
                
    def _initialize_bifurcation_graph(self):
        self._bifurcation_graph = BifurcationGraph.from_bader(
            self, 
            labeler="ElfAnalyzer",
            )

    def _mark_cores(self):
        logging.info("Marking atomic cores")
        # cores are reducible domains that contain an atom. They should be within
        # one voxel of the atom. We must check for this as it is possible for
        # a different type of basin to contain the atom if too few valence electrons
        # are in the pseudopotential
        max_dist = self.reference_grid.max_point_dist
        for node in self.bifurcation_graph.unassigned_nodes:
            if len(node.contained_atoms) != 1:
                continue
            if node.atom_distance <= max_dist:
                node.feature_subtype = "core"
                
    def _mark_shells(self):
        logging.info("Marking atomic shells")
        # shells are "reducible" features that surround exactly one atom. The
        # main difficulty is distinguishing them from heterogeneous covalent
        # bonds.
        
        # as a first step, we label any features that were combined when the
        # graph was generated due to being very shallow. These will be marked
        # "shallow" and contain one atom
        for node in self.bifurcation_graph.unassigned_nodes:
            if node.feature_subtype == "shallow" and len(node.contained_atoms) == 1:
                node.feature_subtype = "shell"
                
        # Now we label shells that may be slightly deeper. Regardless of depth,
        # shells will always surround a single atom. To distinguish them from
        # heterogenous covalent bonds (which also may surround 1 atom), we gauge
        # the degree to which the potential shell features belong to the atom's
        # shell vs. another atoms shell. This can be thought of as a measure of
        # ionicity. To measure this, we compare the highest ELF value where
        # the feature fully surrounds the single atom to the highest value where
        # it surrounds at least one other atom

        possible_shells = []
        for node in self.bifurcation_graph.reducible_nodes:
            # skip nodes that don't surround a single atom or have a dimensionality above 0
            if node.is_infinite or len(node.contained_atoms) != 1:
                continue
            # skip nodes that have children that also contain the single atom
            if any([len(i.contained_atoms) == 1 for i in node.children]):
                continue
            # note this as a possible shell
            possible_shells.append(node)
        
        for node in possible_shells:
            # we want to find the highest ELF value where these features contribute
            # to another atoms outer shell. This isn't necessarily the first
            # parent that contains multiple atoms, as shells/covalent bonds often
            # connect to each other before surrounding a new atom. We want the
            # first ancestor that surrounds an atom that would otherwise not
            # be surrounded without candidate shells/covalent bonds
            
            # TODO: This method of finding the point where a shell surrounds a
            # new atom feels scuffed and I'm probably too tired to be thinking
            # through it right now. I need to think through it carefully.
            # set backup
            highest_neighbor_shell = node.ancestors[-1].max_value
            
            included_atoms = node.contained_atoms
            for parent in node.ancestors:
                if len(parent.contained_atoms) == len(included_atoms):
                    continue
                # check if all atoms are included in at least one child
                distinct_atoms = []
                for child in parent.children:
                    distinct_atoms.extend(child.contained_atoms)
                if not all([i in distinct_atoms for i in parent.contained_atoms]):
                    highest_neighbor_shell = parent.max_value
                    break
                included_atoms = parent.contained_atoms
            # get the highest point where this feature contains the atom
            highest_nearest_shell = node.max_value
            # check if the ratio is below our cutoff
            if (highest_neighbor_shell / highest_nearest_shell) < self.shared_shell_ratio:
                if self.combine_shells:
                    # convert to an irreducible node
                    node = node.make_irreducible()
                    node.feature_subtype = "shell"
                else:
                    for child in node.deep_children:
                        if not child.is_reducible:
                            child.subtype = "shell"


    
    def _mark_covalent(self):
        """
        Takes in a bifurcation graph and labels valence features that
        are obviously metallic or covalent
        """
        logging.info("Marking covalent features")
        graph = self.bifurcation_graph
        
        # get frac coords of unassigned nodes
        valence_nodes = []
        valence_frac = []
        for node in graph.unassigned_nodes:
            # don't include nodes with too low of charge (avoids shallow metallics)
            if node.charge > self.min_covalent_charge:
                valence_frac.append(node.frac_coords)
                valence_nodes.append(node)
        
        # Convert our cutoff angle to radians
        min_covalent_angle = self.min_covalent_angle * math.pi / 180
        
        # get our atom frac coords
        atom_frac_coords = self.structure.frac_coords
        atom_cart_coords = self.structure.cart_coords

        # check which nodes are within our tolerance
        # if/then is to avoid numba disliking empty lists
        if len(valence_frac) > 0:
            nodes_in_tolerance, atom_neighs = check_all_covalent(
                valence_frac, 
                atom_frac_coords, 
                atom_cart_coords, 
                frac2cart=self.structure.lattice.matrix, 
                min_covalent_angle=min_covalent_angle,
                )
        else:
            nodes_in_tolerance = []
            
        for node, in_tolerance, (atom0, atom1) in zip(valence_nodes, nodes_in_tolerance, atom_neighs):
            # skip nodes that aren't within our angle tolerance
            if not in_tolerance:
                continue
            
            # set backup
            contained_atoms = node.ancestors[-1].contained_atoms
            if atom0 in contained_atoms and atom1 in contained_atoms:
                is_covalent = True
            else:
                is_covalent = False
            
            # Sometimes a lone pair happens to align well with an atom that
            # is not part of our covalent system (e.g. CaC2). We can easily 
            # check for this by seeing if our atoms both belong to the parent
            # containing the full molecule
            included_atoms = node.contained_atoms
            for parent in node.ancestors:
                if len(parent.contained_atoms) == len(included_atoms) or len(parent.contained_atoms <= 1):
                    continue
                # check if all atoms are included in at least one child
                distinct_atoms = []
                for child in parent.children:
                    distinct_atoms.extend(child.contained_atoms)
                if not all([i in distinct_atoms for i in parent.contained_atoms]):
                    if atom0 in parent.contained_atoms and atom1 in parent.contained_atoms:
                        is_covalent = True
                    else:
                        is_covalent = False
                    break
                included_atoms = parent.contained_atoms

            if is_covalent:
                node.feature_subtype = "covalent"
                
    def _mark_lonepairs(self):
        # lone-pairs separate off from covalent bonds or rarely from an ionic
        # core/shell.
        
        for node in self.bifurcation_graph.unassigned_nodes:
            # get the first parent with an atom
            for parent in node.ancestors:
                if len(parent.contained_atoms) > 0:
                    break
            # now check if all the children of this parent are covalent or atomic
            all_covalent = True
            all_atomic = True
            for child in parent.deep_children:
                if child.is_reducible:
                    continue
                if not child.feature_subtype in ["covalent", "lone-pair", None]:
                    all_covalent = False
                if not child.feature_subtype in ["core", "shell", None]:
                    all_atomic = False
            if all_covalent or (node.charge > self.min_covalent_charge and all_atomic):
                node.feature_subtype = "lone-pair"

            
    
    def _mark_covalent_lonepair(self):
        """
        Takes in a bifurcation graph and labels valence features that
        are obviously metallic or covalent
        """
        logging.info("Marking covalent features")
        graph = self.bifurcation_graph
        
        # Make a first pass to collect nodes that might be covalent. We do
        # this so that the more expensive operation can be done in parallel
        # with numba
        valence_nodes = []
        valence_frac = []
        for node in graph:
            if node.reducible or getattr(node, "basin_type") != "val":
                continue
            # check if we are under our charge tolerance
            if node.charge < self.min_covalent_charge:
                # this isn't a covalent feature. If we previously noted our
                # subtype as "other" this feature is part of an atom/molecule
                # and must be a lone pair. Otherwise, its a bare electron or metal bond
                if node.basin_subtype == "other":
                    node.basin_subtype = "lone-pair"
                else:
                    node.basin_subtype = "bare electron"
                continue
            # otherwise, we want to investigate the position relative to nearby
            # neighbors
            valence_nodes.append(node)
            valence_frac.append(node.frac_coords)
        
        # Now get which of the remaining nodes are covalent
        # convert bond angle cutoff to radians
        min_covalent_angle = self.min_covalent_angle * math.pi / 180
        atom_frac_coords = self.structure.frac_coords
        atom_cart_coords = self.structure.cart_coords
        if len(valence_frac) > 0:
            covalent_nodes = check_all_covalent(
                valence_frac, 
                atom_frac_coords, 
                atom_cart_coords, 
                frac2cart=self.structure.lattice.matrix, 
                min_covalent_bond_ratio=self.min_covalent_bond_ratio, 
                min_covalent_angle=min_covalent_angle,
                )
        else:
            covalent_nodes = []

        for node, covalent in zip(valence_nodes, covalent_nodes):
            if covalent:
                node.basin_subtype = "covalent"
            else:
                if node.basin_subtype == "other":
                    node.basin_subtype = "lone-pair"
                else:
                    node.basin_subtype = "bare electron"
        
        

        # !!! I'm not sure if the following code is really necessary
        # # There is an exception to the lone-pair rule that can result in missing
        # # a lone-pair assignment. If a covalent/lone-pair feature surrounds two atoms
        # # these features won't be assigned as "other".
        # # This happens in CaC2 around the C2 molecules for example. The covalent
        # # bonds are labeled in the loop above, but the lone-pair will
        # # still be labeled as a bare electron. We correct for this in an
        # # additional loop by checking for bare electrons that are siblings with
        # # covalent bonds.
        # # BUG-FIX rather than exact siblings, we want all of the features that
        # # are children of the parent domain that fully surrounds the molecule
        # # TODO: This could be moved to a Node property
        # def get_molecule_parent(node):
        #     # get parent that fully surrounds at least one atom
        #     molecule_parent = None
        #     parent = node.parent
        #     while molecule_parent is None:
        #         if len(parent.atoms) != 0:
        #             molecule_parent = parent
        #         else:
        #             parent = parent.parent
        #     return molecule_parent
        
        # # keep track of nodes to reassign as lone-pairs. We can't reassign them
        # # in this loop because we check that at least one sibling is a covalent
        # # bond, and we don't want to accidentally relabel them.
        # nodes_to_relabel = []
        # for node in graph:
        #     if node.reducible or getattr(node, "basin_type") != "val":
        #         continue
        #     if node.basin_subtype == "bare electron":

        #         all_cov_lp_be = True
        #         at_least_one_cov = False
        #         molecule_parent = get_molecule_parent(node)

        #         # Check if all siblings are covalent, bare electrons, or lone-pairs. If so,
        #         # this is a lone-pair
        #         for sibling in molecule_parent.deep_children:
        #             # skip reducible siblings
        #             if sibling.reducible:
        #                 continue
        #             # make sure this sibling isn't the child of a different submolecule
        #             direct_parent = get_molecule_parent(sibling)
        #             if len(direct_parent.atoms) != 0 and direct_parent != molecule_parent:
        #                 continue

        #             # We need to make sure there's at least one covalent bond as well
        #             if sibling.basin_subtype == "covalent":
        #                 at_least_one_cov = True
        #             elif sibling.basin_subtype not in [
        #                 "bare electron",
        #                 "covalent",
        #                 "lone-pair",
        #             ]:
        #                 all_cov_lp_be = False
        #         if all_cov_lp_be and at_least_one_cov:
        #             nodes_to_relabel.append(node)
        # for node in nodes_to_relabel:
        #     node.basin_subtype = "lone-pair"
            
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
        Corrects any 'shell' nodes that are outside the radius of the atom
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
        min_elf = 50
        nearest_atom = -1
        nearest_atom_type = None
        frac_coords = None
        # update all of our shell characteristics
        for child in nodes:
            # update atom distance if better than other children
            if child.atom_distance < atom_distance:
                atom_distance = child.atom_distance
                nearest_atom = child.nearest_atom
                nearest_atom_type = child.nearest_atom_type
            
            # add basins to our list
            basins.extend(child.basins)
            
            # add volume and charge to our total
            volume += child.volume
            charge += child.charge
            
            # update the value the feature disappears at if its
            # greater.
            if child.max_elf > max_elf:
                max_elf = child.max_elf
                try:
                    frac_coords = child.frac_coords
                except:
                    breakpoint()
            
            # update the value the feature appears at if lower. This
            # should get overwritten if we delete the parent node
            # anyways
            if child.min_elf < min_elf:
                min_elf = child.min_elf

        # Add the attributes
        node = nodes[0]
        node.basin_type = "atom"
        node.basin_subtype = "shell"
        node.basins = np.array(basins)
        node.atom_distance = atom_distance
        node.volume = volume
        node.charge = charge
        node.min_elf = min_elf
        node.max_elf = max_elf
        node.nearest_atom = nearest_atom
        node.nearest_atom_type = nearest_atom_type
        node.frac_coords = frac_coords
        
        # Recalculate depth
        node.depth = node.max_elf - node.min_elf
        node.depth_to_infinite = self._get_depth_to_infinite(node)

        children_to_remove = nodes[1:]
        # delete all of the unused nodes
        for child in children_to_remove:
            child.remove()

    def _reduce_atomic_shells(self):
        """
        Reduces shell nodes to a single node
        """
        logging.info("Reducing shell features")
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
                # BUG-FIX: assign the value this parent appears at to the
                # child and recalculate depth
                child.min_elf = parent.min_elf
                new_depth = child.max_elf - child.min_elf
                child.depth = new_depth                
                
                child.reducible = True # We will need to check for this later
                # BUG-FIX: assign the child the atoms in the parent
                child.atoms = parent.atoms
                child.is_infinite = parent.is_infinite
                child.dimensionality = parent.dimensionality
                # delete the parent
                parent.remove()
                # recalculate depth 3d
                child.depth_to_infinite = self._get_depth_to_infinite(node)

    
    def _mark_feature_radii(self):
        basin_radii = self.basin_surface_distances
        for node in self.bifurcation_graph:
            # skip atomic and reducible features
            if node.reducible or getattr(node, "basin_type") != "val":
                continue
            basins = node.basins
            node.feature_radius = basin_radii[basins].min()
        
    def _mark_bare_electron_indicator(self):
        """
        Takes in a bifurcation graph and calculates an electride character
        score for each valence feature. Electride character ranges from
        0 to 1 and is the combination of several different metrics:
        ELF value, charge, depth, volume, and atom distance.
        """
        # create a structure object with oxidation states for improved 
        # crystalnn
        temp_structure = self.structure.copy()
        temp_structure.add_oxidation_state_by_guess()
        
        for node in track(self.bifurcation_graph, description="Calculating bare electron character"):
            # skip atomic and reducible features
            if node.reducible or getattr(node, "basin_type") != "val":
                continue
            
            # We want to get a metric of how "bare" each feature is. To do this,
            # we need a value that ranges from 0 to 1 for each attribute we have
            # available. We can combine these later with or without weighting to
            # get a final value from 0 to 1.
            # First, the ELF value already ranges from 0 to 1, with 1 being more
            # localized. We don't need to alter this in any way.
            elf_contribution = node.max_elf

            # next, we look at the charge. If we are using a spin polarized result
            # the maximum amount should be 1. Otherwise, the value could be up
            # to 2. We make a guess at what the value should be here
            charge = node.charge
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
            depth_contribution = node.depth_to_infinite

            # Next is the volume. Bare electrons are usually thought of as being
            # similar to a free s-orbital with a similar size to a hydride. Therefore
            # we use the hydride crystal radius to calculate an ideal volume and set
            # this contribution as a fraction of this, capping at 1.
            hydride_radius = 1.34  # Taken from wikipedia and subject to change
            hydride_volume = 4 / 3 * 3.14159 * (hydride_radius**3)
            volume_contribution = min(node.volume / hydride_volume, 1)

            # Next is the distance from the atom. Ideally this should be scaled
            # relative to the radius of the atom, but which radius to use is a
            # difficult question. We use CrystalNN to get the neighbors around
            # the nearest atom and get the EN difference. We use this to guess
            # whether covalent or ionic radii should be used, then pull the appropriate one.
            # First, we also want to get the coordination environment of this
            # feature, even though this doesnt feed into our BEI.
            # TODO: I don't like this because it probably scales poorly due to
            # CrystalNN. Is there a way to get the neighbors without this?
            frac_coords = node.frac_coords
            feature_structure = temp_structure.copy()
            feature_structure.append("H-", frac_coords)
            cnn = CrystalNN(distance_cutoffs=None)
            coordination = cnn.get_nn_info(feature_structure, -1)
            coord_num = len(coordination)
            coord_indices = [i["site_index"] for i in coordination]
            coord_atoms = [feature_structure[i].specie.symbol for i in coord_indices]
            # Now that we have the nearby atoms, we want to get the smallest radius
            # of this basin
            atom_indices = np.unique(coord_indices)
            atom_radius = 10
            atom_distance = 10
            dist_minus_radius = 10
            nearest_atom_idx = -1
            nearest_atom_species = None
            for atom_idx in atom_indices:
                atom_radius_new = self.atomic_radii[atom_idx]
                dist = feature_structure.get_distance(atom_idx, -1)
                dist_minus_radius_new = dist-atom_radius_new
                if dist_minus_radius_new < dist_minus_radius:
                    dist_minus_radius = dist_minus_radius_new
                    atom_radius = atom_radius_new
                    atom_distance = dist
                    nearest_atom_idx = atom_idx
                    nearest_atom_species = feature_structure[atom_idx].specie.symbol
                    
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
                    node.volume,
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
            node.unnormalized_bare_electron_indicator = unnormalized_contributors
            node.bare_electron_indicator = bare_electron_indicator
            node.bare_electron_scores = contributers
            node.dist_beyond_atom = dist_minus_radius
            node.coord_num = coord_num
            node.coord_indices = coord_indices
            node.coord_atoms = coord_atoms
            node.atom_distance = atom_distance
            node.nearest_atom = nearest_atom_idx
            node.nearest_atom_type = nearest_atom_species
            

    def _correct_far_covalent_features(self):
        # BUG-FIX On occasion, a metal feature will sit very close to being along
        # an atom-atom bond, but will sit well outside that atoms ELF radius. In
        # these cases they will be mislabeled as covalent. We correct for that here
        # skip atomic and reducible features
        for node in self.bifurcation_graph:
            if node.reducible or getattr(node, "basin_type") != "val":
                continue
            dist_beyond_atom = node.dist_beyond_atom
            feature_subtype = node.basin_subtype
            if dist_beyond_atom > 0.2 and feature_subtype in ["covalent", "lone-pair"]:
                breakpoint()
                node.basin_subtype = "bare electron"
    
    def _mark_metallic_or_electride(self):
        logging.info("Marking metallic and bare electron features")
        # create an array of our conditions to check against
        conditions = np.array(
            [
                self.electride_elf_min,
                self.electride_depth_min,
                self.electride_charge_min,
                self.electride_volume_min,
                self.electride_radius_min,
            ]
        )

        for node in self.bifurcation_graph:
            if getattr(node, "basin_subtype", "") != "bare electron":
                # skip features that aren't bare electrons
                continue
            # we have a bare electron. We check each condition
            condition_test = np.array(
                [
                    node.max_elf,
                    node.depth_to_infinite,  # Note we use the depth to an infinite connection rather than true depth
                    node.charge,
                    node.volume,
                    # attributes["feature_radius"],
                    node.dist_beyond_atom
                ]
            )
            # check if we meet all conditions. If so we have a bare electron/electride
            if np.all(condition_test > conditions):
                subtype = "bare electron"
            else:
                # We don't meet our conditions so we consider this some form
                # of metallic feature
                subtype = "metallic"
            node.basin_subtype = subtype


    ###########################################################################
    # Post Graph Construction
    ###########################################################################

    def get_feature_structure(
            self, 
            include_shared_features: bool = True,
            include_lone_pairs: bool = True,
            ):
        
        # First, we get the valence features for this graph and create a
        # structure that we will add features to
        structure = self.structure.copy()
        structure.remove_oxidation_states()
        structure_index_to_node = {}
        for feat_idx, node in enumerate(self.bifurcation_graph):
            # skip nodes that aren't valence
            if node.reducible or getattr(node, "basin_type") != "val":
                continue

            # get our subtype
            subtype = node.basin_subtype
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
            frac_coords = node.frac_coords
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
            self.bifurcation_graph[node].feature_structure_index = index

        return sorted_structure
    
    ###########################################################################
    # Utilities
    ###########################################################################
    @property
    def _site_voxel_coords(self) -> np.array:
        frac_coords = self.structure.frac_coords
        vox_coords = self.reference_grid.get_voxel_coords_from_frac(frac_coords)
        return vox_coords.astype(int)
    
    # @cached_property
    # def _site_sphere_voxel_coords(self) -> list:
    #     site_sphere_coords = []
    #     for vox_coord in self._site_voxel_coords:
    #         nearby_voxels = self.reference_grid.get_voxels_in_radius(0.05, vox_coord)
    #         site_sphere_coords.append(nearby_voxels)
    #     return site_sphere_coords
    
    def _get_atomic_radii(self):
        # We will need to get radii from the ELF. To do this, we need a labeled
        # bader result to pass to our PartitioningToolkit
        frac_coords = self.basin_maxima_frac
        feature_structure = self.structure.copy()
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
                feature_structure.append(species, frac_coord)
        
        # recalculate the atoms for our bader object
        _,_, feature_labels = self.assign_basins_to_structure(feature_structure)
        
        radii_tools = IonicRadiiTools(
            grid=self.reference_grid,
            feature_labels=feature_labels,
            feature_structure=feature_structure,
            )
        self._atomic_radii = radii_tools.atomic_radii
    
    # @staticmethod
    # def _get_shared_feature_neighbors(structure: Structure) -> NDArray:
    #     """
    #     For each covalent bond or metallic feature in a dummy atom labeled
    #     structure, returns a list of nearest atom neighbors.
    #     """
    #     # We want to get the atoms and electride sites that are closest to each
    #     # shared feature. However, we don't want to find any nearby shared features
    #     # as neighbors.
    #     # To do this we will remove all of the shared dummy atoms, and create
    #     # temporary structures with only one of the shared dummy atoms at a time.
    #     shared_feature_indices = []
    #     cleaned_structure = structure.copy()
    #     for symbol in ["Z", "M", "Le", "Lp"]:
    #         if not symbol in cleaned_structure.symbol_set:
    #             continue
    #         cleaned_structure.remove_species([symbol])
    #         shared_feature_indices.extend(structure.indices_from_symbol(symbol))
    #     shared_feature_indices = np.array(shared_feature_indices)
    #     shared_feature_indices.sort()
    #     # We will be using the indices of the cleaned structure to note neighbors,
    #     # so these must match the original structure. We assert that here
    #     assert all(
    #         cleaned_structure[i].species == structure[i].species
    #         for i in range(len(cleaned_structure))
    #     ), "Provided structure must list atoms and electride dummy atoms first"

    #     # Replace any electrides with "He" so that CrystalNN doesn't throw an error
    #     if "E" in cleaned_structure.symbol_set:
    #         cleaned_structure.replace_species({"E": "He"})
    #     # for each index, we append a dummy atom ("He" because its relatively small)
    #     # then get the nearest neighbors
    #     cnn = CrystalNN(distance_cutoffs=None)
    #     all_neighbors = []
    #     for idx in shared_feature_indices:
    #         neigh_indices = []
    #         # Add this dummy atom to the temporary structure
    #         frac_coords = structure[idx].frac_coords
    #         feature_structure = cleaned_structure.copy()
    #         feature_structure.append("He", frac_coords)
    #         # Get the nearest neighbors to this dummy atom
    #         nn = cnn.get_nn(feature_structure, -1)
    #         # Get the index for each neighboras a list, then append this list
    #         # to our full list. Note that it is important that these indices be
    #         # the same as in the original structure, so atoms and electrides must
    #         # come before shared electrons in the provided structure.
    #         for n in nn:
    #             neigh_indices.append(n.index)
    #         all_neighbors.append(neigh_indices)
    #     return all_neighbors

    # def _get_atom_en_diff_and_cn(self, site: int) -> list([float, int]):
    #     """
    #     Uses the coordination environment of an atom to get the EN diff
    #     between it and it's neighbors as well as its coordination number.
    #     This is useful for guessing which radius to use.
    #     """
    #     # get the neighbors for this site and its electronegativity
    #     neigh_list = self.atom_coordination_envs[site]
    #     site_en = self.structure.species[site].X
    #     # create a variable for storing the largest EN difference
    #     max_en_diff = 0
    #     for neigh_dict in neigh_list:
    #         # get the EN for each neighbor and calculate the difference
    #         neigh_site = neigh_dict["site_index"]
    #         neigh_en = self.structure.species[neigh_site].X
    #         en_diff = site_en - neigh_en
    #         # if the difference is larger than the current stored one, replace
    #         # it.
    #         if abs(en_diff) > max_en_diff:
    #             max_en_diff = en_diff
    #     # return the en difference and number of neighbors
    #     return max_en_diff, len(neigh_list)
    ###########################################################################
    # Read methods
    ###########################################################################
    @classmethod
    def from_vasp(
        cls,
        charge_filename: Path | str = "CHGCAR",
        reference_filename: Path | str = "ELFCAR",
        **kwargs,
    ) -> Self:
        """
        Creates an ElfAnalyzer class object from VASP files.

        Parameters
        ----------
        charge_filename : Path | str, optional
            The path to the CHGCAR like file that will be used for summing charge.
            The default is "CHGCAR".
        reference_filename : Path | str
            The path to ELFCAR like file that will be used for partitioning.
            If None, the charge file will be used for partitioning.
        **kwargs : dict
            Keyword arguments to pass to the Bader class.

        Returns
        -------
        Self
            An ElfAnalyzer class object.

        """
        # This is just a wrapper of the Bader class to update the default to
        # load the ELFCAR
        return super().from_vasp(
            charge_filename=charge_filename,
            reference_filename=reference_filename,
            **kwargs
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
            node_keys: list[int], 
            directory: str | Path = None,
            write_reference: bool = True,
            use_feature_structure: bool = True,
            output_format: str | Format = None,
            **writer_kwargs,
            ):
        """
        For a give list of node keys, writes the bader basins associated with
        each.
        """
        # get the data to use
        if write_reference:
            data_array = self.reference_grid.total
            data_type = self.reference_grid.data_type
        else:
            data_array = self.charge_grid.total
            data_type = self.charge_grid.data_type
        
        # get the structure to use
        if use_feature_structure:
            structure = self.feature_structure
        else:
            structure = self.structure

        if directory is None:
            directory = Path(".")
            
        # TODO: Update to be more like bader
        graph = self.bifurcation_graph
        for key in node_keys:
            node = graph.node_from_key(key)
            basin_indices = node.basins
            # create a mask including each of the requested basins
            mask = np.isin(self.basin_labels, basin_indices)
            # copy data to avoid overwriting. Set data off of basin to 0
            data_array_copy = data_array.copy()
            data_array_copy[~mask] = 0.0
            grid = Grid(
                structure=structure,
                data={"total": data_array_copy},
                data_type=data_type,
            )
            file_path = directory / f"{grid.data_type.prefix}_f{key}"
            # write file
            grid.write(filename=file_path, output_format=output_format, **writer_kwargs)

    
    def write_valence_basins(
            self,
            directory: str | Path = None,
            write_reference: bool = True,
            use_feature_structure: bool = True,
            output_format: str | Format = None,
            **writer_kwargs,
            ):
        graph = self.bifurcation_graph
        nodes = []
        for node in graph:
            if node.reducible or getattr(node, "basin_type") != "val":
                continue
            nodes.append(node.key)
        self.write_feature_basins(
            nodes,
            directory,
            write_reference,
            use_feature_structure,
            output_format,
            **writer_kwargs,
            )