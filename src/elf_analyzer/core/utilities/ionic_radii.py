# -*- coding: utf-8 -*-
"""
This is a reimplementation of the ionic radius finder I created for BadELF in
[Simmate](https://github.com/jacksund/simmate/blob/main/src/simmate/apps/badelf/core/partitioning.py)
"""
from pymatgen.core import Structure
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from functools import cached_property
from scipy.interpolate import RegularGridInterpolator
from baderkit.core import Grid, Bader
import logging

class IonicRadiiTools:
    def __init__(
            self,
            grid: Grid,
            bader: Bader,
            ):
        self.grid = grid
        self.bader = bader
    
    @cached_property
    def all_site_neighbor_pairs(self):
        """
        A dataframe containing information about all site-neighbor pairs in a
        structure within 15A of each other.
        """
        grid = self.grid
        structure = grid.structure
        # logging.info("Getting all neighboring atoms for each site in structure")
        # Get all neighbors within 15 Angstrom
        nearest_neighbors = structure.get_neighbor_list(15)
        # Get the equivalent atom index for each atom
        equivalent_atoms = grid.equivalent_atoms
        equiv_site_index = equivalent_atoms[nearest_neighbors[0]]
        equiv_neigh_index = equivalent_atoms[nearest_neighbors[1]]
        # Create dataframe with important info about each site/neighbor pair
        site_neigh_pairs = pd.DataFrame()
        # Add sites and neighbors indices
        site_neigh_pairs["site_index"] = nearest_neighbors[0]
        site_neigh_pairs["neigh_index"] = nearest_neighbors[1]
        site_neigh_pairs["equiv_site_index"] = equiv_site_index
        site_neigh_pairs["equiv_neigh_index"] = equiv_neigh_index
        site_neigh_pairs["site_symbol"] = None
        site_neigh_pairs["neigh_symbol"] = None

        site_cart_coords = []
        for site_index, site in enumerate(structure):
            # Add species strings to all site and neighbor indices with given index
            species_string = site.species_string
            site_condition = site_neigh_pairs["site_index"] == site_index
            neigh_condition = site_neigh_pairs["neigh_index"] == site_index
            site_neigh_pairs.loc[site_condition, "site_symbol"] = species_string
            site_neigh_pairs.loc[neigh_condition, "neigh_symbol"] = species_string
            # Get the coordinate for this site index. Create a list of arrays made up
            # of this coordinate and add it to our site coords list
            site_coord = list(site.coords)
            atom_neigh_len = len(site_neigh_pairs.loc[site_condition])
            site_coords_array = np.tile(site_coord, (atom_neigh_len, 1))
            site_cart_coords.extend(site_coords_array)

        # Add the distances for each site-neighbor pair, then round them to 5 decimals
        site_neigh_pairs["dist"] = nearest_neighbors[3]
        site_neigh_pairs["dist"] = site_neigh_pairs["dist"].round(5)
        # Add the site coordinates
        site_neigh_pairs["site_coords"] = site_cart_coords
        # Get the fractional coordinates for each neighbor atom. Then calculate the cartesian coords
        neigh_frac_coords = (
            structure.frac_coords[nearest_neighbors[1]] + nearest_neighbors[2]
        )
        neigh_cart_coords = []
        neigh_cart_coords.extend(
            grid.get_cart_coords_from_frac(neigh_frac_coords)
        )
        # Add the neighbors cartesian coordinates
        site_neigh_pairs["neigh_coords"] = neigh_cart_coords

        # Create columns for the partitioning fraction and radius
        site_neigh_pairs["partitioning_frac"] = None
        site_neigh_pairs["radius"] = None
        site_neigh_pairs.sort_values(by="dist", inplace=True)

        return site_neigh_pairs
    
    def get_partitioning_line_from_voxels(
        self,
        site_voxel_coord: NDArray,
        neigh_voxel_coord: NDArray,
        method: str = "linear",
        steps: int = 200,  #!!! This should be set dynamically in the future
    ):
        """
        Finds a line of voxel positions between two atom sites and then finds the value
        of the partitioning grid at each of these positions. The values are found
        using an interpolation function defined using SciPy's RegularGridInterpeter.

        Args:
            site_voxel_coord (ArrayLike):
                The voxel coordinates of an atomic site
            neigh_voxel_coord (ArrayLike):
                The voxel coordinates of a neighboring
                site
            method (str):
                The method of interpolation. 'cubic' is more rigorous
                than 'linear'
            steps (int):
                The number of voxel coordinates to interpolate. Default is 200

        Results:
            A list with 200 pairs of voxel coordinates and data values along
            a line between two positions.
        """
        grid_data = self.grid.copy().total
        label_data = self.bader.atom_labels
        slope = [b - a for a, b in zip(site_voxel_coord, neigh_voxel_coord)]
        slope_increment = [float(x) / steps for x in slope]

        # get a list of points along the connecting line. First add the original
        # site
        position = site_voxel_coord
        line = [[round(float(a % b), 12) for a, b in zip(position, grid_data.shape)]]
        for i in range(steps):
            # move position by slope_increment
            position = [float(a + b) for a, b in zip(position, slope_increment)]

            # Wrap values back into cell
            # We must do (a-1) to shift the voxel index (1 to grid_max+1) onto a
            # normal grid, (0 to grid_max), then do the wrapping function (%), then
            # shift back onto the VASP voxel index.
            position = [
                round(float(a % b), 12) for a, b in zip(position, grid_data.shape)
            ]

            line.append(position)

        # The partitioning uses a padded grid and grid interpolation to find the
        # location of dividing planes.
        padded_grid_data = np.pad(grid_data, 1, mode="wrap")
        padded_label_data = np.pad(label_data, 1, mode="wrap")

        # interpolate grid to find values that lie between voxels. This is done
        # with a cruder interpolation here and then the area close to the minimum
        # is examened more closely with a more rigorous interpolation in
        # get_line_frac_min
        a, b, c = self.grid.get_padded_grid_axes(1)
        fn = RegularGridInterpolator((a, b, c), padded_grid_data, method=method)
        fn_label = RegularGridInterpolator((a, b, c), padded_label_data, "nearest")
        # get a list of the ELF values along the line
        values = []
        label_values = []

        for pos in line:
            adjusted_pos = [x + 1 for x in pos]
            value = float(fn(adjusted_pos))
            label_value = int(fn_label(adjusted_pos))
            values.append(value)
            label_values.append(label_value)

        return line, values, label_values
    
    def get_partitioning_line_from_cart_coords(
        self,
        site_cart_coords: NDArray,
        neigh_cart_coords: NDArray,
        method: str = "linear",
    ):
        """
        Gets the voxel positions and elf values for points between two sites in
        the structure given as cartesian coordinates. This method can also be
        used to find the values in the ELF between two arbitrary points in the
        structure.

        Args:
            site_cart_coords (ArrayLike):
                cartesian coordinates of a site in the structure
            neigh_cart_coords (ArrayLike):
                cartesian coordinates of a second site in the structure
            method (str):
                The method of interpolation. 'cubic' is more rigorous
                than 'linear'

        Returns:
            Two lists, one of positions in voxel coordinates and another of elf
            values
        """
        grid = self.grid.copy()
        site_voxel_coord = grid.get_voxel_coords_from_cart(site_cart_coords)
        neigh_voxel_coord = grid.get_voxel_coords_from_cart(neigh_cart_coords)
        return self.get_partitioning_line_from_voxels(
            site_voxel_coord, neigh_voxel_coord, method=method
        )
    
    @staticmethod
    def find_minimum(values: NDArray):
        """
        Finds the local minima in a list of values and returns the index and value
        at each as a list of form [[min_index1, min_value1], [min_index2, min_value2], ...]

        Args:
            values (list):
                The list of values to find the minima of

        results:
            A list of minima represented by [index, value]
        """
        minima = [
            [i, y]
            for i, y in enumerate(values)
            if ((i == 0) or (values[i - 1] >= y))
            and ((i == len(values) - 1) or (y < values[i + 1]))
        ]
        return minima
    
    @staticmethod
    def find_maximum(values: NDArray):
        """
        Finds the local maxima in a list of values and returns the index and value
        at each as a list of form [[max_index1, max_value1], [max_index2, max_value2], ...]

        Args:
            values (list):
                The list of values to find the minima of

        results:
            A list of maxima represented by [index, value]
        """
        maxima = [
            [i, y]
            for i, y in enumerate(values)
            if ((i == 0) or (values[i - 1] <= y))
            and ((i == len(values) - 1) or (y > values[i + 1]))
        ]
        return maxima
    
    @staticmethod
    def get_closest_extrema_to_center(
        values: NDArray,
        extrema: NDArray,
    ):
        """
        Takes a list of values and the relative extrema (either minima or maxima)
        and finds which extrema is closest to the center of the line.

        Args:
            values (list):
                A list of values
            extrema (list):
                A list of extrema of form [index, value]

        results:
            The global extreme of form [index, value]
        """
        midpoint = len(values) / 2
        differences = []
        for pos, val in extrema:
            diff = abs(pos - midpoint)
            differences.append(diff)
        min_pos = differences.index(min(differences))
        global_extrema = extrema[min_pos]
        return global_extrema
    
    def _refine_line_part_frac(
        self,
        positions: list,
        elf_min_index: int,
        extrema: str,
        method: str = "cubic",
    ):
        """
        Refines the location of the minimum along an ELF line between two sites.
        To do this, the initial estimate from a linear interpolation of the line
        is used and a cubic interpolation is used in a smaller area around the
        estimated point. The sampled area is adjusted if it is found to not be
        centered on the new more accurate minimum.

        Args:
            positions (list):
                A list of positions given as voxel coordinates along the line
                of interest
            elf_min_index (int):
                The index along the line at which the linear interpolation estimated
                the minimum.
            extrema (str):
                Which type of extrema to refine. Either max or min.
            method (str):
                The method to use for interpolation

        Returns:
            The global minimum of form [line_position, value, frac_position]
        """
        amount_to_pad = 10
        grid = self.grid.copy()
        padded = np.pad(grid.total, amount_to_pad, mode="wrap")

        # interpolate the grid with a more rigorous method to find more exact value
        # for the plane.
        a, b, c = grid.get_padded_grid_axes(10)
        fn = RegularGridInterpolator((a, b, c), padded, method=method)

        # create variables for if the line needs to be shifted from what the
        # rough partitioning found
        centered = False
        amount_to_shift = 0
        attempts = 0

        while centered == False:
            if attempts == 5:
                break
            else:
                attempts += 1
                # If the position wasn't centered previously, we need to shift
                # the index
                elf_min_index = elf_min_index + amount_to_shift
                line_section = positions[elf_min_index - 3 : elf_min_index + 4]
                line_section_x = [
                    i for i in range(elf_min_index - 3, elf_min_index + 4)
                ]

                values_fine = []
                # Get the list of values from the interpolated grid
                for pos in line_section:
                    new_pos = [i + amount_to_pad for i in pos]
                    value_fine = float(fn(new_pos))
                    values_fine.append(value_fine)

                # Find the minimum value of this line as well as the index for this value's
                # position.
                try:
                    if extrema == "min":
                        minimum_value = min(values_fine)
                    elif extrema == "max":
                        minimum_value = max(values_fine)
                except:
                    attempts = 5
                    continue
                min_pos = values_fine.index(minimum_value)  # + global_min_pos[0]-5

                if min_pos == 4:
                    # Our line is centered and we can move on
                    centered = True
                else:
                    # Our line is not centered and we need to adjust it
                    amount_to_shift = min_pos - 4

        if not centered:
            # The above sometimes fails because the linear fitting gives a guess
            # for the minimum that isn't close. To handle this we treat these
            # situations rigorously
            values = []

            # Get the ELF value for every position in the line.
            for pos in positions:
                new_pos = [i + amount_to_pad for i in pos]
                value = float(fn(new_pos))
                values.append(value)

            # Get a list of all of the minima along the line
            if extrema == "min":
                minima = self.find_minimum(values)
            elif extrema == "max":
                minima = self.find_maximum(values)

            # then we grab the local minima closest to the midpoint of the line
            global_min = self.get_closest_extrema_to_center(values, minima)

            # now we want a small section of the line surrounding the minimum
            values_fine = values[global_min[0] - 3 : global_min[0] + 4]
            line_section_x = [i for i in range(global_min[0] - 3, global_min[0] + 4)]

        # now that we've found the values surrounding the minimum of our line,
        # we can fit these values to a 2nd degree polynomial and solve for its
        # minimum point
        try:
            d, e, f = np.polyfit(line_section_x, values_fine, 2)
            x = -e / (2 * d)
            elf_min_index_new = x
            elf_min_value_new = np.polyval(np.array([d, e, f]), x)
            elf_min_frac_new = elf_min_index_new / (len(positions) - 1)
        except:
            raise Exception("Refinement reached end of bond and failed to find radius.")

        return [elf_min_index_new, elf_min_value_new, elf_min_frac_new]
    
    def get_elf_ionic_radii(
        self,
        refine_method: str = "cubic",
        labeled_structure: Structure = None,
    ):
        """
        Gets the ELF radius for all atoms in the grid structure. See
        get_elf_ionic_radius for more detail.

        Args:
            refine_method (str):
                The method to use to interpolate ELF during refinement.
                "cubic" is more accurate but takes longer, "linear" is
                faster but can change significantly with grid density
            labeled_structure (Structure):
                A structure labeled with dummy atoms. This is used to
                determine what type of non-atomic basin is between atoms
                if there is one and should match the atoms in the
                bader parameter.

        Returns:
            A list of ELF radii for each site
        """
        equiv_elements = self.grid.equivalent_atoms

        unique_radii = np.zeros(len(equiv_elements))

        for atom_idx in np.unique(equiv_elements):
            radius = self.get_elf_ionic_radius(
                atom_idx, refine_method, labeled_structure
            )
            unique_radii[np.where(equiv_elements == atom_idx)[0]] = radius

        return unique_radii
    
    def get_elf_ionic_radius(
        self,
        site_index: int,
        refine_method: str = "cubic",
        labeled_structure: Structure = None,
    ):
        """
        This method gets the ELF ionic radius. It interpolates the ELF values
        between a site and it's closest neighbor. For ionic bonds, the
        radius is defined as the minimum between the two atoms. This has
        been shown to be very similar to the Shannon Crystal Radius,
        but gives more specific values.
    
        For covalent bonds and some electrides (e.g. NaBa3N) there will
        be a region that does not belong to only one of the atoms. For
        covalent bonds the radius is defined at the maximum of the covalent
        basin. For metal/electride features, the radius is defined at
        the last point belonging to the atom of interest.
        Note that this second case is not equivalent to the partitioning
        planes used for the BadELF algorithm, which will always use the
        ionic/covalent separation.
    
        Args:
            site_index (int):
                An integer value referencing an atom in the structure
            refine_method (str):
                The method to use to interpolate ELF during refinement.
                "cubic" is more accurate but takes longer, "linear" is
                faster but can change significantly with grid density
            labeled_structure (Structure):
                A structure labeled with dummy atoms. This is used to
                determine what type of non-atomic basin is between atoms
                if there is one and should match the atoms in the
                bader parameter.
    
        Returns:
            The distance the ELF ionic radius of the site
        """
        # get closest neighbor for the given site
    
        neighbors = self.all_site_neighbor_pairs
        # get only this sites dataframe
        site_df = neighbors.loc[neighbors["site_index"] == site_index]
        site_df.reset_index(inplace=True, drop=True)
    
        # Get to the closest neighbor to the site that isn't a He dummy atom
        for i, row in site_df.iterrows():
            site_cart_coords = row["site_coords"]
            neigh_cart_coords = row["neigh_coords"]
            neighbor_string = row["neigh_symbol"]
            neigh_index = row["neigh_index"]
            if neighbor_string != "E":
                bond_dist = row["dist"]
                break
    
        # Interpolate the elf along this line
        (
            elf_positions,
            elf_values,
            label_values,
        ) = self.get_partitioning_line_from_cart_coords(
            site_cart_coords,
            neigh_cart_coords,
        )
    
        # Make sure we don't have only assignments to a single site. If we do
        # we want to place our radius right at the middle.
        if len(np.unique(label_values)) == 1:
            bond_frac = 0.5
            distance_to_min = bond_frac * bond_dist
            return distance_to_min
    
        # Now we check if there is a covalent bond along our line
        covalent = False
        for label in np.unique(label_values):
            if labeled_structure[label].specie.symbol == "Z":
                covalent = True
                break
    
        # If there is, we want to use the maximum closest to the center as our
        # radius
        if covalent:
            # we find the closest maximum to the center
            maxima = self.find_maximum(elf_values)
            unrelated_indices = np.where(
                ~np.isin(label_values, [site_index, neigh_index])
            )
            new_maxima = []
            for maximum in maxima:
                if (
                    np.isin(maximum[0], unrelated_indices)
                    and (maximum[1] - min(elf_values)) > 0.01
                ):
                    new_maxima.append(maximum)
            if len(new_maxima) > 0:
                elf_min_index = self.get_closest_extrema_to_center(
                    elf_values, new_maxima
                )[0]
                extrema = "max"
            else:
                elf_min_index = np.where(np.array(label_values) == site_index)[0].max()
                extrema = "min"
    
        else:
            # We want to use the standard ionic radius, or the first point where
            # we no longer have a basin related to our atom
            try:
                elf_min_index = np.where(np.array(label_values) != site_index)[0][0] - 1
                extrema = "min"
            except:
                raise Exception(
                    f"No radius could be found for atom index {site_index}. This can"
                    " result from using too few valence electrons in your PPs. If you"
                    " are sure this is not the case, please contact our team."
                )
    
        # refine the location of the radius
        try:
            global_min = self._refine_line_part_frac(
                positions=elf_positions,
                elf_min_index=elf_min_index,
                extrema=extrema,
                method=refine_method,
            )
            distance_to_min = global_min[2] * bond_dist
        except:
            breakpoint()
            bond_frac = elf_min_index / (len(elf_positions) - 1)
            logging.warning(
                f"Refinement of radius failed. Unrefined bond fraction of {bond_frac} will be used."
            )
    
            distance_to_min = bond_frac * bond_dist
    
        return distance_to_min