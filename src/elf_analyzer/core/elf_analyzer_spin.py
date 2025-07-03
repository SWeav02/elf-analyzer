# -*- coding: utf-8 -*-

"""
Extends the ElfAnalyzer to spin polarized calculations
"""

from pathlib import Path
import logging

import numpy as np
from pymatgen.core import Structure
from baderkit.core import Grid, Bader
import plotly.graph_objects as go

from elf_analyzer.core.elf_analyzer import ElfAnalyzer
from elf_analyzer.core.utilities import BifurcationGraph

class SpinElfAnalyzer:
    def __init__(
            self,
            elf_grid: Grid,
            charge_grid: Grid,
            **kwargs,
            ):
        # First make sure the grids are actually spin polarized
        assert elf_grid.is_spin_polarized and charge_grid.is_spin_polarized, "ELF must be spin polarized. Use a spin polarized calculation or switch to the ElfAnalyzer class."
        # store the original grid
        self.original_elf_grid = elf_grid
        self.original_charge_grid = charge_grid
        # split the grids to spin up and spin down
        self.elf_grid_up, self.elf_grid_down = elf_grid.split_to_spin()
        self.charge_grid_up, self.charge_grid_down = charge_grid.split_to_spin(
            "charge"
        )
        # check if spin up and spin down are the same
        if np.allclose(
            self.elf_grid_up.total, self.elf_grid_down.total, rtol=0, atol=1e-4
        ):
            logging.info("Spin grids are found to be equal. Only spin-up system will be used for speed.")
            self._equal_spin = True
        else:
            self._equal_spin = False
        # create spin up and spin down elf analyzer instances
        self.elf_analyzer_up = ElfAnalyzer(
            elf_grid=self.elf_grid_up,
            charge_grid=self.charge_grid_up,
            **kwargs,
            )
        if not self._equal_spin:
            self.elf_analyzer_down = ElfAnalyzer(
                elf_grid=self.elf_grid_down, 
                charge_grid=self.charge_grid_down, 
                )
        else:
            self.elf_analyzer_down = self.elf_analyzer_up
        
    
    ###########################################################################
    # Properties for spin up and spin down systems
    ###########################################################################
    
    @property
    def structure(self) -> Structure:
        """
        Shortcut to grid's structure object
        """
        structure = self.original_elf_grid.structure.copy()
        structure.add_oxidation_state_by_guess()
        return structure
    
    @property
    def labeled_structure_up(self) -> Structure:
        return self.elf_analyzer_up.labeled_structure
    
    @property
    def labeled_structure_down(self) -> Structure:
        return self.elf_analyzer_down.labeled_structure
    
    @property
    def bader_up(self) -> Bader:
        return self.elf_analyzer_up.bader
    
    @property
    def bader_down(self) -> Bader:
        return self.elf_analyzer_down.bader
    
    @property
    def bifurcation_graph_up(self) -> BifurcationGraph:
        return self.elf_analyzer_up.bifurcation_graph
    
    @property
    def bifurcation_graph_down(self) -> BifurcationGraph:
        return self.elf_analyzer_down.bifurcation_graph
    
    @property
    def bifurcation_plot_up(self) -> go.Figure:
        return self.elf_analyzer_up.bifurcation_plot
    
    @property
    def bifurcation_plot_down(self) -> go.Figure:
        return self.elf_analyzer_down.bifurcation_plot
    
    def write_plots(self, filename: str | Path):
        filename = Path(filename)
    
        if filename.suffix:
            filename_up = filename.with_name(f"{filename.stem}_up{filename.suffix}")
            filename_down = filename.with_name(f"{filename.stem}_down{filename.suffix}")
        else:
            filename_up = filename.with_name(f"{filename.name}_up")
            filename_down = filename.with_name(f"{filename.name}_down")
    
        self.elf_analyzer_up.write_bifurcation_plot(filename_up)
        self.elf_analyzer_down.write_bifurcation_plot(filename_down)

    
    ###########################################################################
    # From methods
    ###########################################################################
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