# ElfAnalyzer

ElfAnalyzer is a package designed to assist in the topological analysis of the
Electron Localization Function (ELF). It is designed for a grid representation
of the ELF, particularly from *ab initio* codes that utilzie pseudopotentials
such as [VASP](https://www.vasp.at/).

The project builds off of our package for performing Bader charge analysis, [Baderkit](https://github.com/SWeav02/baderkit).
It performs an initial Bader analysis to separate out ELF domains, then assigns
each basin to chemical features such as atom cores/shells, covalent bonds, metal bonds,
or bare electrons. These assignments are made using a series of filters related
to the basins ELF value, position, and shape.

ElfAnalyzer is currently a work in progress as we pull the core functionality from its
original home in [Simmate](https://github.com/jacksund/simmate) and make it more
user friendly.