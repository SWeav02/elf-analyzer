# -*- coding: utf-8 -*-

"""
This file defines options for feature labels and settings for colors and
dummy atoms
"""

from enum import Enum
import logging

FEATURE_COLORS = {
    "root": "rgba(128, 128, 128, 1)",  # grey
    "reducible": "rgba(128, 128, 128, 1)",  # grey
    "dimension reduction": "rgba(154, 128, 128, 1)",  # redish grey
    "contained atom reduction": "rgba(128, 128, 154, 1)",  # bluish grey
    "irreducible": "rgba(47, 79, 79, 1)", # dark slate gray
    "shallow": "rgba(47, 79, 79, 1)", # dark slate gray
    "shell": "rgba(60, 60, 60, 1)", # dark grey
    "deep shell": "rgba(60, 60, 60, 1)", # dark grey
    "core": "rgba(0, 0, 0, 1)", # black
    "covalent": "rgba(0, 255, 255, 1)",  # aqua
    "metallic": "rgba(112, 128, 144, 1)", # slate gray
    "lone-pair": "rgba(128, 0, 128, 1)",  # purple
    "non-nuclear attractor": "rgba(128, 0, 0, 1)",  # maroon
    "electride": "rgba(170, 0, 0, 1)",  # dark red
    }

FEATURE_DUMMY_ATOMS = {
    "shell": "Xs",
    "deep shell": "Xds",
    "core": "Xc",
    "covalent": "Z",
    "metallic": "M",
    "lone-pair": "Lp",
    "non-nuclear attractor": "Xn",
    "electride": "E",
    }

LINE_COLOR = "rgba(128, 128, 128, 1)" # grey

class classproperty(property):
    def __get__(self, obj, cls):
        return self.fget(cls)


class FeatureSubtype(str, Enum):
    reducible = "reducible"
    root = "root"
    dim_reduction = "dimension reduction"
    contained_atom_reduction = "contained atom reduction"
    irreducible = "irreducible"
    shallow = "shallow"
    shell = "shell"
    deep_shell = "deep shell"
    core = "core"
    covalent = "covalent"
    metallic = "metallic"
    lone_pair = "lone-pair"
    non_nuclear_attractor = "non-nuclear attractor"
    electride = "electride"

    @classproperty
    def reducible_types(cls):
        return [
            cls.root,
            cls.reducible,
            cls.dim_reduction,
            cls.contained_atom_reduction,
        ]

    @classproperty
    def irreducible_types(cls):
        return [
            cls.irreducible,
            cls.shallow,
            cls.shell,
            cls.deep_shell,
            cls.core,
            cls.covalent,
            cls.metallic,
            cls.lone_pair,
            cls.non_nuclear_attractor,
            cls.electride,
        ]

    @classproperty
    def atomic_types(cls):
        return [cls.shell, cls.deep_shell, cls.core]

    @classproperty
    def valence_types(cls):
        return [
            cls.covalent,
            cls.metallic,
            cls.lone_pair,
            cls.non_nuclear_attractor,
            cls.electride,
        ]

    @classproperty
    def subtypes(cls):
        return {
            "ReducibleNode": cls.reducible_types,
            "IrreducibleNode": cls.irreducible_types,
        }    
    
    @property
    def plot_color(self):
        color = FEATURE_COLORS.get(self.value, None)
        if color is None:
            logging.warning(f"No plot color found for feature of type {self.name}")
        return color
    
    @property
    def dummy_species(self):
        species = FEATURE_DUMMY_ATOMS.get(self.value, None)
        if species is None:
            logging.warning(f"No dummy species label found for feature of type {self.name}. Using 'X'")
            return "X"
        return species
