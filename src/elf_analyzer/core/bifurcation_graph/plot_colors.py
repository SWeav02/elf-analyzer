# -*- coding: utf-8 -*-

"""
This file defines plot coloring defaults for different node types
"""

NODE_COLORS = {
    "root": "rgba(128, 128, 128, 1)",  # grey
    "reducible": "rgba(128, 128, 128, 1)",  # grey
    "dim_change": "rgba(154, 128, 128, 1)",  # redish grey
    "atom_change": "rgba(128, 128, 154, 1)",  # bluish grey
    "irreducible": "rgba(47, 79, 79, 1)", # dark slate gray
    "shallow": "rgba(47, 79, 79, 1)", # dark slate gray
    "shell": "rgba(60, 60, 60, 1)", # dark grey
    "core": "rgba(0, 0, 0, 1)", # black
    "covalent": "rgba(0, 255, 255, 1)",  # aqua
    "metallic": "rgba(112, 128, 144, 1)", # slate gray
    "lone-pair": "rgba(128, 0, 128, 1)",  # purple
    "bare electron": "rgba(128, 0, 0, 1)",  # maroon
    }

LINE_COLOR = "rgba(128, 128, 128, 1)" # grey