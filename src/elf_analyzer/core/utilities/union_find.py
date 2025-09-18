# -*- coding: utf-8 -*-

# class UnionFind:
#     """
#     Simple union finding class from chatgpt.
#     """
#     def __init__(self):
#         self.parent = {}

#     def find(self, x):
#         # Path compression
#         if x != self.parent.setdefault(x, x):
#             self.parent[x] = self.find(self.parent[x])
#         return self.parent[x]

#     def union(self, x, y):
#         # Union by root
#         self.parent[self.find(x)] = self.find(y)

#     def groups(self):
#         from collections import defaultdict
#         comps = defaultdict(set)
#         for item in self.parent:
#             root = self.find(item)
#             comps[root].add(item)
#         return list(comps.values())
    
import numpy as np
from numba import njit #, prange

@njit(cache=True)
def _find(parent, x):
    while x != parent[x]:
        parent[x] = parent[parent[x]]  # Path compression
        x = parent[x]
    return x

@njit(cache=True)
def _union(parent, x, y):
    # if parent isn't long enough for x or y, extend it
    higher = max(x, y)
    while len(parent) <= higher:
        parent.append(len(parent))
        
    rx = _find(parent, x)
    ry = _find(parent, y)
    parent[rx] = ry
    return parent
    
@njit(cache=True)
def _bulk_union(parent, xs, ys):
    for i in range(len(xs)):
        parent = _union(parent, xs[i], ys[i])
    return parent
        
@njit(cache=True)
def _find_roots(parent):
    roots = np.empty(len(parent), dtype=np.int64)
    for i in range(len(parent)):
        roots[i] = _find(parent, i)
    return roots


class UnionFind:
    def __init__(self):
        self.parent = [0]
    
    def find(self, x):
        return _find(self.parent, x)

    def union(self, x, y):
        self.parent = _union(self.parent, x, y)

    def bulk_union(self, xs, ys):
        """Union multiple pairs at once (xs[i], ys[i])"""
        self.parent = _bulk_union(parent=self.parent, xs=xs, ys=ys)

    def groups(self):
        """Return groups as list of arrays"""
        roots = _find_roots(self.parent)
        unique_roots = np.unique(roots)
        # TODO: Replace the following for loop if possible
        return [np.where(roots == r)[0] for r in unique_roots]
    
    def groups_sets(self):
        return {frozenset(s) for s in self.groups()}
