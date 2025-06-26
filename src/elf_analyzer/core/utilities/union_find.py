# -*- coding: utf-8 -*-

class UnionFind:
    """
    Simple union finding class from chatgpt.
    """
    def __init__(self):
        self.parent = {}

    def find(self, x):
        # Path compression
        if x != self.parent.setdefault(x, x):
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # Union by root
        self.parent[self.find(x)] = self.find(y)

    def groups(self):
        from collections import defaultdict
        comps = defaultdict(set)
        for item in self.parent:
            root = self.find(item)
            comps[root].add(item)
        return list(comps.values())
    