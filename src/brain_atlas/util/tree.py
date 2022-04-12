from __future__ import annotations

from collections import defaultdict
from typing import Counter, Dict, List

from brain_atlas import Key


class MultiNode:
    """
    A ClusterNode-like tree structure that allows for more than two children per node
    """

    def __init__(
        self, index: int, node_id: Key, children: List[MultiNode] = None, count: int = 0
    ):
        self.index = index
        self.node_id = node_id
        self.children = children
        if self.children is None:
            self.count = count
            self.node_count = 1
        else:
            self.count = count + sum(c.count for c in self.children)
            self.node_count = 1 + sum(c.node_count for c in self.children)

    @property
    def is_leaf(self):
        return self.children is None

    def pre_order(self, include_internal: bool = False, func=(lambda x: x.node_id)):
        cur_node = [self]
        cur_node.extend(None for _ in range(self.node_count))
        visited = set()

        k = 0
        preorder = []
        while k >= 0:
            nd = cur_node[k]

            if nd.is_leaf:
                preorder.append(func(nd))
                k = k - 1
            else:
                if nd.node_id not in visited:
                    if include_internal:
                        preorder.append(func(nd))

                    # reversing the list because they will be visited backwards
                    for i, nd_c in enumerate(nd.children[::-1]):
                        cur_node[k + i + 1] = nd_c

                    visited.add(nd.node_id)
                    k = k + len(nd.children)
                else:
                    k = k - 1

        return preorder


NodeTree = Dict[Key, MultiNode]


def to_tree(leaf_list: List[Key], node_counts: Counter[Key] = None) -> NodeTree:
    node_list = sorted(leaf_list)
    node_tree = dict()

    if node_counts is None:
        # leaf nodes are count 1, everything else 0
        node_counts = defaultdict(int, {k: 1 for k in leaf_list})

    # internal nodes
    all_nodes = {k[:i] for k in leaf_list for i in range(len(k))}
    # this sorts the nodes to be in bottom-up order
    node_list.extend(sorted(all_nodes, key=lambda k: (-len(k), k)))

    node_children = defaultdict(list)
    for k in node_list[:-1]:
        node_children[k[:-1]].append(k)

    # initialize node objects from the bottom up
    for i, k in enumerate(node_list):
        if i < len(leaf_list):
            node_tree[k] = MultiNode(i, k, count=node_counts[k])
        else:
            children = [node_tree[c_k] for c_k in sorted(node_children[k])]
            node_tree[k] = MultiNode(i, k, count=node_counts[k], children=children)

    return node_tree


def calc_node_depth(node_tree: NodeTree) -> Dict[Key, int]:
    node_depth = dict()

    # add up depth, from bottom
    for k in sorted(node_tree, key=lambda key: node_tree[key].index):
        if node_tree[k].is_leaf:
            node_depth[k] = 0
        else:
            node_depth[k] = (
                max(node_depth[nd.node_id] for nd in node_tree[k].children) + 1
            )

    return node_depth
