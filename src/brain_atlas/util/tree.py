from __future__ import annotations

from typing import List, Optional, Tuple


class ClusterNode:
    def __init__(
        self, node_id: int, children: List[ClusterNode] = None, count: int = 1
    ):
        if node_id < 0:
            raise ValueError("The id must be non-negative.")
        if count < 1:
            raise ValueError(
                "A cluster must contain at least one original observation."
            )
        self.node_id = node_id
        self.children = children
        if self.children is None:
            self.count = count
        else:
            self.count = sum(c.count for c in self.children)

    @property
    def is_leaf(self):
        return self.children is None

    def pre_order(self, include_internal: bool = False, func=(lambda x: x.node_id)):
        n = self.count

        # maximum number of nodes is 2*n+1, but might be less
        cur_node: List[Optional[ClusterNode]] = [None] * (2 * n)
        visited = set()
        cur_node[0] = self
        k = 0
        preorder = []
        while k >= 0:
            nd = cur_node[k]
            ndid = nd.node_id
            if nd.is_leaf:
                preorder.append(func(nd))
                k = k - 1
            else:
                if ndid not in visited:
                    if include_internal:
                        preorder.append(func(nd))
                    # reversing the list because they will be visited backwards
                    for i, nd_c in enumerate(nd.children[::-1]):
                        cur_node[k + i + 1] = nd_c
                    visited.add(ndid)
                    k = k + len(nd.children)
                else:
                    k = k - 1

        return preorder


def to_tree(leaf_list: List[Tuple[int]]):
    node_list = leaf_list.copy()

    n = len(leaf_list)
    node_dict = dict()
    node_depth = dict()

    # internal nodes
    all_nodes = {k[:i] for k in leaf_list for i in range(len(k))}
    # this sorts the nodes to be in bottom-up order
    node_list.extend(sorted(all_nodes, key=lambda k: (-len(k), k)))

    k2i = {k: i for i, k in enumerate(node_list)}

    # initialize node objects
    for i in range(len(node_list)):
        if i >= n:
            node_dict[i] = ClusterNode(i, children=[])
        else:
            node_dict[i] = ClusterNode(i)

    # add up counts and depth, from bottom
    for i, node in enumerate(node_list):
        if len(node):
            node_dict[k2i[node[:-1]]].children.append(node_dict[i])
            node_dict[k2i[node[:-1]]].count += node_dict[i].count

        if node_dict[i].is_leaf:
            node_depth[i] = 0
        else:
            node_depth[i] = (
                max(node_depth[n.node_id] for n in node_dict[i].children) + 1
            )

    # sort the children by node id
    for i in node_dict:
        if not node_dict[i].is_leaf:
            node_dict[i].children.sort(key=lambda nd: nd.node_id)

    return node_list, node_depth, node_dict
