from __future__ import annotations

from typing import List

from brain_atlas import Key


class MultiNode:
    """
    A ClusterNode-like tree structure that allows for more than two children per node
    """

    def __init__(self, node_id: Key, children: List[MultiNode] = None, count: int = 1):
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
        # maximum number of nodes is 2*n, but might be less
        cur_node = [self]
        cur_node.extend(None for _ in range(2 * self.count - 1))
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


def to_tree(leaf_list: List[Key]):
    node_list = leaf_list.copy()
    node_dict = dict()
    node_depth = dict()

    # internal nodes
    all_nodes = {k[:i] for k in leaf_list for i in range(len(k))}
    # this sorts the nodes to be in bottom-up order
    node_list.extend(sorted(all_nodes, key=lambda k: (-len(k), k)))

    # initialize node objects
    for i, k in enumerate(node_list):
        if i >= len(leaf_list):
            node_dict[k] = MultiNode(k, children=[])
        else:
            node_dict[k] = MultiNode(k)

    # add up counts and depth, from bottom
    for k in node_list:
        if len(k):
            node_dict[k[:-1]].children.append(node_dict[k])
            node_dict[k[:-1]].count += node_dict[k].count

        if node_dict[k].is_leaf:
            node_depth[k] = 0
        else:
            node_depth[k] = (
                max(node_depth[nd.node_id] for nd in node_dict[k].children) + 1
            )

    # sort the children by node id
    for k in node_dict:
        if not node_dict[k].is_leaf:
            node_dict[k].children.sort(key=lambda nd: nd.node_id)

    return node_list, node_depth, node_dict
