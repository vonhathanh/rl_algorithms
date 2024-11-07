import numpy as np


def init(tree: np.ndarray, data: np.ndarray, i: int, l: int, r: int):
    """
    initialize the tree, we must initialize the leaves first then the n-1th layer..., 0th layer (root)
    The initialization procedure usually implemented by recursive calls:
    init(root) = init(left_branch) + init(right_branch)
    init(node) returns the value of the node
    besides `node`, init must take two additional params: l, r -> the range that a node covers (left, right)
    if l == r -> it's a leaf node
    else: left_node_idx = root * 2 + 1, right_node_idx = root * 2 + 2
    mid = (l + 2) // 2
    tree[root] = init(left_node_idx, l, mid), init(right_node_idx, mid+1, r)
    """
    if l == r:
        tree[i] = data[l]
    else:
        mid = (l + r) // 2
        tree[i] = init(tree, data, i*2+1, l, mid) + init(tree, data, i*2+2, mid+1, r)
    return tree[i]

def update_tree(tree: np.ndarray, data: np.ndarray, diff: float, i: int):
    # update value of data[i] by `diff`
    assert 0 <= i < len(data)
    data[i] += diff
    # update value of branches i belongs to
    update_branch(tree, diff, i, 0, 0, len(data) - 1)

def update_branch(tree: np.ndarray, diff: float, data_idx: int, tree_idx: int, l: int, r: int):
    # rules for update tree's branches
    # update value of the tree at tree_idx first
    tree[tree_idx] += diff
    # leaf node, it's value is updated on the line above, stop
    if l == r:
        return

    mid = (l + r) // 2
    # data_idx in (l, mid), update sum of the left branch
    if data_idx <= mid:
        update_branch(tree, diff, data_idx, tree_idx * 2 + 1, l, mid)
    # data_idx in (mid+1, right) update sum of the right branch
    else:
        update_branch(tree, diff, data_idx, tree_idx * 2 + 2, mid+1, r)

def query_sum(tree, i, li, ri, l, r):
    # Query sum in a specific range (l, r):
    # case 1: l and r have the same root
    if li == l and ri == r:
        return tree[i]
    # case 2: current range (li, ri) is outside of query branch (l, r), stop
    if ri < l or li > r:
        return 0
    # case 3: leaf node, return node's value
    if li == ri:
        return tree[i]
    # case 4: split the current range into two smaller ranges
    mid = (li + ri) // 2
    return query_sum(tree, i*2+1, li, mid, l, r) + query_sum(tree, i*2+2, mid+1, ri, l, r)


data = np.array([1, 3, -2, 8, -7])
tree = np.zeros(len(data) * 4)

init(tree, data, 0, 0, len(data)-1)
print(tree)

# update_tree(tree, data, -2, 4)
# print(tree)

print(query_sum(tree, 0, 0, len(data)-1, 0, 3))