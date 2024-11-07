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

def query_sum(tree, li, ri, l, r, i=0):
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
    return query_sum(tree, li, mid, l, r, i*2+1) + query_sum(tree, mid+1, ri, l, r, i*2+2)


def test_init_tree(input, output):
    tree = np.zeros(len(input) * 4)
    init(tree, input, 0, 0, len(input) - 1)
    assert np.all(tree == output)
    return input, tree

def test_update(input, diff, i):
    tree = np.zeros(len(input) * 4)
    init(tree, input, 0, 0, len(input) - 1)
    prev_sum = np.sum(input)
    update_tree(tree, input, diff, 0)
    new_sum = np.sum(input)

    assert prev_sum + diff == new_sum
    assert prev_sum + diff == query_sum(tree, 0, len(input)-1, 0, len(input) - 1, i)

def tests():
    # test with even-length array
    data, tree = test_init_tree(np.array([1, 3, -2, 8, -7, 3]),
                                np.array([ 6, 2, 4, 4,-2, 1, 3, 1, 3, 0, 0, 8,-7, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0]))
    # test update at the start index
    test_update(data, 3, 0)
    # new_sum = np.sum(data)
    # assert new_sum == sum_data - 2
    # sum_data = new_sum
    # print(tree)
    # # test update at the end index
    # update_tree(tree, data, 3, len(data) - 1)
    # new_sum = np.sum(data)
    # assert new_sum == sum_data + 3
    # sum_data = new_sum
    # print(tree)
    # # test update at the middle
    # update_tree(tree, data, 5, len(data) // 2)
    # new_sum = np.sum(data)
    # assert new_sum == sum_data + 5
    # print(tree)
    # # test query sum in full range
    # s = query_sum(tree, 0, len(data)-1, 0, len(data)-1)
    # assert s == np.sum(data)
    # # test query sum in single leaf
    # s = query_sum(tree, 0, len(data) - 1, 0, 0)
    # assert s == data[0]
    # # test query sum in single leaf
    # s = query_sum(tree, 0, len(data) - 1, len(data)-1, len(data)-1)
    # assert s == data[-1]
    # # test query sum in random range
    # s = query_sum(tree, 0, len(data) - 1, 0, 3)
    # assert s == np.sum(data[0:4])
    # # test query sum in random range
    # s = query_sum(tree, 0, len(data) - 1, 2, len(data)-1)
    # assert s == np.sum(data[2:])
    #
    # # test with odd-length array
    # data = np.array([1, 3, -2, 8, -7])
    # tree = np.zeros(len(data) * 4)
    # init(tree, data, 0, 0, len(data) - 1)
    # sum_data = np.sum(data)
    # print(tree)
    # # test update at the start index
    # update_tree(tree, data, -2, 0)
    # new_sum = np.sum(data)
    # assert new_sum == sum_data - 2
    # sum_data = new_sum
    # print(tree)
    # # test update at the end index
    # update_tree(tree, data, 3, len(data) - 1)
    # new_sum = np.sum(data)
    # assert new_sum == sum_data + 3
    # sum_data = new_sum
    # print(tree)
    # # test update at the middle
    # update_tree(tree, data, 5, len(data) // 2)
    # new_sum = np.sum(data)
    # assert new_sum == sum_data + 5
    # print(tree)
    #
    # print(query_sum(tree, 0, 0, len(data) - 1, 0, 3))
    # # test query sum in full range
    # print(query_sum(tree, 0, len(data) - 1, 0, len(data) - 1))
    # # test query sum in single leaf
    # print(query_sum(tree, 0, len(data) - 1, 0, 0))
    # # test query sum in single leaf
    # print(query_sum(tree, 0, len(data) - 1, len(data) - 1, len(data) - 1))
    # # test query sum in random range
    # print(query_sum(tree, 0, len(data) - 1, 0, 3))
    # # test query sum in random range
    # print(query_sum(tree, 0, len(data) - 1, 2, len(data)))
    #
    # # test with array length = 2**n
    # data = np.array([1, 3, -2, 8, -7, 3, 2, 5])
    # tree = np.zeros(len(data) * 4)
    # init(tree, data, 0, 0, len(data) - 1)
    # print(tree)
    # # test update at the start index
    # update_tree(tree, data, -2, 0)
    # print(tree)
    # # test update at the end index
    # update_tree(tree, data, 3, len(data) - 1)
    # print(tree)
    # # test update at the middle
    # update_tree(tree, data, 5, len(data) // 2)
    # print(tree)
    # # test query sum in full range
    # print(query_sum(tree, 0, len(data) - 1, 0, len(data) - 1))
    # # test query sum in single leaf
    # print(query_sum(tree, 0, len(data) - 1, 0, 0))
    # # test query sum in single leaf
    # print(query_sum(tree, 0, len(data) - 1, len(data) - 1, len(data) - 1))
    # # test query sum in random range
    # print(query_sum(tree, 0, len(data) - 1, 0, 3))
    # # test query sum in random range
    # print(query_sum(tree, 0, len(data) - 1, 2, len(data)))

if __name__ == '__main__':
    tests()