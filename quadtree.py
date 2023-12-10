import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
import tqdm

class Node:
    def __init__(self, l, t, r, b, p):
        self.l = l
        self.t = t
        self.r = r
        self.b = b
        self.x = (l + r) / 2
        self.y = (b + t) / 2
        self.val = -1.
        self.parent = p
        self.children = []
        self.id = hash(self)
        
class Tree:
    def __init__(self,root):
        self.root = root
        self.nodes = {}

def build_quadtree(img):
    root = Node(0, 0, img.shape[0], img.shape[1], None)
    tree = Tree(root)
    q = [root]
    while len(q) > 0:
        node = q[0]
        q = q[1:]
        tree.nodes[node.id] = node
        total = 0
        s = 0
        for i in range(node.l, node.r):
            for j in range(node.t, node.b):
                total += 1
                if img[i][j]:
                    s += 1
        if s == 0:
            node.val = 0
            continue
        if s > total * 0.3:
            node.val = 1
            continue
        xmid = (node.l + node.r) // 2
        ymid = (node.t + node.b) // 2
        node.children = [
            Node(node.l, node.t, xmid, ymid, node),
            Node(xmid, node.t, node.r, ymid, node),
            Node(node.l, ymid, xmid, node.b, node),
            Node(xmid, ymid, node.r, node.b, node),
        ]
        for x in node.children:
            q.append(x)
    return tree

def draw_quadtree(node):
    res = np.zeros((node.r, node.b))
    def draw_node(node, res):
        if node.val != -1:
            if node.val == 1:
                cv2.rectangle(res, [node.t, node.l], [node.b, node.r], 1, -1)
            return 1
        s = 0
        for x in node.children:
            s += draw_node(x, res)
        return s
    draw_node(node, res)
    plt.imshow(res)
    plt.show()


def build_graph_from_quadtree(root, graph):
    if root.val != -1:
        return [root], [root], [root], [root]
    NW_l, NW_t, NW_r, NW_b =  build_graph_from_quadtree(root.children[0], graph)
    NE_l, NE_t, NE_r, NE_b =  build_graph_from_quadtree(root.children[1], graph)
    SW_l, SW_t, SW_r, SW_b =  build_graph_from_quadtree(root.children[2], graph)
    SE_l, SE_t, SE_r, SE_b =  build_graph_from_quadtree(root.children[3], graph)
    
    def make_edge(i, j, graph):
        graph[i] = graph.get(i, []) + [j]
        graph[j] = graph.get(j, []) + [i]
    
    def merge_by_y(L, R, graph):
        ptr_a = 0
        ptr_b = 0
        while ptr_a < len(L) and ptr_b < len(R):
            ymin = max(L[ptr_a].t, R[ptr_b].t)
            ymax = min(L[ptr_a].b, R[ptr_b].b)
            if ymin < ymax:
                make_edge(L[ptr_a].id, R[ptr_b].id, graph)
            if L[ptr_a].b < R[ptr_b].b:
                ptr_a += 1
            else:
                ptr_b += 1

    def merge_by_x(T, B, graph):
        ptr_a = 0
        ptr_b = 0
        while ptr_a < len(T) and ptr_b < len(B):
            xmin = max(T[ptr_a].l, B[ptr_b].l)
            xmax = min(T[ptr_a].r, B[ptr_b].r)
            if xmin < xmax:
                make_edge(T[ptr_a].id, B[ptr_b].id, graph)
            if T[ptr_a].r < B[ptr_b].r:
                ptr_a += 1
            else:
                ptr_b += 1

    merge_by_y(NW_r, NE_l, graph)
    merge_by_y(SW_r, SE_l, graph)
    merge_by_x(NW_b, SW_t, graph)
    merge_by_x(NE_b, SE_t, graph)    
    return NW_l + SW_l, NW_t + NE_t, NE_r + SE_r, SW_b + SE_b


def draw_quadtree_graph(tree, graph):
    draw_quadtree(tree.root)
    res = np.zeros((tree.root.r, tree.root.b))
    cnt = 0
    for v in graph:
        a = tree.nodes[v]
        cv2.rectangle(res, [a.t, a.l], [a.b, a.r], 255, 2)
        for u in graph[v]:    
            b = tree.nodes[u]
            cv2.line(res, (int(a.y), int(a.x)), (int(b.y), int(b.x)), 255, 3)
            cnt += 1
    print(f"total {cnt} edges and {len(graph)} vertices")
    plt.imshow(res)
    plt.show()
            

def find_node(root, x, y):
    if root.val != -1:
        return root
    xmid = root.x
    ymid = root.y
    if x < xmid and y < ymid:
        return find_node(root.children[0], x, y)
    if x >= xmid and y < ymid:
        return find_node(root.children[1], x, y)
    if x < xmid and y >= ymid:
        return find_node(root.children[2], x, y)
    if x >= xmid and y >= ymid:
        return find_node(root.children[3], x, y)
    
def quadtree_penalty(c1_idx, c2_idx, circles, img, tree, graph, visualize=False):
    n1 = find_node(tree.root, *circles[c1_idx]).id
    n2 = find_node(tree.root, *circles[c2_idx]).id
    c1 = circles[c1_idx]
    c2 = circles[c2_idx]
    q = []
    dist = {}
    pre = {}
    dist[n1] = 0
    approx_metric = np.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2) * 0.9
    heapq.heappush(q, (0, n1))
    while len(q) > 0:
        d, v = heapq.heappop(q)
        if v == n2:
            break
        if v not in graph:
            continue
        for u in graph[v]:
            curd = dist.get(u, 10**9)
            can = d
            area2 = ((tree.nodes[u].b - tree.nodes[u].t) + (tree.nodes[u].r - tree.nodes[u].l)) / 2
            if tree.nodes[u].val == 0:
                can += area2
            elif tree.nodes[u].val == 1:
                can += 0.1 * area2
#             if u != n2:
#                 can += get_angle_impl(tree.nodes[u].x, tree.nodes[u].y, tree.nodes[n2].x, tree.nodes[n2].y, tree.nodes[v].x, tree.nodes[v].y)
            if can < curd:
                dist[u] = can
                pre[u] = v
                heapq.heappush(q, (can, u))
    if visualize:
        res = np.uint8(img * 255)
        node = n2
        while node in pre:
            a = tree.nodes[node]
            node = pre[node]
            b = tree.nodes[node]
            cv2.line(res, (int(a.y), int(a.x)), (int(b.y), int(b.x)), 255, 10)
        plt.imshow(res)
        plt.show()
    return dist.get(n2, 10 ** 9) / (0.1 * approx_metric) - 1

def lines_penalty(c1_idx, c2_idx, circles, img, tree, graph, visualize=False):
    c1 = circles[c1_idx]
    c2 = circles[c2_idx]
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    score = 0
    add_per_bad_pixel = 3
    
    for i in range(int(dist)):
        x = int(c1[0] + dx * i / dist)
        y = int(c1[1] + dy * i / dist)
#         n1 = find_node(tree.root, x,y).id
#         print(x, y, img.shape)
        s = 0
        d = 3
        for v in range(-d, d+1):
            for u in range(-d, d+1):
                s += img[x+v, y+u]
        if s == 0:
            score += add_per_bad_pixel
#             print('add')
    return max(0, score - add_per_bad_pixel * 5) / (add_per_bad_pixel * dist)


def quadtree_dist_matrix(img, circles,tree, graph):
    res = []
    n = len(circles)
    for i in tqdm.tqdm(range(n)):
        res.append([])
        for j in range(n):
            if i == j:
                res[-1].append(0)
                continue
            if i < j:
                quadtree_p = quadtree_penalty(i, j, circles, img, tree, graph, visualize=True)
                lines_p = lines_penalty(i, j, circles, img, tree, graph)
                res[-1].append(min(quadtree_p, lines_p))
            else:
                res[-1].append(res[j][i])
    return res


from scipy.optimize import linear_sum_assignment

from python_tsp.heuristics import solve_tsp_simulated_annealing
from python_tsp.exact import solve_tsp_dynamic_programming


def get_best_permutation(circles, d, img, visualize=False):
    n = len(circles)
    matrix = np.array(d)
#     matrix[:, 0] = 0
    res, distance = solve_tsp_simulated_annealing(matrix)
#     print(res, distance)
    # if visualize:
    #     draw_permutation(img, circles, res)
    return res