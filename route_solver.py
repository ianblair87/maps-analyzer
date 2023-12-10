from quadtree import find_node
import heapq
import numpy as np
import cv2

def astar(g, start, finish):
    def findist(x, y, finish):
        return np.sqrt((x - finish[0]) ** 2 + (y - finish[1]) ** 2)
    points = []
    dist = np.full(g.shape, 10 ** 10)
    start[0] = int(start[0])
    start[1] = int(start[1])
    finish[0] = int(finish[0])
    finish[1] = int(finish[1])
    dist[start[0], start[1]] = 0
    prev = {}
    used = set()
    
    heapq.heappush(points, (findist(start[0], start[1], finish), (start[0], start[1])))
    while len(points) > 0:
        _, v = heapq.heappop(points)
        if v in used:
            continue
        if v[0] == finish[0] and v[1] == finish[1]:
            break
        x, y = v
        used.add(v)
        for dx in [0, -1, 1, 2, -2, 3, -3]:
            for dy in [0, 1, -1, 2, -2, 3, -3]:
                if x + dx < dist.shape[0] and x + dx >= 0 and y + dy < dist.shape[1] and y + dy >= 0 and dist[x + dx, y + dy] > dist[v] + np.sqrt(dx ** 2 + dy ** 2) and g[x + dx, y + dy] != 1:
                    dist[x + dx, y + dy] = dist[v] + np.sqrt(dx ** 2 + dy ** 2)
                    prev[(x + dx, y + dy)] = v
                    heapq.heappush(points, (dist[x + dx, y + dy] + findist(x + dx, y + dy, finish), (x + dx, y + dy)))
    return prev

def a_star_quadtree(c1, c2, tree, graph):
    n1 = find_node(tree.root, *c1).id
    n2 = find_node(tree.root, *c2).id
    q = []
    dist = {}
    pre = {}
    dist[n1] = 0
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
            val = ((tree.nodes[u].b - tree.nodes[u].t) + (tree.nodes[u].r - tree.nodes[u].l)) / 2
            if tree.nodes[u].val == 0:
                can += val
            elif tree.nodes[u].val == 1:
                can += val * 100000
            if can < curd:
                dist[u] = can
                pre[u] = v
                heapq.heappush(q, (can, u)) 
    return (n1, n2, pre)

def get_best_routes(runnable, tree, graph, legs):
    res = np.zeros((tree.root.r, tree.root.b, 3), dtype=np.float64)
    for fr, to in legs:
        pre = astar(runnable, fr, to)
        # n1, n2, pre = a_star_quadtree((fr[0], fr[1]), (to[0], to[1]), tree, graph)
        # node = n2
        cur = (to[0], to[1])
        while cur in pre and pre[cur][0] != -1:
            a = cur
            cur = pre[cur]
            b = cur
            cv2.line(res, (int(a[1]), int(a[0])), (int(b[1]), int(b[0])), (0.1, 0.7, 0.2), 10)
    return res