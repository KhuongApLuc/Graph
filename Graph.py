import os
os.system('cls')
# 1    Create text files to store the adjacency matrix of a graph in Figure 1. Write the Graph class in Python with the following members :
# Data members:
#   a  -  two dimentional array representing an adjacency matrix
#   label - label of vertices
#   n - number of vertices.
# Methods :
# def setAMatrix(b, m) - set m to n and b matrix to adjancy matrix.
# def setLabel(c) - set labels for vertices
# and two methods for breadth first traverse and depth first traverse.
# 2.   Write the WGraph class which contains weighted matrix and methods for Dijkstra shortest path algorithm.
# 3.   Write the WGraph class which contains weighted matrix and methods for finding the minimum spanning tree of a graph.
# 4.   Write the Graph class which contains adjacency matrix and methods for assigning colors to vertices with the sequential coloring algorithm.

#mũi tên biểu thị hướng đi từ node viết dọc đến node viết ngang
class Graph1:
    def __init__(self, n):
        self.n = n
        self.a = [[0] * n for _ in range(n)]
        self.label = [None] * n

    def setAMatrix(self, b, m):
        self.n = m
        self.a = b

    def setLabel(self, c):
        if len(c) != self.n:
            raise ValueError("Number of labels doesn't match the number of vertices.")
        self.label = c

    def breadthFirstTraverse(self, start):
        visited = [False] * self.n
        queue = [start]
        visited[start] = True

        while queue:
            vertex = queue.pop(0)
            print(self.label[vertex])

            for i in range(self.n):
                if self.a[vertex][i] and not visited[i]:
                    queue.append(i)
                    visited[i] = True

    def depthFirstTraverse(self, start):
        visited = [False] * self.n
        self._depthFirstUtil(start, visited)

    def _depthFirstUtil(self, vertex, visited):
        visited[vertex] = True
        print(self.label[vertex])

        for i in range(self.n):
            if self.a[vertex][i] and not visited[i]:
                self._depthFirstUtil(i, visited)

from heapq import heappop, heappush

class WGraph1:
    def __init__(self, n):
        self.n = n
        self.w = [[float('inf')] * n for _ in range(n)]

    def setWeightMatrix(self, weights):
        if len(weights) != self.n or any(len(row) != self.n for row in weights):
            raise ValueError("Invalid weight matrix dimensions.")
        self.w = weights

    def dijkstraShortestPath(self, start):
        dist = [float('inf')] * self.n
        dist[start] = 0

        pq = [(0, start)]

        while pq:
            cost, vertex = heappop(pq)

            if cost > dist[vertex]:
                continue

            for i in range(self.n):
                if self.w[vertex][i] != float('inf'):
                    new_cost = cost + self.w[vertex][i]
                    if new_cost < dist[i]:
                        dist[i] = new_cost
                        heappush(pq, (new_cost, i))

        return dist

class WGraph2:
    def __init__(self, n):
        self.n = n
        self.w = [[float('inf')] * n for _ in range(n)]

    def setWeightMatrix(self, weights):
        if len(weights) != self.n or any(len(row) != self.n for row in weights):
            raise ValueError("Invalid weight matrix dimensions.")
        self.w = weights

    def findMinimumSpanningTree(self):
        parent = [-1] * self.n
        key = [float('inf')] * self.n
        mstSet = [False] * self.n

        key[0] = 0
        parent[0] = -1

        for _ in range(self.n):
            min_key = float('inf')
            min_index = -1

            for v in range(self.n):
                if not mstSet[v] and key[v] < min_key:
                    min_key = key[v]
                    min_index = v

            mstSet[min_index] = True

            for i in range(self.n):
                if (
                    0 < self.w[min_index][i] < key[i] and
                    not mstSet[i]
                ):
                    parent[i] = min_index
                    key[i] = self.w[min_index][i]

        return parent
    
class Graph2:
    def __init__(self, n):
        self.n = n
        self.a = [[0] * n for _ in range(n)]

    def setAMatrix(self, b, m):
        self.n = m
        self.a = b

    def assignColors(self):
        colors = [-1] * self.n
        available_colors = [True] * self.n

        colors[0] = 0

        for v in range(1, self.n):
            for i in range(self.n):
                if self.a[v][i] and colors[i] != -1:
                    available_colors[colors[i]] = False

            for color in range(self.n):
                if available_colors[color]:
                    colors[v] = color
                    break

            available_colors = [True] * self.n

        return colors

def get_content(): 
    with open('Graph.txt', 'r') as f:
        data = f.readlines()

    cleaned_matrix = [] 
    for raw_line in data:
        split_line = raw_line.strip().split(",")
        nums_ls = [int(x.replace('"', '')) for x in split_line] 
        cleaned_matrix.append(nums_ls)
    return cleaned_matrix
  
# Tạo một đối tượng Graph với số đỉnh là 4
graph = Graph1(7)

# Thiết lập ma trận kề cho đồ thị
adj_matrix = [
    [0,1,1,1,0,0,1],
    [1,0,1,0,1,0,0],
    [1,1,0,0,1,0,0],
    [1,0,0,0,0,1,0],
    [0,1,1,0,0,1,1],
    [0,0,0,1,1,0,0],
    [1,0,0,0,1,0,0]
]
graph.setAMatrix(get_content(), 7)

# Thiết lập nhãn cho các đỉnh
labels = ["A", "B", "C", "D","E","F","G"]
graph.setLabel(labels)

# Thực hiện duyệt theo chiều rộng (breadth-first traverse)
print("Breadth-First Traversal:", end = '')
graph.breadthFirstTraverse(0)

# Thực hiện duyệt theo chiều sâu (depth-first traverse)
print("\nDepth-First Traversal:", end = '')
graph.depthFirstTraverse(0)

# Tạo một đối tượng WGraph với số đỉnh là 5
wgraph = WGraph1(7)
#----------------------------------------------
# Thiết lập ma trận trọng số cho đồ thị
weight_matrix = [
    [0,1,1,1,0,0,1],
    [1,0,1,0,1,0,0],
    [1,1,0,0,1,0,0],
    [1,0,0,0,0,1,0],
    [0,1,1,0,0,1,1],
    [0,0,0,1,1,0,0],
    [1,0,0,0,1,0,0]
]
wgraph.setWeightMatrix(weight_matrix)

# Thực hiện thuật toán Dijkstra để tìm đường đi ngắn nhất
start_vertex = 0
distances = wgraph.dijkstraShortestPath(start_vertex)

print(f"Shortest distances from vertex {start_vertex}:")
for i in range(len(distances)):
    print(f"Vertex {i}: {distances[i]}")




            