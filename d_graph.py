# Course: CS261 - Data Structures
# Author: Ethan Rietz
# Assignment: 6
# Description: Contains a class which represents a directed graph

from collections import deque
import heapq

class DirectedGraph:
    """
    Class to implement directed weighted graph
    - duplicate edges not allowed
    - loops not allowed
    - only positive edge weights
    - vertex names are integers
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency matrix
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.v_count = 0
        self.adj_matrix = []

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            v_count = 0
            for u, v, _ in start_edges:
                v_count = max(v_count, u, v)
            for _ in range(v_count + 1):
                self.add_vertex()
            for u, v, weight in start_edges:
                self.add_edge(u, v, weight)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        if self.v_count == 0:
            return 'EMPTY GRAPH\n'
        out = '   |'
        out += ' '.join(['{:2}'.format(i) for i in range(self.v_count)]) + '\n'
        out += '-' * (self.v_count * 3 + 3) + '\n'
        for i in range(self.v_count):
            row = self.adj_matrix[i]
            out += '{:2} |'.format(i)
            out += ' '.join(['{:2}'.format(w) for w in row]) + '\n'
        out = f"GRAPH ({self.v_count} vertices):\n{out}"
        return out

    # ------------------------------------------------------------------ #

    def add_vertex(self) -> int:
        """
        TODO: Write this implementation
        """
        self.v_count += 1

        size_matrix = self.v_count

        if size_matrix == 0:
            self.adj_matrix.append([0])
        else:
            for row in self.adj_matrix:
                row.append(0)
            self.adj_matrix.append([0] * size_matrix)

        return self.v_count

    def add_edge(self, src: int, dst: int, weight=1) -> None:
        """
        TODO: Write this implementation
        """
        size_matrix = self.v_count

        if src >= size_matrix or dst >= size_matrix:
            return
        if src == dst:
            return
        if weight <= 0:
            return

        self.adj_matrix[src][dst] = weight

    def remove_edge(self, src: int, dst: int) -> None:
        """
        TODO: Write this implementation
        """
        size_matrix = self.v_count

        if src >= size_matrix or src < 0:
            return
        if dst >= size_matrix or dst < 0:
            return
        if src == dst:
            return

        self.adj_matrix[src][dst] = 0

    def get_vertices(self) -> []:
        """
        TODO: Write this implementation
        """
        return list(range(len(self.adj_matrix)))

    def get_edges(self) -> []:
        """
        TODO: Write this implementation
        """
        edges = []
        size_matrix = self.v_count

        for i in range(size_matrix):
            for j in range(size_matrix):
                if i != j:
                    weight = self.adj_matrix[i][j]
                    if weight != 0:
                        edges.append((i, j, weight))

        return edges


    def is_valid_path(self, path: []) -> bool:
        """
        TODO: Write this implementation
        """
        length = len(path)

        if length == 0 or length == 1:
            return True

        for i in range(length - 1):
            src = path[i]
            dst = path[i+1]
            if self.adj_matrix[src][dst] == 0:
                return False

        return True


    def dfs(self, v_start, v_end=None) -> []:
        """
        TODO: Write this implementation
        """
        traversal_path = []
        size_matrix = self.v_count

        if v_start < 0 or v_start > size_matrix - 1:
            return traversal_path

        visited_verticies = set()
        stack = deque()

        i = v_start
        stack.append(i)

        while stack:
            i = stack.pop()

            if i not in visited_verticies:
                visited_verticies.add(i)
                traversal_path.append(i)

                if i == v_end:
                    return traversal_path

                for j in range(size_matrix - 1, -1, -1):
                    weight = self.adj_matrix[i][j]
                    if weight != 0 and i != j:
                        stack.append(j)

        return traversal_path


    def bfs(self, v_start, v_end=None) -> []:
        """
        TODO: Write this implementation
        """
        traversal_path = []
        size_matrix = self.v_count

        if v_start < 0 or v_start > size_matrix - 1:
            return traversal_path

        visited_verticies = set()
        queue = deque()

        i = v_start
        queue.append(i)

        while queue:
            i = queue.popleft()

            if i not in visited_verticies:
                visited_verticies.add(i)
                traversal_path.append(i)

                if i == v_end:
                    return traversal_path

                for j in range(size_matrix):
                    weight = self.adj_matrix[i][j]
                    if weight != 0 and i != j:
                        queue.append(j)

        return traversal_path

    def _component_has_cycle(self, src):
        """
        Returns Boolean if component of graph starting at src vertex has a cycle
        """
        # vertex_flags = {i:-1 for i in range(self.v_count)}

        stack = deque()
        for vertex in reversed(self.dfs(src)):
            stack.append(vertex)
            # vertex_flags[vertex] = 0

        get_successors = lambda vertex: [ i for i in range(self.v_count) if self.adj_matrix[vertex][i] != 0 ]

        visited_verticies = set()

        while stack:
            vertex = stack.pop()

            if vertex not in visited_verticies:
                visited_verticies.add(vertex)

            successors = get_successors(vertex)
            if len(successors) == 0:
                # visited_verticies.remove(vertex)
                return False

            for successor in successors:
                if successor in visited_verticies:
                    return True

        return False



    def has_cycle(self):
        """
        TODO: Write this implementation
        """
        # The general approach used to detect a cycle was followed from this
        # video: https://youtu.be/AK7BuT5MgU0

        if self.v_count < 3:
            return False

        for i in range(self.v_count):
            if self._component_has_cycle(i):
                return True

        return False


    def dijkstra(self, src: int) -> []:
        """
        TODO: Write this implementation
        """

        visited_verticies = dict()

        priority_q = []
        heapq.heappush(priority_q, (0, src))

        while priority_q:
            distance, vertex = heapq.heappop(priority_q)

            if visited_verticies.get(vertex) is None:
                visited_verticies[vertex] = distance

                for i in range(self.v_count):
                    successor_weight = self.adj_matrix[vertex][i]
                    if successor_weight != 0:
                        heapq.heappush(priority_q, (distance + successor_weight, i))

            elif distance < visited_verticies.get(vertex):
                visited_verticies[vertex] = distance

        shortest_path = []
        for i in range(self.v_count):
            distance = visited_verticies.get(i)
            if distance is None:
                shortest_path.append(float('inf'))
            else:
                shortest_path.append(distance)

        return shortest_path


if __name__ == '__main__':

    # # method add_vertex() / add_edge example 1 {{{
    # print("\nPDF - method add_vertex() / add_edge example 1")
    # print("----------------------------------------------")
    # g = DirectedGraph()
    # print(g)
    # for _ in range(5):
    #     g.add_vertex()
    # print(g)

    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # for src, dst, weight in edges:
    #     g.add_edge(src, dst, weight)
    # print(g)


    # # }}}
    # # method get_edges() example 1 {{{
    # print("\nPDF - method get_edges() example 1")
    # print("----------------------------------")
    # g = DirectedGraph()
    # print(g.get_edges(), g.get_vertices(), sep='\n')
    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # g = DirectedGraph(edges)
    # print(g.get_edges(), g.get_vertices(), sep='\n')


    # # }}}
    # # method is_valid_path() example 1 {{{
    # print("\nPDF - method is_valid_path() example 1")
    # print("--------------------------------------")
    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # g = DirectedGraph(edges)
    # test_cases = [[0, 1, 4, 3], [1, 3, 2, 1], [0, 4], [4, 0], [], [2]]
    # for path in test_cases:
    #     print(path, g.is_valid_path(path))


    # # }}}
    # method dfs() and bfs() example 1 {{{
    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for start in range(5):
        print(f'{start} DFS:{g.dfs(start)} BFS:{g.bfs(start)}')


    # }}}
    # method has_cycle() example 1 {{{
    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)

    edges_to_remove = [(3, 1), (4, 0), (3, 2)]
    for src, dst in edges_to_remove:
        g.remove_edge(src, dst)
        print(
            g.get_edges(),
            g.has_cycle(),
            sep='\n'
            )

    edges_to_add = [(4, 3), (2, 3), (1, 3), (4, 0)]
    for src, dst in edges_to_add:
        g.add_edge(src, dst)
        print(
            g.get_edges(),
            g.has_cycle(),
            sep='\n'
            )
    print('\n', g)


    print("\nPDF - dijkstra() example 1")
    print("--------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
    g.remove_edge(4, 3)
    print('\n', g)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')

    # }}}
