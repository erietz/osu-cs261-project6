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
        Adds a vertex to the directed graph and returns the number of vertices
        in the graph after the addition. A vertex without an edge has a weight
        of zero.
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
        Adds an edge of weight <weight> to the directed graph that points from
        <src> to <dst>.
        """
        size_matrix = self.v_count

        if src >= size_matrix or src < 0:
            return
        if dst >= size_matrix or dst < 0:
            return
        if src == dst:
            return
        if weight <= 0:
            return

        self.adj_matrix[src][dst] = weight

    def remove_edge(self, src: int, dst: int) -> None:
        """
        Removes the edge between <src> and <dst> vertices.
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
        Returns a list of all the vertices
        """
        return list(range(self.v_count))

    def get_edges(self) -> []:
        """
        Returns a list of tuples representing all of the edges. The tuple is of
        the form (src, dst, weight).
        """
        edges = []
        size_matrix = self.v_count

        for i in range(size_matrix):
            for j in range(size_matrix):
                weight = self.adj_matrix[i][j]
                if weight != 0:
                    edges.append((i, j, weight))
        return edges


    def is_valid_path(self, path: []) -> bool:
        """
        Returns Boolean if the graph can be traversed following <path>
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
        Performs a depth first search starting from vertex <v_start> and ending
        at vertex <v_end> (or the entire graph if not included) and returns a
        list of the vertices visited during the traversal. If a node has
        multiple adjacent nodes, the DFS is performed by taking the nodes in
        ascending order (e.g. if current node = 4, next nodes traversed would 
        be 1, 2, 5 instead of 5, 2, 1).
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

                # Put adjacent nodes on stack in reverse order so they are
                # popped off in correct order
                for j in range(size_matrix - 1, -1, -1):
                    weight = self.adj_matrix[i][j]
                    if weight != 0 and i != j:
                        stack.append(j)

        return traversal_path


    def bfs(self, v_start, v_end=None) -> []:
        """
        Performs a breadth first search starting from vertex <v_start> and ending
        at vertex <v_end> (or the entire graph if not included) and returns a
        list of the vertices visited during the traversal. If a node has
        multiple adjacent nodes, the BFS is performed by taking the nodes in
        ascending order (e.g. if current node = 4, next nodes traversed would 
        be 1, 2, 5 instead of 5, 2, 1).
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

    def _component_has_cycle(self, i, visited, stack):
        """
        Returns Boolean if component of graph starting at vertex <i> has a
        cycle
        """
        visited[i] = True
        stack[i] = True

        # All adjacent vertices to vertex i
        neighbors = [j for j in range(self.v_count) if self.adj_matrix[i][j] != 0]

        # Do a DFS starting from current node. Recursive calls end when node
        # has no neighbors or cycle has been found.
        for neighbor in neighbors:
            # Continue to traverse the graph (in recursive calls) making the
            # current vertex, i one of its neighbors.
            if not visited[neighbor]:
                if self._component_has_cycle(neighbor, visited, stack):
                    return True
            # Neighbor has been visited and is on the stack. This indicates a
            # cycle
            elif stack[neighbor]:
                return True

        # remove node from the stack when backtracking from DFS traversal
        stack[i] = False
        return False

    def has_cycle(self):
        """
        Returns Boolean if the graph has a cycle
        """
        # The idea of this implementation comes from this video
        # https://www.youtube.com/watch?v=0dJmTuMrUZM

        if self.v_count < 3:
            return False

        visited = [False] * self.v_count    # array of nodes visited during DFS
        stack = [False] * self.v_count      # array of nodes being processed
        # Do a DFS starting from each node in the graph since the graph can be
        # disconnected.
        for i in range(self.v_count):
            # Visited array is modified in recursive call for nodes which are
            # visited.
            if visited[i] == False:
                if self._component_has_cycle(i, visited, stack):
                    return True
        return False


    def dijkstra(self, src: int) -> []:
        """
        Returns a list with one value per each vertex in the graph, where the
        value at index 0 is the length of the shortest path from vertex SRC to
        vertex 0, the value at index 1 is the length of the shortest path from
        vertex SRC to vertex 1 etc.
        """

        # key = vertex index
        # value = minimum cumulative distance from src to vertex
        visited_verticies = dict()

        # heap data structure held in a python list. Elements in the list are
        # tuples: (weight, vertex). The weight in this case is the priority
        priority_q = []
        # src vertex has priority 0
        heapq.heappush(priority_q, (0, src))

        # Classic dijkstra algorithm
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

        # Build the list of distance to return. None values in the dictionary
        # must be replaced by infinity since there is not path to them.
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
    # # method dfs() and bfs() example 1 {{{
    # print("\nPDF - method dfs() and bfs() example 1")
    # print("--------------------------------------")
    # edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
    #          (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    # g = DirectedGraph(edges)
    # for start in range(5):
    #     print(f'{start} DFS:{g.dfs(start)} BFS:{g.bfs(start)}')


    # # }}}
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
