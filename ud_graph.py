# Course: CS261 Data Structures
# Author: Ethan Rietz
# Assignment: 6
# Description: Contains a class which represents an undirected graph

from collections import deque

class UndirectedGraph:
    """
    Class to implement undirected graph
    - duplicate edges not allowed
    - loops not allowed
    - no edge weights
    - vertex names are strings
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency list
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.adj_list = dict()

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            for u, v in start_edges:
                self.add_edge(u, v)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        out = [f'{v}: {self.adj_list[v]}' for v in self.adj_list]
        out = '\n  '.join(out)
        if len(out) < 70:
            out = out.replace('\n  ', ', ')
            return f'GRAPH: {{{out}}}'
        return f'GRAPH: {{\n  {out}}}'

    # ------------------------------------------------------------------ #

    def add_vertex(self, v: str) -> None:
        """
        Add new vertex to the graph
        """
        if v in self.adj_list:
            return
        else:
            self.adj_list[v] = []


    def add_edge(self, u: str, v: str) -> None:
        """
        Add edge to the graph
        """
        # graph cannot have a loop
        if u == v:
            return

        u_edges, v_edges = self.adj_list.get(u), self.adj_list.get(v)

        # Edge already exists
        if u_edges is not None and v_edges is not None and u in v_edges:
            return

        # u does not already exist
        if u_edges is None:
            self.add_vertex(u)

        if v_edges is None:
            self.add_vertex(v)

        self.adj_list[u].append(v)
        self.adj_list[v].append(u)


    def remove_edge(self, v: str, u: str) -> None:
        """
        Remove edge from the graph
        """
        u_edges, v_edges = self.adj_list.get(u), self.adj_list.get(v)

        if u_edges is None or v_edges is None:
            return

        if u not in v_edges:
            return

        self.adj_list[u].remove(v)
        self.adj_list[v].remove(u)


    def remove_vertex(self, v: str) -> None:
        """
        Remove vertex and all connected edges
        """
        edges = self.adj_list.get(v)

        if edges is None:
            return
        else:
            for vertex in edges:
                edge = self.adj_list.get(vertex)
                edge.remove(v)

            del self.adj_list[v]


    def get_vertices(self) -> []:
        """
        Return list of vertices in the graph (any order)
        """
        return list(self.adj_list.keys())


    def get_edges(self) -> []:
        """
        Return list of edges in the graph (any order)
        """
        all_edges = set()   # is item in set is O(1) (on average) operation

        for u, edge_list in self.adj_list.items():
            for v in edge_list:
                if (v, u) not in all_edges:
                    all_edges.add((u, v))

        return list(all_edges)


    def is_valid_path(self, path: []) -> bool:
        """
        Return true if provided path is valid, False otherwise
        """
        length_path = len(path)

        if length_path == 1 and path[0] not in self.adj_list.keys():
            return False

        for i in range(length_path - 1):
            vertex = path[i]
            edges = self.adj_list[vertex]
            next_vertex = path[i+1]
            if next_vertex not in edges:
                return False
        return True


    def dfs(self, v_start, v_end=None) -> []:
        """
        Return list of vertices visited during DFS search
        Vertices are picked in alphabetical order
        """
        if v_start not in self.adj_list.keys():
            return []

        traversal_path = []
        visited_verticies = set()
        stack = deque()

        stack.append(v_start)
        while stack:
            vertex = stack.pop()

            # Add surrounding verticies to stack for processing
            if vertex not in visited_verticies:
                visited_verticies.add(vertex)

                # Process the vertex
                traversal_path.append(vertex)
                if vertex == v_end:
                    return traversal_path

                for successor in sorted(self.adj_list[vertex], reverse=True):
                    stack.append(successor)

        return traversal_path


    def bfs(self, v_start, v_end=None) -> []:
        """
        Return list of vertices visited during BFS search
        Vertices are picked in alphabetical order
        """
        if v_start not in self.adj_list.keys():
            return []

        visited_verticies = set()
        traversal_path = []
        queue = deque()

        queue.append(v_start)
        while queue:
            vertex = queue.popleft()

            # Add surrounding verticies to stack for processing
            if vertex not in visited_verticies:
                visited_verticies.add(vertex)

                # Process the vertex
                traversal_path.append(vertex)
                if vertex == v_end:
                    return traversal_path

                for successor in sorted(self.adj_list[vertex]):
                    queue.append(successor)

        return traversal_path


    def count_connected_components(self):
        """
        Return number of connected components in the graph
        """
        connected_components = 0
        visited_verticies = set()

        for vertex in self.adj_list.keys():
            if vertex not in visited_verticies:
                connected_components += 1
                verticies_in_compenent = self.dfs(vertex)
                for other_vertex in verticies_in_compenent:
                    visited_verticies.add(other_vertex)

        return connected_components


    def has_cycle(self):
        """
        Return True if graph contains a cycle, False otherwise
        """
        verticies = list(self.adj_list.keys())
        if len(verticies) < 3:
            return False

        # Since the graph can be disconnected, we have to perform a BFS
        # starting from each node in the graph to check for cycles.
        for vertex in verticies:
            queue = deque()

            vertex_flags = dict()
            for i in verticies:
                vertex_flags[i] = -1

            queue.append(vertex)

            while queue:
                vertex = queue.popleft()
                vertex_flags[vertex] = 1

                for successor in self.adj_list[vertex]:
                    successor_flag = vertex_flags[successor]

                    if successor_flag == 0:
                        return True
                    if successor_flag == -1:
                        queue.append(successor)
                        vertex_flags[successor] = 0

        return False

if __name__ == '__main__':

    # method add_vertex() / add_edge example 1 {{{
    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = UndirectedGraph()
    print(g)

    for v in 'ABCDE':
        g.add_vertex(v)
    print(g)

    g.add_vertex('A')
    print(g)

    for u, v in ['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE', ('B', 'C')]:
        g.add_edge(u, v)
    print(g)


    # }}}

    # method remove_edge() / remove_vertex example 1 {{{
    print("\nPDF - method remove_edge() / remove_vertex example 1")
    print("----------------------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    g.remove_vertex('DOES NOT EXIST')
    g.remove_edge('A', 'B')
    g.remove_edge('X', 'B')
    print(g)
    g.remove_vertex('D')
    print(g)

    # }}}

    # method get_vertices() / get_edges() example 1 {{{
    print("\nPDF - method get_vertices() / get_edges() example 1")
    print("---------------------------------------------------")
    g = UndirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE'])
    print(g.get_edges(), g.get_vertices(), sep='\n')


    # }}}

    # method is_valid_path() example 1 {{{
    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    test_cases = ['ABC', 'ADE', 'ECABDCBE', 'ACDECB', '', 'D', 'Z']
    for path in test_cases:
        print(list(path), g.is_valid_path(list(path)))


    # }}}

    # method dfs() and bfs() example 1 {{{
    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = 'ABCDEGH'
    for case in test_cases:
        print(f'{case} DFS:{g.dfs(case)} BFS:{g.bfs(case)}')
    print('-----')
    for i in range(1, len(test_cases)):
        v1, v2 = test_cases[i], test_cases[-1 - i]
        print(f'{v1}-{v2} DFS:{g.dfs(v1, v2)} BFS:{g.bfs(v1, v2)}')

    # }}}

    # method count_connected_components() example 1 {{{
    print("\nPDF - method count_connected_components() example 1")
    print("---------------------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print(g.count_connected_components(), end=' ')
    print()

    # }}}

    # method has_cycle() example 1 {{{
    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG',
        'add FG', 'remove GE')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print('{:<10}'.format(case), g.has_cycle())

    # }}}
