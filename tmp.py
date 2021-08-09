from d_graph import DirectedGraph

edges = [
    (0, 1, 10),
    (1, 3, 1),
    (1, 4, 15),
    (2, 1, 23),
    (2, 3, 1),
    (4, 3, 1),
    ]

graph = DirectedGraph(edges)

print(graph.dfs(0))

print(graph.has_cycle())

# adj = [0, 2, 0, 0, 4, 5]

# # [1, 4, 5]

# neighbors = []
# for i in range(len(adj)):
#     if adj[i] != 0:
#         neighbors.append(i)

# print(neighbors)



# neighbors = [i for i in range(len(adj)) if adj[i] != 0]

# print(neighbors)
