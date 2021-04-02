from lib.ut_graphs.graph import Graph, Vertex, Edge
from src.algorithm_1.get_mutual_information import get_mutual_information

VertexList = [Vertex]


def create_undirected_tree(training_set):
    events_size = training_set.get_events_size()
    weights = get_mutual_information(training_set)
    graph = Graph(directed=False)
    vertices: VertexList = [Vertex(graph, label=name) for name
                            in training_set.event_names]

    for i in range(1, events_size):
        highest_weight = -1
        highest_j = -1
        for j in range(0, i):
            if weights[i][j] > highest_weight:
                highest_j = j
                highest_weight = weights[i][j]
        # print(highest_weight)

        graph += Edge(vertices[i], vertices[highest_j])
    return graph
