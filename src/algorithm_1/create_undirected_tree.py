from lib.ut_graphs.graph import Graph, Vertex, Edge
from src.algorithm_1.get_mutual_information import get_mutual_information

VertexList = [Vertex]


# noinspection PyTypeChecker
def create_undirected_tree(training_set):
    events_size = training_set.get_events_size()
    weights = get_mutual_information(training_set)


    # now run the maximal_weight_spanning_tree algorithm
    graph = Graph(False)
    available_events = [True] * events_size
    available_events[0] = False
    vertices: VertexList = [Vertex(graph, label=training_set.event_names[0])] + [None for _ in range(events_size - 1)]
    graph.add_vertex(vertices[0])
    for event_names_index in range(1, events_size):
        highest_i = -1
        highest_j = -1
        for i in range(0, events_size):
            for j in range(0, events_size):
                if not available_events[i] and \
                        available_events[j] and \
                        (highest_i == -1 or weights[highest_i][highest_j] <= weights[i][j]):
                    highest_i = i
                    highest_j = j
        available_events[highest_j] = False
        vertices[highest_j] = Vertex(graph, training_set.event_names[event_names_index])
        edge = Edge(vertices[highest_i], vertices[highest_j])
        graph.add_vertex(vertices[highest_j])
        graph.add_edge(edge)
    return graph