from src.config import TOP_EVENT_LABEL
from lib.ut_graphs.graph import Graph, Vertex, Edge


def get_top_event(test_bowtie: Graph):
    """
    Returns the top event, marked by the string TOP_EVENT_LABEL

    :param test_bowtie: graph that contains exactly one top event, labeled with TOP_EVENT_LABEL
    :return: The vertex labeled with TOP_EVENT_LABEL
    """
    top_events = []
    for vtx in test_bowtie.vertices:
        if vtx.label[:len(TOP_EVENT_LABEL)] == TOP_EVENT_LABEL:
            top_events.append(vtx)
    if len(top_events) != 1:
        raise AttributeError(f"Graph has no '{TOP_EVENT_LABEL}' node")
    return top_events[0]


def is_leaf(vertex: Vertex):
    """
    Returns true if the vertex has more than one edge

    :param vertex: Vertex to check edges
    :return: True if vertex has more than one edge
    """
    return len(vertex.incidence) == 1


def orient_edge_direction(edge: Edge, origin_vertex: Vertex):
    """
    Swaps the head and tail of an edge if the head is the origin vertex.

    :return: An edge where the direction is swapped around.
    """
    if edge.head == origin_vertex:
        edge._head = edge.other_end(origin_vertex)
        edge._tail = origin_vertex