from src.algorithm_2.common import get_top_event, orient_edge_direction
from lib.ut_graphs.graph import Graph


def orient_tree(undirected_event_tree: Graph):
    """
    Creates a directed event tree out of an undirected event tree.
    Doesn't use a deepcopy yet, which may give some problems

    :return: A directed Event Tree of a bow tie.
    """
    result = undirected_event_tree.deepcopy()

    top_event = get_top_event(result)

    # Keep track of vertexes which have been checked
    vertexes_to_check_edge_orientation = [top_event]
    checked_vertexes = list()

    # While there are still vertexes to check
    while len(checked_vertexes) < len(result.vertices):

        # Pop one of the vertexes to check
        v = vertexes_to_check_edge_orientation.pop()

        # Check every edge incident to v and add it to the vertexes which need to be checked.
        for e in v.incidence:
            if e.other_end(v) not in checked_vertexes:
                orient_edge_direction(e, v)
                vertexes_to_check_edge_orientation.append(e.other_end(v))

        # v has been checked
        # vertexes_to_check_edge_orientation.remove(v)
        # It's already gone, you popped it!!
        checked_vertexes.append(v)

    # The graph is now directed
    result._directed = True

    return result