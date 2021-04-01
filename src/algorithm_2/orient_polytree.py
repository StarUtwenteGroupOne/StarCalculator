from src.algorithm_2.common import get_top_event, orient_edge_direction, is_leaf
from src.lib.graph import Graph, Vertex, Edge


def orient_polytree(undirected_fault_tree: Graph):
    """
    Creates a directed fault tree out of an undirected fault tree.
    Doesn't use a deepcopy yet, which may give some problems

    :param undirected_fault_tree: The fault tree to orient
    :return: A directed Fault Tree of a bow tie.
    """
    result = undirected_fault_tree.deepcopy()

    top_event = get_top_event(result)

    orient_top_event_for_fault_tree(top_event)

    result._directed = True

    return result


def orient_top_event_for_fault_tree(parent: Vertex):
    """
    Orients the edges of the TE of the fault tree towards the TE (which is the parent, in this case).
    After that, it checks whether the children of the TE are leaves.
    If it is not a leaf, the algorithm will loop over the edges of the articulation point.

    :param parent: Top Event / parent event of the articulation node
    """
    for e in parent.incidence:
        orient_edge_direction(e, e.other_end(parent))
        if not is_leaf(e.other_end(parent)):
            loop_over_articulation_point_edges_to_check_independence(e)


def loop_over_articulation_point_edges_to_check_independence(parent_edge: Edge):
    """
    Loops over the edges of an articulation point to check the independence of
    the articulation point's parent and child.

    :param parent_edge: The parent edge which doesn't need to be checked. This
    is also used to get the articulation point and its parent.
    """
    parent = parent_edge.head
    articulation_point = parent_edge.tail
    for e in articulation_point.incidence:
        child = e.other_end(articulation_point)
        if not child == parent:
            determine_independence_of_parent_and_child(parent, articulation_point, child, e)


def determine_independence_of_parent_and_child(parent: Vertex, articulation_point: Vertex, child: Vertex, edge: Edge):
    """
    Determines the independence of the parent and child of an articulation point based on expert judgement.
    If the child is independent, then the child will be treated as a second parent of the articulation point.
    If the child is not independent, then the child will be treated as another articulation point if it isn't a leaf.

    :param parent: The parent vertex of the articulation point.
    :param articulation_point: The articulation point.
    :param child: The child of the articulation point (which could become a second parent if it is independent from
    the parent parameter.
    :param edge: The edge which will be oriented (between the articulation point and the child.
    """
    yes_or_no = input("Are " + parent.label + " and " + child.label + " independent? (y/n)")
    while not (yes_or_no == "y" or yes_or_no == "n"):
        yes_or_no = input("Wrong input! Are " + parent.label + " and " + child.label + " independent? (y/n)")
    if yes_or_no == "y":
        orient_edge_direction(edge, articulation_point)
        if not is_leaf(child):
            treat_as_parent(child, articulation_point)
    elif yes_or_no == "n":
        orient_edge_direction(edge, child)
        if not is_leaf(child):
            loop_over_articulation_point_edges_to_check_independence(edge)


def treat_as_parent(child: Vertex, articulation_point: Vertex):
    """
    Treats the child as second parent if it is independent from the parent of an articulation point.
    For that, it orients all other edges towards itself and treats its children as articulation points if they
    are not a leaf.

    :param child: The child which now becomes a second parent of the articulation point.
    :param articulation_point: The originating articulation point which now gets a second parent.
    """
    for child_edge in child.incidence:
        if not child_edge.other_end(child) == articulation_point:
            orient_edge_direction(child_edge, child_edge.other_end(child))
            if not is_leaf(child_edge.other_end(child)):
                loop_over_articulation_point_edges_to_check_independence(child_edge)