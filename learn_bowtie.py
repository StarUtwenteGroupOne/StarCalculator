import os

import numpy as np

from config import TOP_EVENT_LABEL, OUTPUT_DIR
from lib.graph import Graph, Vertex, Edge
from write_graph import write_graph_to_dotfile


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


def orient_edge_direction(edge: Edge, origin_vertex: Vertex):
    """
    Swaps the head and tail of an edge if the head is the origin vertex.

    :return: An edge where the direction is swapped around.
    """
    if edge.head == origin_vertex:
        edge._head = edge.other_end(origin_vertex)
        edge._tail = origin_vertex


def create_bowtie(event_tree, fault_tree):
    # And now we need to union the graph
    bowtie = Graph(directed=True)

    bowtie._v += fault_tree.vertices + event_tree.vertices
    bowtie._e += fault_tree.edges + event_tree.edges

    for v in bowtie._v:
        v._graph = bowtie

    top_events = []
    # Find the two top events:
    for vtx in bowtie.vertices:
        if vtx.label == TOP_EVENT_LABEL:
            top_events.append(vtx)

    assert len(top_events) == 2

    for edge in top_events[1].incidence:
        bowtie.add_edge(Edge(tail=top_events[0], head=edge.head, weight=edge.weight))

    # Delete the now duplicate top event
    bowtie.del_vertex(top_events[1])

    write_graph_to_dotfile(bowtie, 'bowtie.dot')
    return bowtie


def create_quantitative_event_tree(directed_event_tree, training_set_event_tree):
    probability_of_happening = []
    G = directed_event_tree
    tr = training_set_event_tree
    for v in G.vertices:
        probability_of_happening_i = {}
        list1 = tr.get_observations_by_event_name(v.label)
        for e in v.incidence:
            list2 = tr.get_observations_by_event_name(e.head.label)
            number_instances_i_j = 0
            number_instances_vertex = 0
            for i in range(0, len(list1)):
                if list1[i] == list2[i] and list2[i]:
                    number_instances_i_j += 1
                if list1[i]:
                    number_instances_vertex += 1
            if e.head.label != v.label:
                if number_instances_vertex != 0:
                    probability_of_happening_i[e.head] = number_instances_i_j / number_instances_vertex
                e._weight = number_instances_i_j / number_instances_vertex
        v.probability.update(probability_of_happening_i)
        probability_of_happening.append(probability_of_happening_i)

    write_graph_to_dotfile(G,'create_quantitative_event_tree.dot')
    return G, probability_of_happening


def create_quantitative_fault_tree(directed_fault_tree, training_set_fault_tree):
    cpt = []
    G = directed_fault_tree
    tr = training_set_fault_tree
    alpha = 1
    for v in G.vertices:
        cpt_i = {}
        vertices_set = [tr.get_observations_by_event_name(v.label)]
        helping_dict = {}
        edges = []
        for e in G.edges:
            if e.head == v:
                vertices_set.append(tr.get_observations_by_event_name(e.tail.label))
                edges.append(e)
        if len(vertices_set) > 1:
            vertices_set = np.array(vertices_set).transpose()
            total = vertices_set.shape[0] + (2 ** (vertices_set.shape[1] - 1)) * alpha

            for p in vertices_set:
                p_as_np = np.array(p[1:])
                if not (str(p_as_np) in helping_dict.keys()):
                    helping_dict[str(p_as_np)] = 1
                else:
                    helping_dict[str(p_as_np)] = helping_dict[str(p_as_np)] + 1

            for k in helping_dict.keys():
                cpt_i[k] = (helping_dict[k] + alpha) / total
            v.probability.update(cpt_i)

            cpt.append(cpt_i)
        else:
            vertices_set = np.array(vertices_set).transpose()
            size = len(vertices_set)
            for obs in vertices_set:
                if str(obs) in v.probability:
                    v.probability[str(obs)] += 1
                else:
                    v.probability[str(obs)] = 1
            # obs is either 0 or 1, so add alpha and divide by number of observations
            for k in v.probability.keys():
                v.probability[k] = (v.probability[k] + alpha) / (size + 2 * alpha)

    write_graph_to_dotfile(G, 'create_quantitative_fault_tree.dot')
    return G, cpt