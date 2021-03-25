# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import math
import numpy as np

from graph import Graph, Vertex, Edge
from trainingset import TrainingSet

# noinspection PyTypeChecker
VertexList = list(Vertex)


def start():
    test_bowtie = create_test_bowtie()
    training_set_event_tree = create_trainingset_event_tree(test_bowtie)
    training_set_fault_tree = create_trainingset_fault_tree(test_bowtie)
    top_event = get_top_event(test_bowtie)
    learning_parameters = get_learning_parameters()
    bowtie = create_quantitative_bowtie(training_set_event_tree,
                                        training_set_fault_tree,
                                        top_event,
                                        learning_parameters)
    print_quantitative_bowtie(bowtie)


def create_quantitative_bowtie(training_set_event_tree, training_set_fault_tree, top_event, learning_parameters):
    undirected_event_tree = create_undirected_tree(training_set_event_tree)
    undirected_fault_tree = create_undirected_tree(training_set_fault_tree)
    directed_event_tree = create_directed_event_tree(undirected_event_tree, top_event)
    directed_fault_tree = create_directed_fault_tree(undirected_fault_tree, top_event)
    quantitative_event_tree, probability_of_event_tree = create_quantitative_event_tree(directed_event_tree,
                                                                                        training_set_event_tree)
    quantitative_fault_tree, probability_of_fault_tree = create_quantitative_fault_tree(directed_fault_tree,
                                                                                        training_set_fault_tree)
    quantitative_bowtie = create_quantitative_bowtie_from_trees(quantitative_event_tree, quantitative_fault_tree)
    return quantitative_bowtie


def create_test_bowtie():
    print("createTestBowtie")
    # A hardcoded fault tree
    return 1


def create_trainingset_fault_tree(test_bowtie):
    print("createTrainingSetFaultTree")
    return TrainingSet([[False,False,False],[False,True,False],[True,False,False],[True,True,True]],{"a":1,"b":2,"c":3})


def create_trainingset_event_tree(test_bowtie):
    print("createTrainingSetEventTree")
    return TrainingSet([1])


def get_top_event(test_bowtie):
    print("getTopEvent")
    return 1


def get_learning_parameters():
    print("getLearningParameters")
    return 1


# noinspection PyTypeChecker
def create_undirected_tree(training_set):
    events_size = training_set.get_events_size()
    weights = [[0] * events_size for _ in range(events_size)]
    for i in range(0, events_size - 1):
        for j in range(1, events_size):
            weights[i][j] = compute_mutual_information(training_set, i, j)

    graph = Graph(False)
    available_events = [True] * events_size
    available_events[0] = False
    vertices: VertexList = [Vertex(graph)] + [None for _ in range(events_size - 1)]
    graph.add_vertex(vertices[0])
    for _ in range(1, events_size):
        highest_i = -1
        highest_j = -1
        for i in range(0, events_size):
            for j in range(1, events_size):
                if not available_events[i] and \
                        available_events[j] and \
                        (highest_i == -1 or weights[highest_i][highest_j] <= weights[i][j]):
                    highest_i = i
                    highest_j = j
        available_events[highest_j] = False
        vertices[highest_j] = Vertex(graph)
        edge = Edge(vertices[highest_i], vertices[highest_j])
        graph.add_vertex(vertices[highest_j])
        graph.add_edge(edge)
    return graph


def compute_mutual_information(training_set, event1, event2):
    weight = 0
    for event1_state in True, False:
        for event2_state in True, False:
            probability_event1 = training_set.compute_single_probability(event1, event1_state)
            probability_event2 = training_set.compute_single_probability(event2, event2_state)
            probability_event1_and_event2 = training_set.compute_combined_probability(event1, event1_state, event2,
                                                                                      event2_state)
            probability = probability_event1_and_event2 / (probability_event1 * probability_event2)
            weight += probability_event1_and_event2 * math.log(probability, 10)
    return weight


def create_directed_event_tree(undirected_event_tree: Graph, top_event: Vertex):
    """
    Creates a directed event tree out of an undirected event tree.
    Doesn't use a deepcopy yet, which may give some problems

    :return: A directed Event Tree of a bow tie.
    """
    result = undirected_event_tree

    # Keep track of vertexes which have been checked
    vertexes_to_check_edge_orientation = list()
    vertexes_to_check_edge_orientation += top_event
    checked_vertexes = list()

    # While there are still vertexes to check
    while len(checked_vertexes) < len(result.vertices):

        # Pop one of the vertexes to check
        v = vertexes_to_check_edge_orientation.pop()

        # Check every edge incident to v and add it to the vertexes which need to be checked.
        for e in v.incidence:
            if e.other_end(v) not in checked_vertexes:
                orient_edge_direction(e, v)
                vertexes_to_check_edge_orientation += e.other_end(v)

        # v has been checked
        vertexes_to_check_edge_orientation.remove(v)
        checked_vertexes += v

    # The graph is now directed
    result._directed = True

    return result


def create_directed_fault_tree(undirected_fault_tree: Graph, top_event: Vertex):
    """
    Creates a directed fault tree out of an undirected fault tree.
    Doesn't use a deepcopy yet, which may give some problems

    :param undirected_fault_tree: The fault tree to orient
    :param top_event: The TE of the fault tree
    :return: A directed Fault Tree of a bow tie.
    """
    result = undirected_fault_tree

    orient_top_event_for_fault_tree(top_event)

    result._directed = True

    return result


def is_leaf(vertex: Vertex):
    """
    Returns true if the vertex has more than one edge

    :param vertex: Vertex to check edges
    :return: True if vertex has more than one edge
    """
    return len(vertex.incidence) > 1


def orient_top_event_for_fault_tree(parent: Vertex):
    """
    Orients the edges of the TE of the fault tree towards the TE (which is the parent, in this case).
    After that, it checks whether the children of the TE are leafs.
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
    yes_or_no = input("Are " + parent.label + " and " + child.label + "independent? (y/n)")
    while not (yes_or_no == "y" or yes_or_no == "n"):
        yes_or_no = input("Wrong input! Are " + parent.label + " and " + child.label + "independent? (y/n)")
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


def create_quantitative_event_tree(directed_event_tree, training_set_event_tree):
    probability_of_happening = []
    G = directed_event_tree
    tr = training_set_event_tree
    for v in G.vertices:
        vertex = v
        probability_of_happening_i = {}
        list1 = tr.get_observations_by_event_name(v.label)
        for e in G.edges:
            if e.tail == vertex:
                list2 = tr.get_observations_by_event_name(e.head.label)
                number_instances_i_j = 0
                number_instances_vertex = 0
                for i in range(0, len(list1)):
                    if list1[i] == list2[i] and list2[i]:
                        number_instances_i_j += 1
                    if list1[i]:
                        number_instances_vertex += 1
                if number_instances_vertex != 0:
                    probability_of_happening_i[e.head] = number_instances_i_j / number_instances_vertex
            e._weight = number_instances_i_j / number_instances_vertex
        v.probability = probability_of_happening_i
        probability_of_happening.append(probability_of_happening_i)
    print("createQuantitativeEventTree")
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
                if helping_dict[p] is None:
                    helping_dict[p] = 1
                else:
                    helping_dict[p] = helping_dict[p] + 1

            for k in helping_dict.keys():
                cpt_i[k] = (helping_dict[k] + alpha) / total
            v.probability = cpt_i
            cpt.append(cpt_i)
    print("createQuantitativeFaultTree")
    return G, cpt


def create_quantitative_bowtie_from_trees(quantitative_event_tree, quantitative_fault_tree):
    print("createBowTie")
    return 1


def print_quantitative_bowtie(quantitative_bowtie):
    print("printQuantitativeBowTie")
    return 1


if __name__ == '__main__':
    start()
