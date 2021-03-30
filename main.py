# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import csv
import math
import random
from decimal import Decimal

import numpy as np

import graph_io
from graph import Graph, Vertex, Edge
from trainingset import TrainingSet

# noinspection PyTypeChecker
VertexList = [Vertex]

TOP_EVENT_LABEL = "TE"

TRAININGSET_DIR = './trainingset/'

def start():
    # test_bowtie = create_test_bowtie()

    # We will get this from file for now
    # training_set_event_tree = create_event_tree_trainingset(test_bowtie)
    # training_set_fault_tree = create_fault_tree_trainingset(test_bowtie)

    training_set_fault_tree = get_paper_fault_tree()
    training_set_event_tree = get_paper_event_tree()

    learning_parameters = get_learning_parameters()
    bowtie = create_quantitative_bowtie(training_set_event_tree,
                                        training_set_fault_tree,
                                        Vertex(graph=Graph(directed=False), label=TOP_EVENT_LABEL),
                                        learning_parameters)
    write_graph_to_dotfile(bowtie, filename="final_bowtie.dot")


def create_quantitative_bowtie(training_set_event_tree, training_set_fault_tree, top_event, learning_parameters):
    undirected_fault_tree = create_undirected_tree(training_set_fault_tree)
    undirected_event_tree = create_undirected_tree(training_set_event_tree)
    directed_fault_tree = create_directed_fault_tree(undirected_fault_tree)
    directed_event_tree = create_directed_event_tree(undirected_event_tree)
    quantitative_event_tree, probability_of_event_tree = create_quantitative_event_tree(directed_event_tree,
                                                                                        training_set_event_tree)
    quantitative_fault_tree, probability_of_fault_tree = create_quantitative_fault_tree(directed_fault_tree,
                                                                                        training_set_fault_tree)
    quantitative_bowtie = create_quantitative_bowtie_from_trees(quantitative_event_tree, quantitative_fault_tree)
    return quantitative_bowtie


def read_trainingset(filename):
    ft_trainingset = None
    with open(TRAININGSET_DIR + filename, 'r') as f:
        csv_file = csv.reader(f)
        ft_trainingset = TrainingSet()

        had_first_line = False
        for line in csv_file:
            # The first line are the column names
            if not had_first_line:
                ft_trainingset.event_names = [e for e in line]
                had_first_line = True
            else:
                ft_trainingset.observations.append([e == "T" for e in line])
    return ft_trainingset


def get_paper_fault_tree():

    filenames = ['faulttree.csv', 'FT_other_order.csv', 'FT_and_or.csv']

    return read_trainingset(filenames[2])


def get_paper_event_tree():

    filenames = ['eventtree.csv', 'ET_other_order.csv']

    return read_trainingset(filenames[1])


def create_test_bowtie(size=6):
    print("createTestBowtie")

    # The size needs to be one larger because we will merge the top events later.
    # size += 1
    #
    # fault_tree = create_fault_tree(ceil(size / 2))
    # event_tree = create_event_tree(size - ceil(size / 2))
    #
    # return create_quantitative_bowtie_from_trees(fault_tree, event_tree)
    ft = Graph(directed=True)
    ft_event_names = ["TE", "AND1", "AND2", "B1", "B2", "B3", "B4"]
    #ft_event_names = ["TE", "SI", "HGL", "PS", "TVF", "CTP", "EF", "DTA", "GO"]
    #ft_event_names = ["DTA", "GO", "TE", "EF", "CTP", "TVF", "HGL", "SI", "PS"]
    for label in ft_event_names:
        ft += Vertex(graph=ft, label=label)

    ft_trainingset = None

    filename = ['trainingset/faulttree.csv', 'trainingset/FT_other_order.csv','trainingset/FT_and_or.csv']

    with open(filename[2], 'r') as f:
        csv_file = csv.reader(f)
        ft_trainingset = TrainingSet(training_set={
        'event_names': ft_event_names,
        'observations': [[e == "T" for e in line] for line in csv_file]
        })

    # print(ft_trainingset.observations)

    et = Graph(directed=True)
    et_event_names = ["TE", "PF", "THE", "PPS", "TOE", "TDP", "DT", "LD", "TODP", "DE"]
    for label in et_event_names:
        et += Vertex(graph=et, label=label)

    et_trainingset = None

    with open('trainingset/eventtree.csv', 'r') as f:
        csv_file = csv.reader(f)

        et_trainingset = TrainingSet(training_set={
        'event_names': et_event_names,
        'observations': [[e == "T" for e in line] for line in csv_file]
        })

    # print(ft_trainingset.observations)

    et_from_ts = create_undirected_tree(et_trainingset)
    ft_from_ts = create_undirected_tree(ft_trainingset)

    write_graph_to_dotfile(et_from_ts, 'et_from_ts.dot')
    write_graph_to_dotfile(ft_from_ts, 'ft_from_ts.dot')

    dir_et_from_ts = create_directed_event_tree(et_from_ts)
    dir_ft_from_ts = create_directed_fault_tree(ft_from_ts)

    write_graph_to_dotfile(dir_ft_from_ts, 'dir_ft_from_ts.dot')
    write_graph_to_dotfile(dir_et_from_ts, 'dir_et_from_ts.dot')


def create_quantitative_bowtie_from_trees(event_tree, fault_tree):
    # And now we need to union the graph
    bowtie = Graph(directed=True)

    bowtie._v += fault_tree.vertices + event_tree.vertices
    bowtie._e += fault_tree.edges + event_tree.edges

    for v in bowtie._v:
        v._graph = bowtie

    # print([v.label for v in bowtie.vertices])

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

    write_graph_to_dotfile(bowtie, './bowtie.dot')
    return bowtie


def create_fault_tree(size=3) -> Graph:
    level_sizes = []
    orig_size = size
    next_level_size = 1  # We start at 1
    while orig_size > 0:
        level_sizes.append(min(orig_size, next_level_size))
        orig_size -= next_level_size
        next_level_size += 1

    level_sizes.reverse()

    graph = Graph(directed=True)

    last_level_vertices = []

    set_top_event = 0
    for level_size in level_sizes:
        this_level_vertices = [Vertex(graph) for _ in range(level_size)]

        # adding to the graph
        [graph.add_vertex(v) for v in this_level_vertices]

        # add connection to previous level, if already defined
        if last_level_vertices:
            for last_level_vertex in last_level_vertices:
                graph.add_edge(
                    Edge(tail=last_level_vertex,
                         head=this_level_vertices[random.randint(0, len(this_level_vertices) - 1)]))

        last_level_vertices = this_level_vertices

        # Mark the Top event
        if set_top_event == len(level_sizes) - 1:
            this_level_vertices[0].label = TOP_EVENT_LABEL
        set_top_event += 1

    with open('./rand_fault_tree.dot', 'w') as f:
        graph_io.write_dot(graph, f, directed=True)
    return graph


def create_event_tree(size=3):
    event_tree_reversed = create_fault_tree(size)

    # Reverse all edges in event_tree_reversed
    for edge in event_tree_reversed.edges:
        temp = edge.head
        edge._head = edge.tail
        edge._tail = temp

    with open('./rand_event_tree.dot', 'w') as f:
        graph_io.write_dot(event_tree_reversed, f, directed=True)
    return event_tree_reversed


def create_fault_tree_trainingset(test_fault_tree):
    print("createTrainingSetFaultTree")
    return TrainingSet([[False, False, False], [False, True, False], [True, False, False], [True, True, True]],
                       {"a": 1, "b": 2, "c": 3})


def create_event_tree_trainingset(test_event_tree):
    print("createTrainingSetEventTree")
    return TrainingSet([1])


def get_top_event(test_bowtie: Graph):
    print("getTopEvent")
    top_events = []
    for vtx in test_bowtie.vertices:
        if vtx.label[:len(TOP_EVENT_LABEL)] == TOP_EVENT_LABEL:
            top_events.append(vtx)
    if len(top_events) != 1:
        raise AttributeError(f"Graph has no '{TOP_EVENT_LABEL}' node")
    return top_events[0]


def get_learning_parameters():
    print("getLearningParameters")
    return 1


# noinspection PyTypeChecker
def create_undirected_tree(training_set):
    events_size = training_set.get_events_size()
    weights = [[0] * events_size for _ in range(events_size)]
    for i in range(0, events_size):
        for j in range(0, events_size):
            weights[i][j] = compute_mutual_information(training_set, i, j)

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


def compute_mutual_information(training_set, event1, event2):
    weight = 0
    for event1_state in [True, False]:
        for event2_state in [True, False]:
            probability_event1 = training_set.compute_single_probability(event1, event1_state)
            probability_event2 = training_set.compute_single_probability(event2, event2_state)
            probability_event1_and_event2 = training_set.compute_combined_probability(event1, event1_state, event2,
                                                                                      event2_state)
            probability = probability_event1_and_event2 / (probability_event1 * probability_event2)
            if any([i == 0 for i in [probability_event1_and_event2, probability_event1, probability_event2]]):
                weight += 0
            else:
                probabilityLog = probability.log10() / Decimal(2).log10()
                weight += probability_event1_and_event2 * probabilityLog

    return weight if event1 != event2 else None


def create_directed_event_tree(undirected_event_tree: Graph):
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


def create_directed_fault_tree(undirected_fault_tree: Graph):
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
    print("createQuantitativeEventTree")

    write_graph_to_dotfile(G, './create_quantitative_event_tree.dot')
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
    print("createQuantitativeFaultTree")

    write_graph_to_dotfile(G, './create_quantitative_fault_tree.dot')
    return G, cpt


def write_graph_to_dotfile(quantitative_bowtie, filename):
    graph = quantitative_bowtie.deepcopy()

    with open(filename, 'w') as f:
        for v in graph.vertices:
            v.new_label = f"[{v.label}]"
            # print(v.probability)
            for k in v.probability.keys():
                # print(type(k))
                if isinstance(k, Vertex):
                    v.new_label += f"\n {k.label} -> {v.probability[k]}"
                elif isinstance(k, str):
                    v.new_label += f"\n {k} -> {v.probability[k]}"
                else:
                    v.new_label += f"\n {k} -> {v.probability[k]}"
        for v in graph.vertices:
            v.label = v.new_label

        for e in graph.edges:
            e._weight = " "

        graph_io.write_dot(graph, f, True)


if __name__ == '__main__':
    start()
