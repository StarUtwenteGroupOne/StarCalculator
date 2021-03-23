# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import math
import random
import numpy as np

import graph_io
from algorithm_three_test import tr, G
from graph import *
from trainingset import *
from math import floor, ceil

from graph import Graph, Vertex, Edge

from graph import Graph, Vertex, Edge
from trainingset import TrainingSet

# noinspection PyTypeChecker
VertexList = [Vertex]

TOP_EVENT_LABEL = "TE"

def start():
    test_bowtie = create_test_bowtie()
    training_set_event_tree = create_event_tree_trainingset(test_bowtie)
    training_set_fault_tree = create_fault_tree_trainingset(test_bowtie)
    top_event = get_top_event(test_bowtie)
    learning_parameters = get_learning_parameters()
    bowtie = create_quantitative_bowtie(training_set_event_tree, training_set_fault_tree, top_event,
                                        learning_parameters)
    print_quantitative_bowtie(bowtie)


def create_quantitative_bowtie(training_set_event_tree, training_set_fault_tree, top_event, learning_parameters):
    undirected_event_tree = create_undirected_tree(training_set_event_tree)
    undirected_fault_tree = create_undirected_tree(training_set_fault_tree)
    directed_event_tree = create_directed_tree(undirected_event_tree, top_event)
    directed_fault_tree = create_directed_tree(undirected_fault_tree, top_event)
    quantitative_event_tree, probability_of_event_tree = create_quantitative_event_tree(directed_event_tree,
                                                                                        training_set_event_tree)
    quantitative_fault_tree, probability_of_fault_tree = create_quantitative_fault_tree(directed_fault_tree,
                                                                                        training_set_fault_tree)
    quantitative_bowtie = create_quantitative_bowtie_from_trees(quantitative_event_tree, quantitative_fault_tree)
    return quantitative_bowtie


def create_test_bowtie(size=6):
    print("createTestBowtie")

    # The size needs to be one larger because we will merge the top events later.
    size += 1

    fault_tree = create_fault_tree(ceil(size / 2))
    event_tree = create_event_tree(size - ceil(size / 2))

    return create_quantitative_bowtie_from_trees(fault_tree, event_tree)


def create_quantitative_bowtie_from_trees(event_tree, fault_tree):
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

    with open('./rand_bowtie.dot', 'w') as f:
        graph_io.write_dot(bowtie, f, directed=True)
    return bowtie


def create_fault_tree(size=3) -> Graph:
    level_sizes = []
    orig_size = size
    next_level_size = 1  # We start at 1
    while orig_size > 0:
        level_sizes.append(next_level_size)
        orig_size -= next_level_size
        next_level_size += 1


    level_sizes = reversed(level_sizes)

    graph = Graph(directed=True)

    last_level_vertices = []
    for level_size in level_sizes:
        this_level_vertices = [Vertex(graph) for _ in range(level_size)]

        # adding to the graph
        [graph.add_vertex(v) for v in this_level_vertices]

        # add connection to previous level, if already defined
        if last_level_vertices:
            for last_level_vertex in last_level_vertices:
                graph.add_edge(
                    Edge(tail=last_level_vertex,
                         head=this_level_vertices[random.randint(0, len(this_level_vertices) - 1)],
                         weight=random.random()))

        last_level_vertices = this_level_vertices

        # Mark the Top event
        if len(this_level_vertices) == 1:
            this_level_vertices[0].label = TOP_EVENT_LABEL

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


    return TrainingSet([1])


def create_event_tree_trainingset(test_event_tree):
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
    weights = [[0]*events_size for _ in range(events_size)]
    for i in range(0, events_size - 1):
        for j in range(1, events_size):
            weights[i][j] = compute_mutual_information(training_set, i, j)

    graph = Graph(False)
    available_events = [True]*events_size
    available_events[0] = False
    vertices: VertexList = [Vertex(graph)] + [None for _ in range(events_size-1)]
    graph.add_vertex(vertices[0])
    for step in range(1, events_size):
        highest_i = -1
        highest_j = -1
        for i in range(0, events_size):
            for j in range(1, events_size):
                if not available_events[i] and available_events[j]:
                    if highest_i == -1 or weights[highest_i][highest_j] <= weights[i][j]:
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


def create_directed_tree(undirected_tree, top_event):
    print("createDirectedTree")
    return 1


def create_quantitative_event_tree(directed_event_tree, training_set_event_tree):
    probability_of_happening = []
    G = directed_event_tree
    tr = training_set_event_tree
    for v in G.vertices:
        vertex = v
        probability_of_happening_i = {}
        list1 = tr.get_observations_by_event_name(v.label)
        for e in v.incidence:
        # for e in G.edges:
        #     if e.tail == vertex:
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

    with open('./create_quantitative_event_tree.dot', 'w') as f:
        for v in G.vertices:
            for k in v.probability.keys():
                v.label += f" {k.label} -> {v.probability[k]}"
        graph_io.write_dot(G, f, True)
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



def print_quantitative_bowtie(quantitative_bowtie):
    print("printQuantitativeBowTie")
    return 1


if __name__ == '__main__':
    et = create_event_tree(20)
    ft = create_fault_tree(20)
    (_, _) = create_quantitative_event_tree(directed_event_tree=G, training_set_event_tree=tr)

    create_quantitative_bowtie_from_trees(et, ft)
