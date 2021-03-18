# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import math
import random
import numpy as np
from graph import *
from trainingset import *

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


def create_test_bowtie():
    print("createTestBowtie")
    # A hardcoded fault tree
    return 1


def create_trainingset_fault_tree(test_bowtie):
    print("createTrainingSetFaultTree")
    return TrainingSet([1])


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
