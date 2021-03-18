# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import math
import random
from math import floor, ceil

from graph import Graph, Vertex, Edge

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
    quantitative_event_tree = create_quantitative_event_tree(directed_event_tree, learning_parameters)
    quantitative_fault_tree = create_quantitative_fault_tree(directed_fault_tree, learning_parameters)
    quantitative_bowtie = create_quantitative_bowtie_from_trees(quantitative_event_tree, quantitative_fault_tree)
    return quantitative_bowtie


def create_test_bowtie(size=6):
    print("createTestBowtie")

    # The size needs to be one larger because we will merge the top events later.
    size += 1

    fault_tree = create_fault_tree(ceil(size / 2))
    event_tree = create_event_tree(size - ceil(size / 2))

    # And now we need to union the graph
    bowtie = Graph(directed=True)

    bowtie._v += fault_tree.vertices + event_tree.vertices
    bowtie._e += fault_tree.edges + event_tree.edges

    top_events = []
    # Find the two top events:
    for vtx in bowtie.vertices:
        if vtx.label == "TE":
            top_events.append(vtx)

    assert len(top_events) == 2

    for neighbor in top_events[1].neighbours:
        bowtie.add_edge(Edge(tail=top_events[0], head=neighbor))

        # Delete the now duplicate top event
        bowtie.del_vertex(top_events[1])

    return bowtie


def create_fault_tree(size=3) -> Graph:
    level_sizes = []
    next_level_size = 1  # We start at 1
    while size > next_level_size + 1:
        level_sizes.append(next_level_size)

    level_sizes = reversed(level_sizes)

    graph = Graph(directed=True)

    last_level_vertices = set()
    for level_size in level_sizes:
        this_level_vertices = [Vertex(graph) for _ in range(level_size)]

        # add connection to previous level, if already defined
        if last_level_vertices:
            for last_level_vertex in last_level_vertices:
                graph.add_edge(
                    Edge(tail=last_level_vertex,
                         head=this_level_vertices[random.randint(len(this_level_vertices))],
                         weight=random.random()))

        # Mark the Top event
        if len(this_level_vertices) == 1:
            this_level_vertices[0].label = "TE"

    return graph


def create_event_tree(size=3):
    event_tree_reversed = create_fault_tree(size)

    # Reverse all edges in event_tree_reversed
    for edge in event_tree_reversed.edges:
        temp = edge.head
        edge._head = edge.tail
        edge._tail = temp
    return event_tree_reversed


def create_fault_tree_trainingset(test_bowtie):
    print("createTrainingSetFaultTree")
    return TrainingSet([1])


def create_event_tree_trainingset(test_bowtie):
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


def create_quantitative_event_tree(directed_event_tree, learning_parameters):
    print("createQuantitativeEventTree")
    return 1


def create_quantitative_fault_tree(directed_fault_tree, learning_parameters):
    print("createQuantitativeFaultTree")
    return 1


def create_quantitative_bowtie_from_trees(quantitative_event_tree, quantitative_fault_tree):
    print("createBowTie")
    return 1


def print_quantitative_bowtie(quantitative_bowtie):
    print("printQuantitativeBowTie")
    return 1


if __name__ == '__main__':
    start()
