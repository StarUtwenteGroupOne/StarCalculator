# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import random
from math import floor, ceil

from graph import Graph, Vertex, Edge


def start():
    test_bowtie = create_test_bowtie()
    training_set_event_tree = create_trainingset_event_tree(test_bowtie)
    training_set_fault_tree = create_trainingset_fault_tree(test_bowtie)
    top_event = get_top_event(test_bowtie)
    learning_parameters = get_learning_parameters()
    bowtie = create_quantitative_bowtie(training_set_event_tree, training_set_fault_tree, top_event, learning_parameters)
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
    # A hardcoded fault tree
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
        del top_events[1]

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
                    Edge(last_level_vertex,
                         this_level_vertices[random.randint(len(this_level_vertices))]))

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


def create_event_tree_trainingset(test_bowtie):
    print("createTrainingSetEventTree")
    return 1


def get_top_event(test_bowtie):
    print("getTopEvent")
    return 1


def get_learning_parameters():
    print("getLearningParameters")
    return 1


def create_undirected_tree(training_set):
    print("createUndirectedTree")
    return 1


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
