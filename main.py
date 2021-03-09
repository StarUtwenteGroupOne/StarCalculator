# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import random


def start():
    testBowtie = create_test_bowtie()
    trainingSetEventTree = create_trainingset_event_tree(testBowtie)
    trainingSetFaultTree = create_trainingset_fault_tree(testBowtie)
    topEvent = get_top_event(testBowtie)
    learningParameters = get_learning_parameters()
    bowtie = createQuantitativeBowTie(trainingSetEventTree, trainingSetFaultTree, topEvent, learningParameters)
    print_quantitative_bowtie(bowtie)


def createQuantitativeBowTie(trainingSetEventTree, trainingSetFaultTree, topEvent, learningParameters):
    undirectedEventTree = create_undirected_tree(trainingSetEventTree)
    undirectedFaultTree = create_undirected_tree(trainingSetFaultTree)
    directedEventTree = create_directed_tree(undirectedEventTree, topEvent)
    directedFaultTree = create_directed_tree(undirectedFaultTree, topEvent)
    quantitativeEventTree = create_quantitative_event_tree(directedEventTree, learningParameters)
    quantitativeFaultTree = create_quantitative_fault_tree(directedFaultTree, learningParameters)
    quantitativeBowtie = create_quantitative_bowtie_from_trees(quantitativeEventTree, quantitativeFaultTree)
    return quantitativeBowtie


def create_test_bowtie():
    print("createTestBowtie")
    # A hardcoded fault tree
    return 1


def create_trainingset_fault_tree(test_bowtie):
    print("createTrainingSetFaultTree")
    return 1


def create_trainingset_event_tree(test_bowtie):
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
