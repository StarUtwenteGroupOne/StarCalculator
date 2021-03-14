import random
import numpy as np
from graph import *
from trainingset import *


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
    quantitative_event_tree = create_quantitative_event_tree(directed_event_tree, training_set_event_tree)
    quantitative_fault_tree = create_quantitative_fault_tree(directed_fault_tree, training_set_fault_tree)
    quantitative_bowtie = create_quantitative_bowtie_from_trees(quantitative_event_tree, quantitative_fault_tree)
    return quantitative_bowtie


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


def create_quantitative_event_tree(directed_event_tree, training_set_event_tree):
    s = []
    G = directed_event_tree
    tr = training_set_event_tree
    for v in G.vertices:
        vertex = v
        s_i = []
        list1 = tr.get_observations_by_event_name(v.label)
        for e in G.edges:
            if e.tail == vertex:
                list2 = tr.get_observations_by_event_name(e.head.label)
                n_i_j = 0
                n_i = 0
                for i in range(0, len(list1)):
                    if list1[i] == list2[i] and list2[i]:
                        n_i_j += 1
                    if list1[i]:
                        n_i += 1
                s_i.append(n_i_j / n_i)
        s.append(s_i)
    print("createQuantitativeEventTree")
    return 1


def create_quantitative_fault_tree(directed_fault_tree, training_set_fault_tree):
    cpt = []
    G = directed_fault_tree
    tr = training_set_fault_tree
    for v in G.vertices:
        cpt_i = []
        parents = [tr.get_observations_by_event_name(v.label)]
        helping_dict = {}
        for e in G.edges:
            if e.head == v:
                parents.append(tr.get_observations_by_event_name(e.tail.label))
        parents = np.array(parents).transpose()
        total = parents.shape[0] + 2 ** (parents.shape[1] - 1)
        for p in parents:
            if helping_dict[p] is None:
                helping_dict[p] = 1
            else:
                helping_dict[p] = helping_dict[p] + 1
        for k in helping_dict.keys():
            cpt_i.append(helping_dict[k] / total)
        cpt.append(cpt_i)
    print("createQuantitativeFaultTree")
    return cpt


def create_quantitative_bowtie_from_trees(quantitative_event_tree, quantitative_fault_tree):
    print("createBowTie")
    return 1


def print_quantitative_bowtie(quantitative_bowtie):
    print("printQuantitativeBowTie")
    return 1


if __name__ == '__main__':
    start()
