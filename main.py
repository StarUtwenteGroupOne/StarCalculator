# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import os

from config import OUTPUT_DIR
from learn_bowtie import create_bowtie, create_quantitative_event_tree, \
    create_quantitative_fault_tree, orient_polytree, orient_tree
from lib.graph import Vertex
from maximal_weight_spanning_tree import maximal_weight_spanning_tree

from read_trainingset import read_trainingset
# noinspection PyTypeChecker
from write_graph import write_graph_to_dotfile


def start():
    training_set_fault_tree = get_paper_fault_tree()
    training_set_event_tree = get_paper_event_tree()

    bowtie = create_quantitative_bowtie(training_set_event_tree,
                                        training_set_fault_tree)
    write_graph_to_dotfile(bowtie, filename="final_bowtie.dot", with_probabilities=True)


def create_quantitative_bowtie(training_set_event_tree, training_set_fault_tree):
    undirected_fault_tree = maximal_weight_spanning_tree(training_set_fault_tree)
    undirected_event_tree = maximal_weight_spanning_tree(training_set_event_tree)
    directed_fault_tree = orient_polytree(undirected_fault_tree)
    directed_event_tree = orient_tree(undirected_event_tree)
    quantitative_event_tree, probability_of_event_tree = create_quantitative_event_tree(directed_event_tree,
                                                                                        training_set_event_tree)
    quantitative_fault_tree, probability_of_fault_tree = create_quantitative_fault_tree(directed_fault_tree,
                                                                                        training_set_fault_tree)
    quantitative_bowtie = create_bowtie(quantitative_event_tree, quantitative_fault_tree)
    return quantitative_bowtie


def get_paper_fault_tree():

    filenames = ['faulttree.csv', 'FT_other_order.csv', 'FT_and_or.csv']

    return read_trainingset(filenames[2])


def get_paper_event_tree():

    filenames = ['eventtree.csv', 'ET_other_order.csv']

    return read_trainingset(filenames[1])


if __name__ == '__main__':
    start()
