# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import os

from src.algorithm_3.learn_bowtie import create_bowtie, create_quantitative_event_tree, \
    create_quantitative_fault_tree
from src.algorithm_2.orient_polytree import orient_polytree
from src.algorithm_2.orient_tree import orient_tree
from src.algorithm_1.create_undirected_tree import create_undirected_tree

from src.trainingset.read_trainingset import read_trainingset
# noinspection PyTypeChecker
from src.write_graph import write_graph_to_dotfile


def start():
    training_set_fault_tree = get_paper_fault_tree()
    training_set_event_tree = get_paper_event_tree()

    bowtie = create_quantitative_bowtie(training_set_event_tree,
                                        training_set_fault_tree)
    write_graph_to_dotfile(bowtie, filename="final_bowtie.dot", with_probabilities=True)


def create_quantitative_bowtie(training_set_event_tree, training_set_fault_tree):
    undirected_fault_tree = create_undirected_tree(training_set_fault_tree)
    undirected_event_tree = create_undirected_tree(training_set_event_tree)
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

    return read_trainingset(filenames[1])


def get_paper_event_tree():

    filenames = ['eventtree.csv', 'ET_other_order.csv','ET_or_b.csv']

    return read_trainingset(filenames[1])


if __name__ == '__main__':
    start()
