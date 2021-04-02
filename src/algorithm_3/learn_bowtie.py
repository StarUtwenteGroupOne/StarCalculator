import numpy as np

from src.config import TOP_EVENT_LABEL, ALPHA
from lib.ut_graphs.graph import Graph, Edge
from src.write_graph import write_graph_to_dotfile


def create_bowtie(event_tree, fault_tree):
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

    write_graph_to_dotfile(bowtie, 'bowtie.dot')
    return bowtie


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

    write_graph_to_dotfile(G,'create_quantitative_event_tree.dot')
    return G, probability_of_happening


def create_quantitative_fault_tree(directed_fault_tree, training_set_fault_tree):
    cpt = []
    G = directed_fault_tree
    tr = training_set_fault_tree
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
            total = vertices_set.shape[0] + (2 ** (vertices_set.shape[1] - 1)) * ALPHA

            for p in vertices_set:
                p_as_np = np.array(p[1:])
                if not (str(p_as_np) in helping_dict.keys()):
                    helping_dict[str(p_as_np)] = 1
                else:
                    helping_dict[str(p_as_np)] = helping_dict[str(p_as_np)] + 1

            for k in helping_dict.keys():
                cpt_i[k] = (helping_dict[k] + ALPHA) / total
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
                v.probability[k] = (v.probability[k] + ALPHA) / (size + 2 * ALPHA)

    write_graph_to_dotfile(G, 'create_quantitative_fault_tree.dot')
    return G, cpt