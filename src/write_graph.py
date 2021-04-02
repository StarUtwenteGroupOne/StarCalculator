import os

from src.config import OUTPUT_DIR
from lib.ut_graphs import graph_io
from lib.ut_graphs.graph import Vertex


def write_graph_to_dotfile(quantitative_bowtie, filename, with_probabilities=True):
    graph = quantitative_bowtie.deepcopy()

    with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
        for v in graph.vertices:
            v.new_label = f"[{v.label}]"
            if with_probabilities:
                for k in v.probability.keys():
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