from random import random, randint

from graph import *
from trainingset import *

G = Graph(directed=True)
v1 = Vertex(G, label="v1")
v2 = Vertex(G, label="v2")
v3 = Vertex(G, label="v3")
v4 = Vertex(G, label="v4")
v5 = Vertex(G, label="v5")
v6 = Vertex(G, label="v6")
v7 = Vertex(G, label="v7")
v8 = Vertex(G, label="v8")
v9 = Vertex(G, label="v9")
v10 = Vertex(G, label="v10")
G.add_vertex(v1)
G.add_vertex(v2)
G.add_vertex(v3)
G.add_vertex(v4)
G.add_vertex(v5)
G.add_vertex(v6)
G.add_vertex(v7)
G.add_vertex(v8)
G.add_vertex(v9)
G.add_vertex(v10)
e1 = Edge(v1, v2)
e2 = Edge(v1, v3)
e3 = Edge(v2, v4)
e4 = Edge(v3, v4)
e5 = Edge(v4, v5)
e6 = Edge(v5, v6)
e7 = Edge(v5, v7)
e8 = Edge(v6, v8)
e9 = Edge(v6, v9)
e10 = Edge(v7, v10)
G.add_edge(e1)
G.add_edge(e2)
G.add_edge(e3)
G.add_edge(e4)
G.add_edge(e5)
G.add_edge(e6)
G.add_edge(e7)
G.add_edge(e8)
G.add_edge(e9)
G.add_edge(e10)
print(G.vertices)
print(G.edges)
tr = TrainingSet(training_set={
    'event_names': [v.label for v in G.vertices],
    'observations': [
        [randint(0, 1) for _ in G.vertices]
        for _ in range(1000)
    ]

})
