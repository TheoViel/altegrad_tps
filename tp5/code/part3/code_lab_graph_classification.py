"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Task 9

def create_dataset(n=100):
    Gs = list()
    y = list()

    for i in range(3, n + 3):
        Gs.append(nx.cycle_graph(i))
        y.append(0)
        Gs.append(nx.path_graph(i))
        y.append(1)

    return Gs, y


Gs, y = create_dataset()
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.1)


# Compute the shortest path kernel

def shortest_path_kernel(Gs_train, Gs_test):
    all_paths = dict()
    sp_counts_train = dict()

    for i, G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    sp_counts_test = dict()

    for i, G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(G_train), len(all_paths)))
    for i in range(len(G_train)):
        for length in sp_counts_train[i]:
            phi_train[i, all_paths[length]] = sp_counts_train[i][length]

    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i, all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


############## Task 10
# Compute the graphlet kernel
def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]

    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0, 1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0, 1)
    graphlets[2].add_edge(1, 2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0, 1)
    graphlets[3].add_edge(1, 2)
    graphlets[3].add_edge(0, 2)

    phi_train = np.zeros((len(G_train), 4))

    for i, G in enumerate(Gs_train):
        for _ in range(n_samples):
            sub = G.subgraph(np.random.choice(np.arange(len(G.edges())), size=3))

            for j, graphlet in enumerate(graphlets):
                if nx.is_isomorphic(sub, graphlet):
                    phi_train[i, j] += 1

    phi_test = np.zeros((len(G_test), 4))

    for i, G in enumerate(Gs_test):
        for _ in range(n_samples):
            sub = G.subgraph(np.random.choice(np.arange(len(G.edges())), size=3))

            for j, graphlet in enumerate(graphlets):
                if nx.is_isomorphic(sub, graphlet):
                    phi_test[i, j] += 1

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


# Task 11

K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)

K_train_g, K_test_g = graphlet_kernel(G_train, G_test, n_samples=200)


# Task 12

clf = SVC(kernel="precomputed")
clf.fit(K_train_sp, y_train)
y_pred = clf.predict(K_test_sp)

print(f"Accuracy with shortest path : {accuracy_score(y_pred, y_test) :.2f}")

clf = SVC(kernel="precomputed")
clf.fit(K_train_g, y_train)
y_pred = clf.predict(K_test_g)

print(f"Accuracy with graphlet : {accuracy_score(y_pred, y_test) :.2f}")

