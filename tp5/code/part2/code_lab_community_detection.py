"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import scipy
import numpy as np
import networkx as nx

from random import randint
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigs


# Task 5

def spectral_clustering(G, k):
    L = nx.normalized_laplacian_matrix(G).astype(float)

    e, U = eigs(L, k, which='SM')
    e = e.real
    U = U.real
    U = U[:, np.argsort(e)]  # sort by eigenvalue

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(U)
    y = kmeans.labels_

    clustering = dict(zip(G.nodes(), y))
    return clustering


# Task 6

G = nx.read_edgelist(
    path="../datasets/CA-HepTh.txt",
    comments="#",
    delimiter="\t",
    nodetype=int,
    create_using=nx.Graph(),
)

G_lcc = max(nx.connected_component_subgraphs(G), key=len)

clustering = spectral_clustering(G_lcc, k=60)


# Task 7

def modularity(G, clustering):
    n_clusters = len(np.unique(clustering.values()))
    mod = 0 
    m = G.number_of_edges()
    
    for i in range(n_clusters):
        node_list = [n for n, v in clustering.items() if v == i]
        sub = G.subgraph(node_list)

        lc = sub.number_of_edges() 
        dc = np.sum([G.degree(node) for node in sub.nodes()])
        
        mod += (lc / m - (dc / (2 * m)) ** 2)
        
    return mod


# Task 8

k = 50

random_clustering = dict(zip(G.nodes(), np.random.randint(k, size=len(G.nodes))))
clustering = spectral_clustering(G_lcc, k=k)

print(f"Modularity of Spectral Clustering: {modularity(G_lcc, clustering) :.3f}")
print(f"Modularity of Random Clustering: {modularity(G_lcc, random_clustering) :.3f}")