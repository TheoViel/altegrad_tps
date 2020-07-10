"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")

# Task 1

G = nx.read_edgelist(
    path="../datasets/CA-HepTh.txt",
    comments="#",
    delimiter="\t",
    nodetype=int,
    create_using=nx.Graph(),
)
G_lcc = max(nx.connected_component_subgraphs(G), key=len)

print("G : ")
print(" - Number of nodes :", G.number_of_nodes())
print(" - Number of edges :", G.number_of_edges())

# Task 2

print(" - Number of connected components :", nx.number_connected_components(G))

G_lcc = max(nx.connected_component_subgraphs(G), key=len)

print("\nLargest component :")

print(
    f" - Number of nodes : {G_lcc.number_of_nodes()} ({G_lcc.number_of_nodes() / G.number_of_nodes():.3f} of the graph)"
)
print(
    f" - Number of nodes : {G_lcc.number_of_edges()} ({G_lcc.number_of_edges() / G.number_of_edges():.3f} of the graph)"
)


# Task 3

print("\nDegrees :")

degree_sequence = [G.degree(node) for node in G.nodes()]

print("- Min :", np.min(degree_sequence))
print("- Max :", np.max(degree_sequence))
print(f"- Mean {np.mean(degree_sequence) :.2f}")

# Task 4


# y = nx.degree_histogram(G)  # ugly

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
sns.distplot(degree_sequence)
plt.xlabel("Degrees")


plt.subplot(2, 1, 2)
sns.distplot(degree_sequence, kde=False)
plt.yscale("log")
plt.xlabel("Degrees")

plt.show()
plt.savefig("degrees.png")
plt.show()
