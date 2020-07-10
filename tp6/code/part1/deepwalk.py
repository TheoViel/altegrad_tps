"""
Deep Learning on Graphs - ALTEGRAD - Dec 2019
"""

import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec


# Task 1

def random_walk(G, node, walk_length):
    """ 
    Simulates a random walk of length "walk_length" starting from node "node"
    """
    walk = [node]
    for i in range(walk_length):
        neighbors = list(G.neighbors(walk[-1]))

        if i > 1 and len(neighbors) > 1:
            neighbors = [n for n in neighbors if n != walk[i-2]]  # Modified with remark of Question 1

        walk.append(np.random.choice(neighbors))


    return [str(node) for node in walk]


# Task 2

def generate_walks(G, num_walks, walk_length):
    """
    Runs "num_walks" random walks from each node
    """
    walks = []
    for i in range(num_walks):
        for node in list(G.nodes()):
            walks.append(random_walk(G, node, walk_length))

    return walks



def deepwalk(G, num_walks, walk_length, n_dim):
    """
    Simulates walks and uses the Skipgram model to learn node representations
    """
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    # model = Word2Vec(size=n_dim, window=8, min_count=0, sg=1, workers=8)
    # model.build_vocab(walks)
    # model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model
