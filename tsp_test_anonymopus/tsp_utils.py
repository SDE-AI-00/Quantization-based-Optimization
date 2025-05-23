#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# TSP Simulating 
###########################################################################
_description = '''\
====================================================
tsp_utils.py : Utility functions for TSP 
====================================================
Example : No operation. Just provide class and functions 
'''

import random
import numpy as np

# Create the distance matrix
def vectorToDistMatrix(coords):
    return np.sqrt((np.square(coords[:, np.newaxis] - coords).sum(axis=2)))

# Computes the initial solution (nearest neighbour strategy)
def nearestNeighbourSolution(args, dist_matrix):
    # If args.loaddatafromfile is True, then node is set to be zero. (fixed)
    node = random.randrange(len(dist_matrix)) if not args.loaddatafromfile else 0
    result = [node]

    nodes_to_visit = list(range(len(dist_matrix)))
    nodes_to_visit.remove(node)

    while nodes_to_visit:
        nearest_node = min([(dist_matrix[node][j], j) for j in nodes_to_visit], key=lambda x: x[0])
        node = nearest_node[1]
        nodes_to_visit.remove(node)
        result.append(node)

    return result
