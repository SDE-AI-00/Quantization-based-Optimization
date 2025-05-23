#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# TSP Simulating 
###########################################################################
_description = '''\
====================================================
nodes_generator.py : Classical Simulated Annealing Test 
                    Generation of Cities for test
====================================================
Example : No operation. Just provide class and functions 
'''
import numpy as np

class NodeGenerator:
    def __init__(self, width, height, nodesNumber):
        self.width = width
        self.height = height
        self.nodesNumber = nodesNumber

    def generate(self):
        xs = np.random.randint(self.width, size=self.nodesNumber)
        ys = np.random.randint(self.height, size=self.nodesNumber)

        return np.column_stack((xs, ys))
