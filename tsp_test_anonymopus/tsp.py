#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Multilayer Perceptron Test
# Base URL     : https://wikidocs.net/63565
# Simple CNN for MNIST test (Truth test)
###########################################################################
_description = '''\
====================================================
tsp.py : TSP and Simulated Annealing Test
====================================================
Example : python tsp.py 
'''
# Common Class
import tsp_IO_func as tspio
from nodes_generator import NodeGenerator
# Original Class
from simulated_annealing import SimulatedAnnealing
# modified Class
from tsp_operation_proc import tsp_op
from tsp_operation_proc import Search_Operation
from sa_proc import SimulatedAnnealing_proc

def Ready_for_Operation():
    _operation_param = []
    args = tspio.ArgumentParse(_description, _operation_param)
    # generate random list of nodes
    if args.loaddatafromfile :
        nodes = tspio.Load_Initial_Data_File(args)
        args.population_size = tspio.Get_number_of_cities(nodes)
    else:
        nodes = NodeGenerator(args.size_width, args.size_height, args.population_size).generate()
        tspio.Save_Initial_Data_File(args, nodes)

    tspio.Parameter_msg(args)
    return args, nodes

def original_main():
    # Ready
    args, nodes = Ready_for_Operation()
    # run simulated annealing algorithm with 2-opt
    sa = SimulatedAnnealing(nodes, args)
    sa.anneal()
    # animate
    sa.animateSolutions(args.active_animation)
    # Comparing to Initial and Final
    tspio.Visual_Comparison(sa)
    # show the improvement over time
    sa.plotLearning()

def my_main():
    # Ready
    args, nodes = Ready_for_Operation()
    # run simulated annealing algorithm with 2-opt
    c_tsp_op    = tsp_op(nodes, args)
    c_sa        = SimulatedAnnealing_proc(pf_cost=c_tsp_op.cost, _init_solution= c_tsp_op.get_init_solution(), args=args)
    c_search    = Search_Operation(c_sa, nodes)

    # Main Operation
    c_search.operation()
    # animate
    c_search.animateSolutions(args.active_animation)
    # Comparing to Initial and Final
    tspio.Visual_Comparison(c_search)
    # show the improvement over time
    c_search.plotLearning()

# =================================================================
# Main Routine
# =================================================================
if __name__ == "__main__":
    b_Original = False
    if b_Original:
        original_main()
    else:
        my_main()

    print("===================================================")
    print("Process Finished ")
    print("Original TSP code based simulation" if b_Original else "Modified TSP code ")
    print("===================================================")