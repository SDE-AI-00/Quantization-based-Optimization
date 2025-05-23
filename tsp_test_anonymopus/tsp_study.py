#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - TSP Test
###########################################################################
_description = '''\
====================================================
tsp.py : TSP Test
====================================================
Example : python tsp_study.py  
'''
# Common Class
import pickle
import tsp_IO_func as tspio
from nodes_generator import NodeGenerator

from tsp_operation_proc import tsp_op
from tsp_operation_proc import parabolic_washboard_potential
from tsp_operation_proc import Search_Operation
from tsp_operation_proc import Search_Operation_PWP
from sa_proc import SimulatedAnnealing_proc
from qz_proc import Quantization_proc
from qa_proc import QuantumAnnealing_proc
from ga_proc import GeneticAlgorithm_proc
# =================================================================
# Core Operation
# =================================================================
class Study_class:
    def __init__(self, bUseParam=False):
        self.gL_optimizer = [SimulatedAnnealing_proc,
                        Quantization_proc,
                        QuantumAnnealing_proc,
                        GeneticAlgorithm_proc
                        ]

        args, nodes = self.Ready_for_Operation(_operation_param, bUseParam=bUseParam)
        self.args       = args
        self.nodes      = nodes

    # -----------------------------------------------------------------
    # Service Function
    # -----------------------------------------------------------------
    def Ready_for_Operation(self, _operation_param, bUseParam=False):
        args = tspio.ArgumentParse(_description, _operation_param, bUseParam=bUseParam)
        # generate random list of nodes
        if args.loaddatafromfile:
            nodes = tspio.Load_Initial_Data_File(args)
            args.population_size = tspio.Get_number_of_cities(nodes)
        else:
            nodes = NodeGenerator(args.size_width, args.size_height, args.population_size).generate()
            tspio.Save_Initial_Data_File(args, nodes)

        tspio.Parameter_msg(args)

        return args, nodes

    def final_operation_proc(self, c_search):
        _ret = 0
        # animate
        c_search.animateSolutions(self.args.active_animation)
        if self.args.debug_message == 1:
            if self.args.tsp_pwp:
                # Comparing to Initial and Final
                tspio.Visual_Comparison(c_search)
            else:
                pass
            # show the improvement over time
            c_search.plotLearning()
        else:
            _ret = c_search.processing_info
        return _ret

    def get_change_in_history(self, f_history, b_op=True):
        _ret = 0
        if b_op :
            _sol_history    = f_history()
            _best_sol       = _sol_history[0]
            for _k, _data in enumerate(_sol_history):
                if _best_sol == _data : pass
                else:
                    _best_sol = _data
                    print("[ %3d ] " %_k, _data)
                    _ret += 1
            print ("Update number : %d" %_ret)
        else:
            _ret = -1
        return _ret

    def Final_Operation(self, c_search):
        # 일반적인 Final Operation
        _ret = self.final_operation_proc(c_search)
        # Study 에서 Final Operation
        _ret = self.get_change_in_history(c_search.get_solution_history, b_op=False)
        _ret = self.get_change_in_history(c_search.get_best_sol_history)

        return _ret

    def __call__(self, _operation_param):
        if self.args.tsp_pwp:
            # TSP with 2-opt
            c_tsp_op    = tsp_op(self.nodes, self.args)
            c_optimizer = self.gL_optimizer[self.args.algorithm](pf_cost=c_tsp_op.cost, _init_solution=c_tsp_op.get_init_solution(), args=self.args)
            c_search    = Search_Operation(c_optimizer, self.nodes)
            # Main Operation
            c_search.operation()
            # Final Operation
            _ret = self.Final_Operation(c_search)
        else:
            # PWP
            print("This program is for the analysis of the TSP problem")
            print("Please Check the program name !! ")
            _ret = 0
        return _ret

# =================================================================
# Main Routine
# =================================================================
if __name__ == "__main__":

    _operation_param= []

    c_study = Study_class()
    c_study(_operation_param)

    print("===================================================")
    print("Process Finished ")
    print("TSP Study ")
    print("===================================================")