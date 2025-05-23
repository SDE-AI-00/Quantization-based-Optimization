#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Multilayer Perceptron Test
###########################################################################
_description = '''\
====================================================
tsp.py : TSP and Simulated Annealing Test
====================================================
Example : python tsp_proc.py -l 1 
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
gL_optimizer    = [ SimulatedAnnealing_proc,
                    Quantization_proc,
                    QuantumAnnealing_proc,
                    GeneticAlgorithm_proc]
# =================================================================
# Core Operation
# =================================================================
def Ready_for_Operation(_operation_param, bUseParam=False):
    args = tspio.ArgumentParse(_description, _operation_param, bUseParam=bUseParam)
    # generate random list of nodes
    if args.loaddatafromfile :
        nodes = tspio.Load_Initial_Data_File(args)
        args.population_size = tspio.Get_number_of_cities(nodes)
    else:
        nodes = NodeGenerator(args.size_width, args.size_height, args.population_size).generate()
        tspio.Save_Initial_Data_File(args, nodes)

    tspio.Parameter_msg(args)

    return args, nodes

def Final_Operation(args, c_search):
    _ret = 0
    # animate
    c_search.animateSolutions(args.active_animation)
    if args.debug_message == 1:
        if args.tsp_pwp:
            # Comparing to Initial and Final
            tspio.Visual_Comparison(c_search)
        else:
            pass
        # show the improvement over time
        c_search.plotLearning()
    else:
        _ret = c_search.processing_info
    return _ret

def _main(_operation_param, bUseParam=False):
    # Ready
    args, nodes = Ready_for_Operation(_operation_param, bUseParam=bUseParam)
    # Initial Setting for Simulation
    if args.tsp_pwp:
        # TSP with 2-opt
        c_tsp_op    = tsp_op(nodes, args)
        c_optimizer = gL_optimizer[args.algorithm](pf_cost=c_tsp_op.cost, _init_solution= c_tsp_op.get_init_solution(), args=args)
        c_search    = Search_Operation(c_optimizer, nodes)
    else:
        # PWP
        c_pwp_op    = parabolic_washboard_potential(nodes, args)
        c_optimizer = gL_optimizer[args.algorithm](pf_cost=c_pwp_op.cost, _init_solution=c_pwp_op.get_init_solution(), args=args)
        c_search    = Search_Operation_PWP(c_optimizer, c_pwp_op)

    # Main Operation
    c_search.operation()
    # Final Processing
    _ret = Final_Operation(args, c_search)
    return _ret

# =================================================================
# Statistical Operation
# =================================================================
def make_parameter(_operation_param, b_PWP=False):
    # common_parameter := "-l 1 -an 0 -dm 0 -al 0"
    _operation_param.append("-l")
    _operation_param.append("1")
    _operation_param.append("-an")
    _operation_param.append("0")
    _operation_param.append("-dm")
    _operation_param.append("0")
    if b_PWP:
        _operation_param.append("-tp")
        _operation_param.append("-0")
        _operation_param.append("-bd")
        _operation_param.append("10.0")
    else: pass
    _operation_param.append("-al")
    _operation_param.append("0")
    return _operation_param

# Processing Information : [_final_update, _initial_cost, _min_cost, _improvement, self.cost_list]
def final_report(_result):
    # Length of result is equal to iterations
    _total = 1.0 * len(_result)

    n_iterations    = len(_result)
    n_algorithms    = len(_result[0])
    n_items         = len(_result[0][0])

    # make empty database
    l_algigned_data = []
    for _algo in range(n_algorithms):
        l_item = []
        for _item in range(n_items):
            l_itr =[]
            for _iter in range(n_iterations):
                l_itr.append(0)
            l_item.append(l_itr)
        l_algigned_data.append(l_item)

    # aligned data on the database
    for _k, _iter in enumerate(_result):
        for _j, _algo in enumerate(_iter):
            l_algigned_data[_j][0][_k] = _algo[0]   # Final Update
            l_algigned_data[_j][1][_k] = _algo[2]   # minimum cost
            l_algigned_data[_j][2][_k] = _algo[3]   # improvement

    # Final Report
    sa_avg_final_update = (1.0 * sum(l_algigned_data[0][0]))/_total
    sa_avg_min_cost     = (1.0 * sum(l_algigned_data[0][1]))/_total
    sa_avg_improvement  = (1.0 * sum(l_algigned_data[0][2]))/_total
    qo_avg_final_update = (1.0 * sum(l_algigned_data[1][0]))/_total
    qo_avg_min_cost     = (1.0 * sum(l_algigned_data[1][1]))/_total
    qo_avg_improvement  = (1.0 * sum(l_algigned_data[1][2]))/_total
    qa_avg_final_update = (1.0 * sum(l_algigned_data[2][0]))/_total
    qa_avg_min_cost     = (1.0 * sum(l_algigned_data[2][1]))/_total
    qa_avg_improvement  = (1.0 * sum(l_algigned_data[2][2]))/_total

    _msg = []
    _msg.append("===================================================")
    _msg.append("Simulated Annealing")
    _msg.append("   Final Update %5.0f" %sa_avg_final_update)
    _msg.append("   Minimum Cost %5.2f" %sa_avg_min_cost)
    _msg.append("   Improvement  %3.2f" %sa_avg_improvement)
    _msg.append("Quantized Optimization")
    _msg.append("   Final Update %5.0f" %qo_avg_final_update)
    _msg.append("   Minimum Cost %5.2f" %qo_avg_min_cost)
    _msg.append("   Improvement  %3.2f" %qo_avg_improvement)
    _msg.append("Quantum Optimization")
    _msg.append("   Final Update %5.0f" %qa_avg_final_update)
    _msg.append("   Minimum Cost %5.2f" %qa_avg_min_cost)
    _msg.append("   Improvement  %3.2f" %qa_avg_improvement)

    with open("Final_result.txt", "wt") as _file:
        for _msg_line in _msg:
            _file.write(_msg_line + "\n")

    for _msg_line in _msg:
        print(_msg_line)


def s_main(_operation_param, bUseParam=True, _attempt=4):
    _result         = []
    _operation_param= make_parameter(_operation_param, b_PWP=False)

    for _k in range(_attempt):
        print("===================================================")
        print("%2d_th Attemp " %_k)
        print("===================================================")
        _iter_res = []

        # Simulated Annealing
        _operation_param.pop(-1)
        _operation_param.append("0")
        _iter_res.append(_main(_operation_param, bUseParam=bUseParam))

        # Quantized Optimization
        _operation_param.pop(-1)
        _operation_param.append("1")
        _iter_res.append(_main(_operation_param, bUseParam=bUseParam))

        # Quantum Optimization
        _operation_param.pop(-1)
        _operation_param.append("2")
        _iter_res.append(_main(_operation_param, bUseParam=bUseParam))
        _result.append(_iter_res)

    with open("result_multi.pkl", "wb") as _file:
        pickle.dump(_result, _file)

    final_report(_result)
# =================================================================
# Main Routine
# =================================================================
if __name__ == "__main__":
    _operation_param= []

    #s_main(_operation_param, _attempt=1)
    _main(_operation_param)

    print("===================================================")
    print("Process Finished ")
    print("TSP Simulation ")
    print("===================================================")