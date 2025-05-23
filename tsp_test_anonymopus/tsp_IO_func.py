#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Multilayer Perceptron Test
# Base URL     : https://wikidocs.net/63565
# Simple CNN for MNIST test (Truth test)
###########################################################################
import matplotlib.pyplot as plt
import pickle
import glob

import argparse
import textwrap

def ArgumentParse(_intro_msg, L_Param, bUseParam=False):
    parser = argparse.ArgumentParser(
        prog='tsp.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(_intro_msg))

    parser.add_argument('-t', '--temp', help="Initial Temperature",
                        type=int, default=1000)
    parser.add_argument('-a', '--alpha', help="alpha",
                        type=float, default=0.9995)
    parser.add_argument('-wd', '--size_width', help="size_width",
                        type=int, default=200)
    parser.add_argument('-ht', '--size_height', help="size_height",
                        type=int, default=200)
    parser.add_argument('-p', '--population_size', help="population_size",
                        type=int, default=100)
    parser.add_argument('-i', '--stopping_iter', help="Stopping Iteration",
                        type=int, default=100000)
    parser.add_argument('-st', '--stopping_temp', help="stopping_temp",
                        type=float, default=0.00000001)
    parser.add_argument('-fn', '--datafilename', help="File Name for data",
                        type=str, default="init_data.pkl")
    parser.add_argument('-l',  '--loaddatafromfile', help="Load Initial Data from PKLfile [default] 0 (int) [1] Load",
                        type=int, default=0)
    parser.add_argument('-al', '--algorithm', help="Algorithm [0:Default] Simulated_Annealing [1] Quantized Optimization [2] Quantum Optimization [3] GA Optimization",
                        type=int, default=0)
    parser.add_argument('-an', '--active_animation', help="Active Animation",
                        type=int, default=1)
    parser.add_argument('-dm', '--debug_message', help="[0] No Debug Message [1:Default] Active Debug Messgae",
                        type=int, default=1)
    parser.add_argument('-tp', '--tsp_pwp', help="[0] parabolic_washboard_potential [1:Default] Travelling Salesman Problem",
                        type=int, default=1)
    parser.add_argument('-lf', '--tst_func_id', help="[0:Default] parabolic_washboard_potential [1] Xin-She Yang N4 [2] Salomon [3] Drop-Wave [4] Schaffel N2",
                        type=int, default=0)
    parser.add_argument('-bd', '--band', help="Band frequency for parabolic_washboard_potential [10.0: Default]",
                        type=float, default=10.0)
    parser.add_argument('-cm', '--comparison', help="Comapre with the initial and final path [0:default] simultaneous, [1] sequential",
                        type=float, default=0)
    parser.add_argument('-gp', '--genetic_population', help="Number of chromossome for the genetic algorithm",
                        type=int, default=10)

    if bUseParam:
        args = parser.parse_args(L_Param)
    else:
        args = parser.parse_args()

    args.stopping_temp      = 0.1 / (1.0 * args.stopping_iter)
    args.loaddatafromfile   = True if args.loaddatafromfile == 1 and glob.glob("*.pkl") else False
    args.active_animation   = True if args.active_animation else False
    args.debug_message      = True if args.debug_message == 1 else False
    args.tsp_pwp            = True if args.tsp_pwp == 1 else False
    args.comparison         = True if args.comparison == 1 else False

    if args.tsp_pwp : pass
    else:
        args.active_animation   = False
        args.stopping_iter      = 10000

    print(_intro_msg)
    return args


def Parameter_msg(args):
    l_optimizer = ["Simulated Annealing", "Quantized Optimization", "Quantum Annealing", "Genetic Algorithm"]

    _msg = []
    _msg.append("===================================================")
    _msg.append("Parameters : ")
    _msg.append("Initial Temperature : %d " % (args.temp))
    _msg.append("Alpha               : %f " % (args.alpha))
    if args.tsp_pwp == 1:
        _msg.append("Size_width          : %d " % (args.size_width))
        _msg.append("Size_height         : %d " % (args.size_height))
        _msg.append("Population_size     : %d " % (args.population_size))
    else:
        _msg.append("Band Frequency      : %3.2f " % (args.band))
    _msg.append("Stopping_Iteration  : %d " % (args.stopping_iter))
    if args.algorithm == 0:
        _msg.append("Stopping_Temperature: %f " % (args.stopping_temp))
    else: pass
    _msg.append("Algorithm           : %s " % (l_optimizer[args.algorithm]))
    _msg.append("Data Initialization : %s " % ("Load from File" if args.loaddatafromfile else "Generation"))
    _msg.append("Animation           : %s " % ("True" if args.active_animation else "False"))
    _msg.append("===================================================")

    for _m in _msg: print(_m)

    return _msg

def Save_Initial_Data_File(args, _data, _File_name=''):
    _Fname = args.datafilename if _File_name == '' else _File_name
    with open(_Fname, "wb") as _file:
        pickle.dump(_data, _file)

def Load_Initial_Data_File(args, _File_name=''):
    _Fname = args.datafilename if _File_name == '' else _File_name
    with open(_Fname, "rb") as _file:
        _data = pickle.load(_file)
    return _data

def Get_number_of_cities(nodes):
    _ret = nodes.shape[0]
    return _ret

def Visual_Comparison(_data_class):
    # Set Initial
    _points     = _data_class.coords
    _history    = _data_class.solution_history

    # initialize node dots on graph
    x  = [_points[i][0] for i in _history[0] + [_history[0][0]]]
    y  = [_points[i][1] for i in _history[0] + [_history[0][0]]]
    fx = [_points[i][0] for i in _history[-1] + [_history[-1][0]]]
    fy = [_points[i][1] for i in _history[-1] + [_history[-1][0]]]

    # Animation or Sequential
    b_comparison = _data_class.pC_search_method.args.comparison
    if b_comparison:
        # Initial Graph
        plt.plot(x, y, 'co-')
        plt.show()
        # Final Graph
        plt.plot(fx, fy, 'ro-')
        plt.show()
    else:
        # Initial Graph
        plt.subplot(1, 2, 1)
        plt.plot(x, y, 'co-')
        # Final Graph
        plt.subplot(1, 2, 2)
        plt.plot(fx, fy, 'ro-')
        plt.show()
    return 1

def Sequential_Comparison(_data_class):
    # Set Initial
    _points     = _data_class.coords
    _history    = _data_class.solution_history

    # initialize node dots on graph
    x  = [_points[i][0] for i in _history[0] + [_history[0][0]]]
    y  = [_points[i][1] for i in _history[0] + [_history[0][0]]]
    fx = [_points[i][0] for i in _history[-1] + [_history[-1][0]]]
    fy = [_points[i][1] for i in _history[-1] + [_history[-1][0]]]

    # Initial Location
    plt.plot(x, y, 'co-')
    plt.show()

    # Final Location
    plt.plot(fx, fy, 'ro-')
    plt.show()