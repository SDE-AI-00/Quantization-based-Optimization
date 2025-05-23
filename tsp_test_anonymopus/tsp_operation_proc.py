#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# TSP Simulating 
###########################################################################
_description = '''\
====================================================
tsp_operation_proc.py : Classical Simulated Annealing Test
====================================================
Example : No operation. Just provide class and functions 
'''
import random
import matplotlib.pyplot as plt
import tsp_utils
import animated_visualizer
from numpy import sin
import numpy as np

# For Arbitrary Test Function
from plot_function import Arbitrary_function

#============================================================================
# Test Function 1
#============================================================================
_guide_line = "----------------------------------------------------"
class parabolic_washboard_potential:
    def __init__(self, coords, args):
        self.args       = args
        self.r_min      = -6.0
        self.r_max      = 6.0

        if args.tst_func_id == 0 :
            self.x_optima   = -0.157
            self.band       = self.args.band # 3.0 or 10.0
            self._name      = "Parabolic Washboard Potential"
            self.equation   = "$f(x) = 0.125 x^2 + 2\sin(10x) + 2$"
            self.op_obj_func= self.PWP_objective_function
        else:
            test_func_id    = self.args.tst_func_id - 1
            c_testfunction  = Arbitrary_function(_param=test_func_id)
            self.x_optima   = c_testfunction.x_optima
            self._name      = c_testfunction.f_name
            self.equation   = c_testfunction.equation
            self.op_obj_func = c_testfunction.op_func


        self.increments = 0.01
        self.inputs     = np.arange(self.r_min, self.r_max, self.increments)
        self.init_solution = 2.3
        self.sample_size = len(self.inputs)

        print("%s %s" %(self._name, self.equation))
        print("Optimal value : %f  @ %f" %(self.x_optima, self.cost(self.x_optima)))
        print(_guide_line)

    #----------------------------------------------------
    # Cost Function
    #----------------------------------------------------
    def PWP_objective_function(self, sol):
        return 0.125 * pow(sol, 2) + 2.0 * sin(self.band * sol) + 2.0

    def _objective_function(self, sol):
        return self.op_obj_func(sol)

    def _premitive_function(self, sol):
        return 0.125 * pow(sol, 2)

    def cost(self, sol, _weight=0):
        if self.args.algorithm == 2:
            # Quantum Annealing
            H_0     = self._premitive_function(sol)
            H_1     = self._objective_function(sol)
            _ret    = _weight * H_0 + (1.0 - _weight) * H_1
        else:
            # Simulated Annealing and Quantized Optimization
            _ret    = self._objective_function(sol)
        return _ret

    #----------------------------------------------------
    # Service Function
    #----------------------------------------------------
    def get_init_solution(self):
        return self.init_solution

    # It is only for plotting
    def get_sample_size(self):
        return self.sample_size

    def get_optima(self):
        return self.x_optima

    def get_range(self):
        return [self.r_min, self.r_max]

# TSP operation
class Search_Operation_PWP:
    def __init__(self, pC_search_method, c_func):
        # Parameters
        self.c_func             = c_func
        #self.sample_size       = len(coords)
        # Core Search Algorithm : For Simulated Annealing, it is the search_algorithm
        self.pC_search_method   = pC_search_method
        self.core_search        = pC_search_method.search_algorithm
        # Operation
        self.solution_history   = [pC_search_method.curr_solution]
        self.cost_list          = [pC_search_method.curr_cost]
        # Processing Information : [_final_update, _initial_cost, _min_cost, _improvement, self.cost_list]
        self.processing_info    = []

    def get_candidate(self, _curr_solution):
        [self.r_min, self.r_max] = self.c_func.get_range()
        candidate = random.uniform(self.r_min, self.r_max)
        return candidate

    # Main Operation Function : Don't modify IT !!!!
    def operation(self):
        while self.pC_search_method.get_stop_condition():
            candidate = self.get_candidate(self.pC_search_method.curr_solution)

            self.core_search(candidate)
            self.pC_search_method.post_processing()

            self.cost_list.append(self.pC_search_method.get_current_cost())
            #self.cost_list.append(self.pC_search_method.get_min_cost())

            self.solution_history.append(self.pC_search_method.get_current_solution())

        self.print_final_result()

    # -----------------------------------------------------------------
    # common Service Function
    # -----------------------------------------------------------------
    def get_processing_result(self):
        return self.processing_info

    def print_final_result(self):
        _initial_cost           = self.pC_search_method.get_initial_cost()
        _final_update, _min_cost= self.pC_search_method.get_final_result()
        _best_solution          = self.pC_search_method.get_best_solution()

        if self.pC_search_method.args.algorithm == 2:
            # Quantum Annealing
            _min_cost       = self.pC_search_method.cost(_best_solution)
        else:
            # Simulated Annealing and Quantized optimization
            pass
        _improvement    = round((_initial_cost - _min_cost) / (_initial_cost), 4) * 100

        self.processing_info.append(_final_update)
        self.processing_info.append(_initial_cost)
        self.processing_info.append(_min_cost)
        self.processing_info.append(_improvement)
        self.processing_info.append(self.cost_list)

        print(_guide_line)
        print("Initial cost   : ", _initial_cost)
        print('Minimum cost   : ', _min_cost)
        print('Improvement    : ', _improvement, '%')
        print("Final Iteration:  %d" %(_final_update))
        print("Solution (x =  : ", _best_solution)
        print("Cost at Solution ", self.pC_search_method.cost(_best_solution))

    def animateSolutions(self, _active):
        return 0

    def plotLearning(self):
        _initial_cost   = self.pC_search_method.get_initial_cost()
        _min_cost       = self.pC_search_method.get_min_cost()

        plt.plot([i for i in range(len(self.cost_list))], self.cost_list)
        line_init = plt.axhline(y=_initial_cost, color='r', linestyle='--')
        line_min = plt.axhline(y=_min_cost, color='g', linestyle='--')
        plt.legend([line_init, line_min], ['Initial cost', 'Optimized cost'])
        plt.ylabel('Cost')
        plt.xlabel('Iteration')
        plt.show()

#============================================================================
# Test Function 2 : tsp function
#============================================================================
class tsp_op:
    def __init__(self, coords, args):
        # Function
        self.coords             = coords
        self.sample_size        = len(coords)
        self.dist_matrix        = tsp_utils.vectorToDistMatrix(coords)
        self.init_solution      = tsp_utils.nearestNeighbourSolution(args, self.dist_matrix)

        self.args               = args

    # Calcuate cost
    def cost(self, sol, _weight=0):
        return sum([self.dist_matrix[i, j] for i, j in zip(sol, sol[1:] + [sol[0]])])

    def get_init_solution(self):
        return self.init_solution

    def get_sample_size(self):
        return self.sample_size


# TSP operation
class Search_Operation:
    def __init__(self, pC_search_method, coords):
        # Parameters
        self.coords             = coords
        self.sample_size        = len(coords)
        # Core Search Algorithm : For Simulated Annealing, it is the search_algorithm
        self.pC_search_method   = pC_search_method
        self.core_search        = pC_search_method.search_algorithm
        # Operation
        self.solution_history   = [pC_search_method.curr_solution]
        self.cost_list          = [pC_search_method.curr_cost]
        self.best_sol_history   = [pC_search_method.curr_solution]
        # Processing Information : [_final_update, _initial_cost, _min_cost, _improvement, self.cost_list]
        self.processing_info    = []

    def get_candidate(self, _curr_solution):
        candidate = list(_curr_solution)
        l = random.randint(2, self.sample_size - 1)
        i = random.randint(0, self.sample_size - l)

        candidate[i: (i + l)] = reversed(candidate[i: (i + l)])
        return candidate

    # Main Operation Function : Don't modify IT !!!!
    def operation(self):
        while self.pC_search_method.get_stop_condition():
            candidate = self.get_candidate(self.pC_search_method.curr_solution)

            self.core_search(candidate)
            self.pC_search_method.post_processing()

            self.cost_list.append(self.pC_search_method.get_current_cost())
            self.solution_history.append(self.pC_search_method.get_current_solution())
            self.best_sol_history.append(self.pC_search_method.get_best_solution())

        self.print_final_result()
    # -----------------------------------------------------------------
    # common Service Function
    # -----------------------------------------------------------------
    def get_processing_result(self):
        return self.processing_info

    def get_solution_history(self):
        return self.solution_history

    def get_best_sol_history(self):
        return self.best_sol_history

    def print_final_result(self):
        _initial_cost           = self.pC_search_method.get_initial_cost()
        _final_update, _min_cost= self.pC_search_method.get_final_result()
        _improvement            = round((_initial_cost - _min_cost) / (_initial_cost), 4) * 100
        self.processing_info.append(_final_update)
        self.processing_info.append(_initial_cost)
        self.processing_info.append(_min_cost)
        self.processing_info.append(_improvement)
        self.processing_info.append(self.cost_list)

        print(_guide_line)
        print("Initial cost   : ", _initial_cost)
        print('Minimum cost   : ', _min_cost)
        print('Improvement    : ', _improvement, '%')
        print("Final Iteration:  %d" %(_final_update))

    def animateSolutions(self, _active):
        if _active:
            animated_visualizer.animateTSP(self.solution_history, self.coords)
        else:
            pass

    def plotLearning(self):
        _initial_cost   = self.pC_search_method.get_initial_cost()
        _min_cost       = self.pC_search_method.get_min_cost()

        plt.plot([i for i in range(len(self.cost_list))], self.cost_list)
        line_init = plt.axhline(y=_initial_cost, color='r', linestyle='--')
        line_min = plt.axhline(y=_min_cost, color='g', linestyle='--')
        plt.legend([line_init, line_min], ['Initial cost', 'Optimized cost'])
        plt.ylabel('Cost')
        plt.xlabel('Iteration')
        plt.show()


