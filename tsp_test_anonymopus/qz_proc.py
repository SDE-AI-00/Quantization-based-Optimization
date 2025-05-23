#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# TSP Simulating 
###########################################################################
_description = '''\
====================================================
qz_proc.py : Classical Simulated Annealing Test
====================================================
Example : No operation. Just provide class and functions 
'''
import math
import random
import numpy as np

class Quantization_proc:
    def __init__(self, pf_cost, _init_solution, args):
        # Solution Parameter, Here is a quantized value
        self.curr_solution  = _init_solution
        self.best_solution  = self.curr_solution
        #cost Parameter
        self.cost           = pf_cost
        self.real_init_cost = self.cost(self.curr_solution)
        self.min_cost       = self.real_init_cost

        # Algorithm Parameter
        self.base           = 2
        self.power_idx      = 0
        self.eta            = self.compute_eta(self.real_init_cost)
        self.QP             = self.eta * math.pow(self.base, self.power_idx)

        self.initial_cost   = self.quantization(self.real_init_cost)
        self.curr_cost      = self.initial_cost
        self.stopping_iter  = args.stopping_iter
        self.iteration      = 1
        self.limit_power_idx= 1023
        #Operating Parameters
        self.args           = args

        # Information
        print("===================================================")
        print(" QP         : ", self.QP)
        print(" Inverse QP : ", 1.0/self.QP)
        print(" Intial cost: %.2f  Quantized Initial: %.2f" %(self.real_init_cost, self.curr_cost))
        print("===================================================")
    # -----------------------------------------------------------------
    # Common IO Functions
    # -----------------------------------------------------------------
    def get_best_solution(self):
        return self.best_solution

    def get_current_cost(self):
        return self.curr_cost

    def get_initial_cost(self):
        self.initial_cost = self.real_init_cost
        return self.initial_cost

    def get_min_cost(self):
        return self.min_cost

    def get_current_solution(self):
        return self.curr_solution

    def get_stop_condition(self):
        _ret = self.power_idx < self.limit_power_idx and self.iteration < self.stopping_iter
        return _ret

    def get_final_result(self):
        return self.final_update, self.min_cost
    # -----------------------------------------------------------------
    # Debug function
    # -----------------------------------------------------------------
    def print_line_information(self):
        # For debug
        if self.args.debug_message == 1:
            _msg = "[%6d ] QP_h: %6.2f fQ: %6.2f min_cost: %6.2f" % (
            self.iteration, self.power_idx, self.curr_cost, self.min_cost)
            print(_msg)
        else: pass
            #print('.', end='')

    # -----------------------------------------------------------------
    # Core Functions of Quantized Optimization
    # -----------------------------------------------------------------
    def compute_eta(self, _initial_cost):
        _log        = math.log(_initial_cost, self.base)
        _ceil_root  = _log + 1.0
        _ceil_value = -1.0 * math.floor(_ceil_root)
        _ret        = math.pow(self.base, _ceil_value)
        return _ret

    def increase_quantization_resolution(self):
        self.power_idx += 1

    def compute_quantizaztion_parameter(self):
        try:
            self.QP = self.eta * math.pow(self.base, self.power_idx)
            #self.QP = self.eta * np.power(self.base, self.power_idx)
        except:
            print("Error Occur!!")
            print("self.power_idx : %d" %self.power_idx)
            print("self.QP        : ", self.QP)
            #exit()

    def quantization(self, _data):
        _root   = _data + 0.5 * 1.0/self.QP
        _floor  = self.QP * _root
        _deno   = math.floor(_floor)
        _ret    = 1.0 * _deno/(1.0 * self.QP)
        return _ret

    def search_algorithm(self, candidate):
        candidate_cost = self.cost(candidate)
        quantized_cost = self.quantization(candidate_cost)
        if quantized_cost <= self.curr_cost:
            self.curr_solution = candidate          # x_opt <- x_t
            self.increase_quantization_resolution()
            self.compute_quantizaztion_parameter()
            self.curr_cost = self.quantization(candidate_cost)
            if candidate_cost < self.min_cost:
                self.min_cost       = candidate_cost
                self.best_solution  = candidate
                self.final_update   = self.iteration
                # For debug
                self.print_line_information()
            else: pass
        else:
            pass

    def post_processing(self):
        self.iteration += 1

