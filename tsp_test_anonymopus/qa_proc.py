#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# TSP Simulating 
###########################################################################
_description = '''\
====================================================
qa_proc.py : Classical Simulated Annealing Test
====================================================
Example : No operation. Just provide class and functions 
'''

import math
import random

class QuantumAnnealing_proc:
    def __init__(self, pf_cost, _init_solution, args):
        # Algorithm Parameter
        self.temp           = args.temp
        self.alpha          = args.alpha
        self.stopping_temp  = args.stopping_temp
        self.stopping_iter  = args.stopping_iter
        self.iteration      = 1
        self.final_update   = 0
        # Solution Parameter
        self.curr_solution  = _init_solution
        self.best_solution  = self.curr_solution
        #cost Parameter
        self.cost           = pf_cost
        self.initial_cost   = self.cost(self.curr_solution)
        self.curr_cost      = self.initial_cost
        self.min_cost       = self.curr_cost
        #Operating Parameters
        self.args           = args

        #print('Intial cost: ', self.curr_cost)

    # -----------------------------------------------------------------
    # Common IO Functions
    # -----------------------------------------------------------------
    def get_best_solution(self):
        return self.best_solution

    def get_current_cost(self):
        return self.curr_cost

    def get_initial_cost(self):
        return self.initial_cost

    def get_min_cost(self):
        return self.min_cost

    def get_current_solution(self):
        return self.curr_solution

    def get_stop_condition(self):
        _ret = self.temp >= self.stopping_temp and self.iteration < self.stopping_iter
        return _ret

    def get_final_result(self):
        return self.final_update, self.min_cost

    # -----------------------------------------------------------------
    # Debug function
    # -----------------------------------------------------------------
    def print_line_information(self):
        # For debug
        if self.args.debug_message == 1:
            _msg = "[%6d ] temp: %6f Current Cost: %6.2f min_cost: %6.2f" % (
            self.iteration, self.temp, self.curr_cost, self.min_cost)
            print(_msg)
        else: pass
            #print('.', end='')

    # -----------------------------------------------------------------
    # Quantum Annealing Core algorithm
    # -----------------------------------------------------------------
    def time_weight_for_H0(self):
        _t      = 1.0 * self.iteration
        _tf     = 1.0 * self.stopping_iter
        _ret_H1 = math.sqrt(_t/_tf)
        _ret_H0 = 1.0 - _ret_H1
        return _ret_H0, _ret_H1

    def qa_cost(self, candidate):
        _weight_H0,_ = self.time_weight_for_H0()
        _ret    = self.cost(candidate, _weight=_weight_H0)
        return _ret

    def acceptance_probability(self, candidate_cost):
        return math.exp(-abs(candidate_cost - self.curr_cost) / self.temp)

    def search_algorithm(self, candidate):
        candidate_cost = self.qa_cost(candidate)
        if candidate_cost < self.curr_cost:
            self.curr_cost      = candidate_cost
            self.curr_solution  = candidate
            if candidate_cost < self.min_cost:
                self.min_cost       = candidate_cost
                self.best_solution  = candidate
                self.final_update   = self.iteration
                # For debug
                self.print_line_information()
        else:
            if random.random() < self.acceptance_probability(candidate_cost):
                self.curr_cost = candidate_cost
                self.curr_solution = candidate

    def post_processing(self):
        self.temp *= self.alpha
        self.iteration += 1