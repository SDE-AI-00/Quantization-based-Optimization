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
# Regard Warning as "error". That enables to use try ~ except
import warnings
warnings.filterwarnings("error")

class GeneticAlgorithm_proc:
    def __init__(self, pf_cost, _init_solution, args):
        # ------- Mandatory Parameters -------
        # Solution Parameter
        self.curr_solution  = _init_solution
        self.best_solution  = self.curr_solution
        #cost Parameter
        self.cost           = pf_cost
        self.initial_cost   = self.cost(self.curr_solution)
        self.curr_cost      = self.initial_cost
        self.min_cost       = self.curr_cost
        #Operating Parameters
        self.stopping_iter  = args.stopping_iter
        self.stopping_temp  = args.stopping_temp
        self.iteration      = 1
        self.args           = args
        self.final_update   = 0

        # ------- GA Parameters -------
        self.n_population   = self.args.genetic_population
        self.mutation_rate  = 0.3
        self.chromosome     = [self.curr_solution for _k in range(self.n_population)]
        self.chromosome_buf = []
        self.parentID       = []
        # Crossover and Mutation method
        self.crossover      = Partially_Mapped_Crossover()
        self.mutation       = TSP_Mutation()
        # ------- Annealing Parameters -------
        self.temp           = args.temp
        self.alpha          = args.alpha

        # Information
        #print("===================================================")
        print("Genetic Algorithm Specification")
        print("Population : ", self.n_population)
        print("===================================================")
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
            _msg = "[%6d ] Current Cost: %6.2f min_cost: %6.2f" % (
            self.iteration, self.curr_cost, self.min_cost)
            print(_msg)
        else: pass
            #print('.', end='')
    # -----------------------------------------------------------------
    # Genetic Algorithm Core code
    # -----------------------------------------------------------------
    def acceptance_probability(self, candidate_cost):
        return math.exp(-abs(candidate_cost - self.curr_cost) / self.temp)
    # Compute the fitness of Chromosome maximum fintness (approx 1) is best
    def get_fitness(self):
        _cost = []
        for _k, _data in enumerate(self.chromosome):
            _cost.append(math.exp(-1.0 * self.cost(_data)))
        n_cost  = np.array(_cost)

        try:
            n_fit   = n_cost/(np.sum(n_cost) + 0.000000000001)
        except RuntimeWarning:
            print("RunTime Warning Arise !!!")
            print("Debug Point ")
            print("self.iteration : ", self.iteration)
            print("self.chromosome: ", self.chromosome)
            print("n_cost         : ", n_cost)

        return n_fit
    # Get the index of parents depending on the disk derived by fitness
    # If the region to a chromosome in the disk is wider than others,
    # the acceptance probability increase according to the region.
    def get_parent_idx(self, _disk):
        _idx = 0
        _rand = np.random.rand()
        for _k, _value in enumerate(_disk):
            if _k == 0:
                if 0 <= _rand and _rand < _value:
                    _idx = _k
                    break
                else: pass
            else:
                if _disk[_k-1] <= _rand and _rand < _value:
                    _idx = _k
                    break
                else:
                    pass
        return _idx

    def make_parents(self, _disk):
        # make Parents
        self.parentID = []
        _count, _stable = 0, True
        self.parentID.append(self.get_parent_idx(_disk))
        while(_stable):
            t_idx = self.get_parent_idx(_disk)
            if t_idx != self.parentID[-1]:
                self.parentID.append(t_idx)
                break
            else:
                _count += 1
            _stable = (_count <= 10)
        # exception processing :
        # if two index is equal, we should choose a different parent index.
        while(not _stable):
            t_idx   = np.random.randint(1, self.n_population)
            t_idx   = (self.parentID[0] + t_idx) % self.n_population
            _stable = (t_idx != self.parentID[0])
            if _stable and len(self.parentID) == 1:
                self.parentID.append(t_idx)
            else:
                print("Stop Processing in MakeParents Line 153")
                print("Debug Information ")
                print("t_idx         : ", t_idx)
                print("_stable       : ", "True" if _stable else "False")
                print("self.parentID : ", self.parentID)
                exit(0)
        # Temporal reserve the parents
        self.chromosome_buf.append(self.chromosome[self.parentID[0]])
        self.chromosome_buf.append(self.chromosome[self.parentID[1]])
        return [self.chromosome[self.parentID[0]], self.chromosome[self.parentID[1]]]

    def restore_parent(self):
        #self.chromosome[self.parentID[0]] = self.chromosome_buf[0]
        #self.chromosome[self.parentID[1]] = self.chromosome_buf[1]
        self.chromosome_buf = []

    def Gnetic_Algorithm_Process(self):
        # Make Roulette disk
        _fitness = self.get_fitness()
        _disk = np.add.accumulate(_fitness)
        #make parents
        _parents = self.make_parents(_disk)
        # Crossover and Mutation
        _child = self.crossover(_parents)
        #print("Crossover (Child) : ", _child[0], _child[1])
        _child[0] = self.mutation(_child[0])
        _child[1] = self.mutation(_child[1])
        #print("Mutation          : ", _child[0], _child[1])
        # Replace Parents
        self.chromosome[self.parentID[0]] = list(_child[0])
        self.chromosome[self.parentID[1]] = list(_child[1])
        # Get best chromosome as candidate and candidate_cost
        _fitness        = self.get_fitness()
        _max_index      = np.argmax(_fitness)
        _candidate      = self.chromosome[_max_index]
        _candidate_cost = self.cost(_candidate)
        #print("maxIndex : ", _max_index, "_candidate ", _candidate, " _candidate_cost", _candidate_cost)
        return _candidate, _candidate_cost

    def search_algorithm(self, candidate):
        #candidate_cost = self.cost(candidate)
        candidate, candidate_cost = self.Gnetic_Algorithm_Process()

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
            else:
                self.restore_parent()
        # Debug
        print("[%6d ] \r" %self.iteration, end='')

    def post_processing(self):
        self.temp *= self.alpha
        self.iteration += 1

# =================================================================
# Partially_mapped_crossover
# =================================================================
class   Partially_Mapped_Crossover:
    def __init__(self):
        self.parent1             = None
        self.parent2             = None
        self.firstCrossPoint     = None
        self.secondCrossPoint    = None
        #self.firstCrossPoint    = 1
        #self.secondCrossPoint   = 4
        self.parent1MiddleCross = None
        self.parent2MiddleCross = None
        self.relations          = None

    def settings(self, parents):
        self.parent1             = parents[0]
        self.parent2             = parents[1]
        self.firstCrossPoint     = np.random.randint(0,len(self.parent1)-2)
        self.secondCrossPoint    = np.random.randint(self.firstCrossPoint+1,len(self.parent1)-1)
        #self.firstCrossPoint    = 1
        #self.secondCrossPoint   = 4
        self.parent1MiddleCross = self.parent1[self.firstCrossPoint:self.secondCrossPoint]
        self.parent2MiddleCross = self.parent2[self.firstCrossPoint:self.secondCrossPoint]
        self.temp_child1 = self.parent1[:self.firstCrossPoint] + self.parent2MiddleCross + self.parent1[self.secondCrossPoint:]
        self.temp_child2 = self.parent2[:self.firstCrossPoint] + self.parent1MiddleCross + self.parent2[self.secondCrossPoint:]

        self.relations = []
        for i in range(len(self.parent1MiddleCross)):
            self.relations.append([self.parent2MiddleCross[i], self.parent1MiddleCross[i]])
        #print("Relations   :", self.relations)

    def recursion(self, temp_child, _type=1):
        _child = np.array([0 for i in range(len(self.parent1))])
        _child_param= self.parent2MiddleCross if _type==1 else self.parent1MiddleCross
        param_id    = 0 if _type==1 else 1
        inv_param_id= 1 if _type==1 else 0

        for i,j in enumerate(temp_child[:self.firstCrossPoint]):
            c=0
            for x in self.relations:
                if j == x[param_id]:
                    _child[i]=x[inv_param_id]
                    c=1
                    break
            if c==0:
                _child[i]=j

        j=0
        for i in range(self.firstCrossPoint,self.secondCrossPoint):
            _child[i]=_child_param[j]
            j+=1

        for i,j in enumerate(temp_child[self.secondCrossPoint:]):
            c=0
            for x in self.relations:
                if j == x[param_id]:
                    _child[i+self.secondCrossPoint]=x[inv_param_id]
                    c=1
                    break
            if c==0:
                _child[i+self.secondCrossPoint]=j
        child_unique=np.unique(_child)
        if len(_child)>len(child_unique):
            _child=self.recursion(_child, _type=_type)
        return(_child)

    def __call__(self, parents):
        self.settings(parents)
        child1 = self.recursion(self.temp_child1, _type=1)
        child2 = self.recursion(self.temp_child2, _type=2)

        return [child1, child2]

class TSP_Mutation:
    def __init__(self):
        self.temp_value = [0, 0]

    def __call__(self, _data):
        _firstCrossPoint    = np.random.randint(0,len(_data)-2)
        _secondCrossPoint   = np.random.randint(_firstCrossPoint+1,len(_data)-1)
        self.temp_value     = [_data[_firstCrossPoint], _data[_secondCrossPoint]]
        _data[_firstCrossPoint]  = self.temp_value[1]
        _data[_secondCrossPoint] = self.temp_value[0]
        return _data

# =================================================================
# Main Routine
# =================================================================
if __name__ == "__main__":
    parent1     = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    parent2     = [5, 4, 6, 7, 2, 1, 3, 9, 8]
    parents     = [parent1, parent2]
    Crossover   = Partially_Mapped_Crossover()
    Mutation    = TSP_Mutation()

    child = Crossover(parents)
    print(child[0])
    print(child[1])

    for _k, _data in enumerate(child):
        child[_k] = Mutation(_data)
        print(child[_k])
