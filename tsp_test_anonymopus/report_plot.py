#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# TSP Simulating 
###########################################################################
_description = '''\
====================================================
report_plot.py : Plotting the result reading result_multi.pkl
====================================================
Example : report_plot.py 
'''
import matplotlib.pyplot as plt
import pickle
import my_debug as DBG
class _report:
    def __init__(self, _filename="[ref]_result_multi.pkl", start_pt= 0,):
        self.filename   = _filename

        try:
            with open(self.filename, 'rb') as _file:
                self.data = pickle.load(_file)
        except Exception as e:
            DBG.dbg("Error Occur!! : ", e)
            exit(0)

        self.cost_list  = []
        self._ini_cost  = []
        self._min_cost  = []
        self._avail_algo= 0
        self._start_pt  = start_pt
        self.parsing_data()

    def parsing_data(self):
        for _d, _dummy_data in enumerate(self.data):
            for _a, _algo_data in enumerate(_dummy_data):
                if len(_algo_data) > 0:
                    self._ini_cost.append(_algo_data[1])
                    self._min_cost.append(_algo_data[2])
                    self.cost_list.append(_algo_data[4])
                    self._avail_algo += 1
                else: pass

    def get_initial_cost(self, _algo_id):
        return self._ini_cost[_algo_id]

    def get_min_cost(self, _algo_id):
        return self._min_cost[_algo_id]

    def get_cost_list(self, _algo_id):
        return self.cost_list[_algo_id]

    def plotLearning(self):
        _min_cost       = 9999999999999999999.9
        _min_data_len   = 99999999999999999
        _initial_cost   = None
        l_labels        = ['Simulated Annealing', 'Quantization-Based Optimization', 'Quantum Annealing']
        std_cost_data   = []

        for _algo_id in range(self._avail_algo):
            _initial_cost   = self.get_initial_cost(_algo_id)
            _cost_list      = self.get_cost_list(_algo_id)
            if _algo_id == 0:
                _min_cost   = self.get_min_cost(_algo_id)
            else:
                t_min_cost  = self.get_min_cost(_algo_id)
                if t_min_cost < _min_cost:
                    _min_cost = t_min_cost
                else: pass

            std_cost_data.append(_cost_list)
            if len(_cost_list) < _min_data_len:
                _min_data_len = len(_cost_list)
            else: pass

            #plt.plot([i for i in range(len(_cost_list))], _cost_list, label=l_labels[_algo_id])
            #line_init = plt.axhline(y=_initial_cost, color='r', linestyle='--', label='Initial cost')

        # Comparison to fastest convergence algorithm
        for _algo_id, _cost_list in enumerate(std_cost_data):
            plt.plot([i for i in range(self._start_pt, _min_data_len)], _cost_list[self._start_pt:_min_data_len], label=l_labels[_algo_id])

        plt.axhline(y=_initial_cost, color='r', linestyle='--', label='Initial cost')
        line_min = plt.axhline(y=_min_cost, color='g', linestyle='--', label='Optimized cost')
        #plt.legend([line_init, line_min], ['Initial cost', 'Optimized cost'])
        plt.legend()
        plt.ylabel('Cost')
        plt.xlabel('Iteration')
        plt.show()

        #---------------------------------------------------------
        # Second Phase
        # ---------------------------------------------------------
        self.plot_error_trend_per_algorithm(std_cost_data, _min_data_len, _initial_cost, l_labels)


    def plot_error_trend_per_algorithm(self, std_cost_data, _min_data_len, _initial_cost, l_labels):
        for _algo_id, _cost_list in enumerate(std_cost_data):
            plt.plot([i for i in range(self._start_pt, _min_data_len)], _cost_list[self._start_pt:_min_data_len], label=l_labels[_algo_id])

            plt.axhline(y=_initial_cost, color='r', linestyle='--', label='Initial cost')
            #line_min = plt.axhline(y=_min_cost, color='g', linestyle='--', label='Optimized cost')
            #plt.legend([line_init, line_min], ['Initial cost', 'Optimized cost'])
            plt.legend()
            plt.ylabel('Cost')
            plt.xlabel('Iteration')
            plt.show()


# =================================================================
# Main Routine
# =================================================================
if __name__ == "__main__":
    #_operation_param= {'start_pt': 30000}
    _operation_param = {'start_pt': 0}

    c_report = _report(_filename="result_multi.pkl", start_pt=_operation_param['start_pt'])
    #c_report = _report()
    c_report.plotLearning()

    print("===================================================")
    print("Process Finished ")
    print("Plotting the result of TSP Simulation              ")
    print("===================================================")