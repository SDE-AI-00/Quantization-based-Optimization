#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################
# nlp_operation.py
# Operation Classes for any styles of NLP test
# Example code is test_nlp.py and read README.md for appropriate usage
###########################################################
_description = '''\
====================================================
test_nlp02.py : Test codes as like BAT or shell script
                using nlp_operation.py 
                    Written by Jinwuk @ 2022-01-26
====================================================
example : test_nlp02.py  
'''
from nlp_operation import Service_class, Test_Class

class Worker_class:
    def __init__(self, _TCobj, _init_pattern_id=1, _store_full_info= False):
        self.TC                 = _TCobj
        self._init_pattern_id   = _init_pattern_id
        self._store_full_info   = _store_full_info
        self.trace_data         = []

    def main_Proc(self, _num_attemp, _q_value, _func_id, _dimension, _additional_args=''):
        # Set Benchmark Function
        self.TC._Test_function.append(_func_id)
        # -----------------------------------------------------
        # Benchmark Test : 1 using random Initial Point (1)
        # -----------------------------------------------------
        self.TC.Set_Argument_String(_init_pattern_id=self._init_pattern_id)
        self.TC.Set_Control_Parameter(_num_attemp, _quantize=False, _appendinit=False)
        self.TC.Main_Processing()
        self.trace_data.append(self.TC.AnalyzeObj.Get_trace_data())
        # -----------------------------------------------------
        # Benchmark Test : 2 using same Initial Point (4)
        # -----------------------------------------------------
        self.TC.Set_Argument_String(_init_pattern_id=4, _qparam=_q_value)
        self.TC.Set_Control_Parameter(_num_attemp, _quantize=True, _appendinit=False)
        self.TC.Main_Processing()
        self.trace_data.append(self.TC.AnalyzeObj.Get_trace_data())
        # -----------------------------------------------------
        # Final Processing
        # -----------------------------------------------------
        _Function_Name   = self.TC.TestObj.Get_Test_Function_Name(_func_id)
        _Report_FileName = "Report_NLP_" + _Function_Name + "-d%d" %_dimension + ".txt"
        _Result_FileName = "Full_Result_NLP_" + _Function_Name + "-d%d" %_dimension + ".txt"

        self.TC.Generate_Report(_Report_FileName, _Result_FileName, _q_value)

        # Plotting
        _c_ns = self.TC.TestObj.Get_Service_Class_for_NLP()
        self.TC.Generate_plot(c_ns=_c_ns, _trace_data= self.trace_data, _func_id=_func_id, _proc=False)

        # Function Draw
        self.TC.draw_objective_function(c_ns=_c_ns, _func_id=_func_id, _proc=True)

# =================================================================
# Test Processing
# ex: -a 1 -s 1 -i -1.232 -1.212 -t 4000 -em 0 -qt 1 -q 4 -qm 2
# =================================================================
if __name__ == "__main__":
    # Set fumdamental parameters
    _num_attemp = 1
    _q_value    = 5
    _func_id    = 9
    _dimension  = 2
    _additional_args = '-em 2 -i -1.155 -1.255 '
    _init_pattern_id = 4
    
    # Make Operation Arguments
    SC = Service_class(_dimension, _additional_args)
    _init_pattern = SC.get_init_pattern()
    
    # Global Definition
    TC      = Test_Class(_init_pattern)
    cWorker = Worker_class(TC, _init_pattern_id=_init_pattern_id)

    cWorker.main_Proc(_num_attemp, _q_value, _func_id, _dimension)


    print("======================================================")
    print("Process Finished")
    print("======================================================")
