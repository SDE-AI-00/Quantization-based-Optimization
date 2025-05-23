#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################
# nlp_operation.py
# Operation Classes for any styles of NLP test
# Example code is test_nlp.py and read README.md for appropriate usage
###########################################################
_description = '''\
====================================================
nlp_operation.py : Operation Classes for any styles of NLP test
====================================================
example : No operation codes 
'''
from nlp_main01 import NLP_Main 
import my_debug as DBG
import sys

# =================================================================
# Global Definiton
# =================================================================
small_hLine = "-----------------------------------------------------"
large_hline = "====================================================="
chk_hline   = "================================================================="

# =================================================================
# Class 1 : Bechmark_Test
# =================================================================
class Bechmark_Test:

    def __init__(self, _param_str, _ctr_list):
        self._num_attemp    = _ctr_list[0]    # 시도 횟수 
        self._quantize      = _ctr_list[1]
        self._appnedinit    = _ctr_list[2]
        
        self._init_str      = _param_str.pop() 
        self._qunt_str      = _param_str.pop()
        self._base_str      = _param_str.pop()
        
        self._res_func      = []
        self._init_points   = []

        self.test_function_name = self.Get_Benchmark_Function_Name()
        self.num_test_functions = len(self.test_function_name)
        self._test_function_idx = []    # Real test function index 
    
        # NLP Object : It is changable to each iteration 
        self.NLPobj         = None

        # To transfer a nlp service class
        self.c_ns           = None
    # -----------------------------------------------------
    # Interface Service Function
    # -----------------------------------------------------
    def Get_Result(self):
        return self._res_func
    
    # -----------------------------------------------------
    # Service Functions 
    # -----------------------------------------------------
    def print_benchmark_functions(self):
        # First, get the NLP_Main Object. Initialization can be done as much as you want.
        self.NLPobj = NLP_Main()    
        _LBf_name   = self.NLPobj.cFunctions.Get_All_Function_Name() 
        _msg        = small_hLine + "\n" + "Name of Bechmark Functions \n" + small_hLine + "\n"
        
        print(_msg); 
        for _k, _fname in enumerate(_LBf_name):
            print("[%4d] %s" %(_k, _fname))
        
        # Since the work is finished, return
        self.NLPobj = None 

    def Change_arg(self, _arg_str):
        _param_str  = _arg_str + (self._qunt_str if self._quantize else '')
        _param_list = _param_str.split()

        _basis = sys.argv[0]
        sys.argv.clear()
        sys.argv.append(_basis)
        for _k in _param_list:
            sys.argv.append(_k)

        print(large_hline)
        DBG.dbg("Argument : %s" %" ".join(sys.argv))
        print(small_hLine)


    def Analysis_info(self, _info, _sub_res):
        for _k, _msg in enumerate(_info):
            _lines = _msg.split('\n')
            for _k, _line in enumerate(_lines): 
                _word = _line.split(' ')
                if "|" in _word:
                    _sub_res[0] = self.method_1(_word, _sub_res[0])
                elif "Initial" in _word:
                    _sub_res[1] = self.method_2(_word, _sub_res[1])
                elif "step:" in _word:
                    _sub_res[2] = self.method_3(_word, _sub_res[2])
                else:
                    pass   
        return _sub_res

    def method_1(self, _word, _sub_res):
        if "step" in _word :
            pass
        else:
            _word = self.remove_data_in_List(_word, '')
            _word = self.remove_data_in_List(_word, '|')
            _sub_res.append(_word)
        return _sub_res

    def method_2(self, _word, _sub_res):
        if "step" in _word :
            pass
        else:
            Initial_point =  self.NLPobj.Get_Init_point()
            _sub_res.append(Initial_point)
        return _sub_res

    # extract_stepinfo
    def method_3(self, _word, _sub_res):
        if "Best" in _word:
            pass
        else:
            _info_idx   = _word.index('X')
            _raw_info   = _word[_info_idx+2:]
            _vector     = []
            for _k, _data in enumerate(_raw_info):
                _value = _data.strip("[""]")
                _vector.append(_value)
            np_stepinfo = np.array(_vector)
            _sub_res.append(np_stepinfo)
        return _sub_res

    def remove_data_in_List(self, _List, _data):
        for _param in _List:
            if _param == _data :
                _List.remove(_param) 
        return _List

    def Get_Benchmark_Function_Name(self):
        _title_line = "\n" + large_hline + "\n" + "Benchmark Functions \n" + small_hLine
        NLP = NLP_Main()

        print(_title_line)
        _function_names = NLP.cFunctions.Get_All_Function_Name()
        for _k, _fname in enumerate(_function_names):
            print("[%2d] %s" %(_k, _fname))
        print(small_hLine)

        return _function_names

    def Get_Benchmnark_function_List(self, _function_index):
        _confirm_msg = small_hLine + "\n"
        for _k, _idx in enumerate(_function_index):
            idx_benchmark_function = self.test_function_name[_idx]
            self._test_function_idx.append(_idx)
            _confirm_msg += ("[%2d] " %_idx) + idx_benchmark_function + "\n" 
        _confirm_msg += small_hLine
        print(_confirm_msg)

    def Get_Initial_Point(self, _init_str, _algorithm_id, _attempt_id ):
        if self._appnedinit:
            _init_vec = self._init_points[_algorithm_id]    
            _res ='-i '
            for _components in _init_vec[_attempt_id]:
                _res += str(_components) + ' '
        else:
            _res =''
        return _init_str + _res            

    def Get_Test_Function_Name(self, _ID):
        return self.test_function_name[_ID]

    def Get_Service_Class_for_NLP(self):
        return self.c_ns

    # -----------------------------------------------------
    # Main Processing Functions 
    # -----------------------------------------------------
    def __call__(self, init_point_info):
        # For development 
        self._init_points   = init_point_info
        _base_str           = self._base_str 

        # Result per Function
        self._res_func.clear()
        for _f in self._test_function_idx:
            _func_str = "-f " + str(_f) + ' '
        
            # Result per Algorithm: Since the value should not change, new Memory must be allocated.
            _res_algo = []
            for _k in range(0, 4, 1):
                _algo_str   = "-a " + str(_k) + ' '

                # Result according to the number of attempts result, init, step info
                _sub_res = [[], [], []]
                for _i in range(self._num_attemp):        
                    # In the case where the given init_point changes according to the number of attempts
                    _init_str = self.Get_Initial_Point(self._init_str, _algorithm_id=_k, _attempt_id=_i)
                    
                    # Initialization again because the argument is changed
                    _param_str  = _algo_str + _base_str + _func_str + _init_str
                    self.Change_arg(_param_str)
                    self.NLPobj = NLP_Main()
                    _outinfo, _args = self.NLPobj.main_processing()
                    # List of results according to the number of attempts
                    _sub_res = self.Analysis_info(_outinfo, _sub_res)

                # Organize results by algorithm
                _res_algo.append(_sub_res)
        
            # Results sorted by benchmark function
            self._res_func.append(_res_algo)

            # NLP Service object setting
            self.c_ns = self.NLPobj.Get_nlp_service()
# =================================================================
# Class 2 : Analysis_Test_Results
# =================================================================
import copy
import numpy as np
class Analysis_Test_Results:
    def __init__(self, _result):
        self.threshold_cost  = 0.01 
        self._num_attemp     = 0
        self._data           = copy.deepcopy(_result)
        self._success_info   = []
        self._init_vec       = []  

        self._File_msg       = []
        self.result_msg      = []
        self.trace_data      = []
    # -----------------------------------------------------------------
    # Data Structure : for self._data and self.trace_data
    #    Function
    #        |___ Algorithm
    #                |____ Attempt (result, init, step info)
    # -----------------------------------------------------------------
    def __call__(self, _ctr_list):
        self._num_attemp    = _ctr_list[0]
        self.lprint(large_hline)

        _aux_trace_alogo = []
        for _k, _algo_data in enumerate(self._data):
            _title_str = "Function for : %d \n" %_k + small_hLine
            self.lprint(_title_str)

            _aux_trace_attemp = []
            for _i, _attemp_data in enumerate(_algo_data):
                _result_data = _attemp_data [0]
                _initia_data = _attemp_data [1]
                _trace_data  = _attemp_data [2]

                _title_str = "Algorithm : %d \n" %_i + small_hLine
                self.lprint(_title_str)

                _RST_data = self.Extract_Result_Data(_result_data)
                self.Analysis_Result(_RST_data)
                _init_vec = self.Extract_Initial_Point(_initia_data)
                self._init_vec.append(_init_vec)
                _aux_trace_attemp.append(_trace_data)

            _aux_trace_alogo.append(_aux_trace_attemp)

        self.trace_data = copy.deepcopy(_aux_trace_alogo)
        _title_str = large_hline + "\n"
        self.lprint(_title_str)

    # -----------------------------------------------------
    # Auxiliary Functions 
    # -----------------------------------------------------
    def Extract_Result_Data(self, _result_data):
        _sub_title = "Result \n" + small_hLine
        self.lprint(_sub_title)
        
        _new_result = []
        for _j, _result in enumerate(_result_data):
            _iteratio = int(_result[0])
            _costdiff = float(_result[1])
            _locadiff = float(_result[2])     
            self.lprint("[%2d] %6d  %12.8f  %12.8f" %(_j, _iteratio, _costdiff, _locadiff))
            _new_result.append([_iteratio, _costdiff, _locadiff])
                
        self.lprint(small_hLine)
        return _new_result


    def Extract_Initial_Point(self, _initia_data):
        _sub_title = "Initial Point \n" + small_hLine
        self.lprint(_sub_title)
        _init_vec = []
        for _j, _result in enumerate(_initia_data):
            _init_temp      = _result.tolist()
            _init_vec.append(_init_temp)
            self.lprint("[%2d] %s" %(_j, str(_init_vec[-1])))
        self.lprint(small_hLine)
        return _init_vec


    def Analysis_Result(self, _RST_data):
        # Ready for Processing 
        _total_attem = len(_RST_data[0])
        _local_data  = copy.deepcopy(_RST_data)

        # Check Fail case and extract the case from the list
        _fail_data = []
        for _j, _data in enumerate(_local_data):
            _costdiff_data = _data[1]
            if _costdiff_data > self.threshold_cost:
                _fail_data.append(_data)

        # remove appropriate components
        _result = [_k for _k in _local_data if _k not in _fail_data]
        _local_data.clear() 
        _local_data = _result

        # Confirm the processing 
        _msg_str = "Check \n" + small_hLine
        self.lprint(_msg_str)
        _iteratio, _costdiff, _locadiff = [], [], []
        for _j, _data in enumerate(_local_data):
            _iteratio.append(_data[0])
            _costdiff.append(_data[1])
            _locadiff.append(_data[2])
            self.lprint("[%2d] %6d  %12.8f  %12.8f" \
                        %(_j, _iteratio[-1], _costdiff[-1], _locadiff[-1]))
        self.lprint("\n")
        _success    = len(_local_data)
        _success_r  = (100.0 * _success)/(1.0 * self._num_attemp)
        self._success_info.append(_success_r)

        np_iteratio = np.array(_iteratio)
        np_costdiff = np.array(_costdiff)
        np_locadiff = np.array(_locadiff)

        _avg_iter   = np.average(np_iteratio)
        _avg_cost   = np.average(np_costdiff)
        _avg_loca   = np.average(np_locadiff)

        # avoid NaN
        _avg_iter   = np.nan_to_num(_avg_iter)

        _msg_str    = "| AVG Iteration  | AVG Cost DIff | AVG Loca Diff | Success Ratio |\n"
        _msg_str   += "|----------------|---------------|---------------|---------------|\n"
        _msg_str   += "|  %12d  | %12.8f  | %12.8f  | %12.8f  |\n" \
                        %(_avg_iter, _avg_cost, _avg_loca, _success_r)
        self.lprint(_msg_str)

        # Final Processing : Store the result
        self.result_msg.append(_msg_str)

    # -----------------------------------------------------
    # Interface Functions 
    # -----------------------------------------------------
    def Get_data(self):
        return self._data

    def lprint(self, _msg):
        _lmsg = _msg + "\n"
        self._File_msg.append(_lmsg)
        print(_lmsg, end='')

    def Get_total_msg(self):
        return self._File_msg

    def Get_result_msg(self):            
        return self.result_msg

    def Get_initial_point(self):
        return self._init_vec

    def Get_trace_data(self):
        return self.trace_data

# =================================================================
# Class 3 : Test Function Class
# =================================================================
class Test_Class:
    def __init__(self, _init_pattern):
        self.TestObj         = None
        self.AnalyzeObj      = None
        
        self._ctr_list       = []
        self._arg_str        = []
        self._Test_function  = []
        self._init_vector    = []

        self._result_list    = []
        self._total_filemsg  = []
        
        # For Reporting 
        self._msg_list       = []
        
        # [0] : Fixed initial point with a same default initial point
        # [1] : Random initial point with a 2 dimensional initial point
        # [2] : Fixed initial point for some variation
        # [3] : Random initial point for some variation
        # [4] : No assign 
        self._init_pattern   = _init_pattern

        # set function Limit
        self.function_Limit  = []
        self.function_Limit.append([[-5, 10],       [-5, 10]])          # 0. Rosenbrock
        self.function_Limit.append([[-1.5, 1.5],    [-1.5, 1.5]])       # 1. Ackley
        self.function_Limit.append([[-10.24, 10.24],[-10.24, 10.24]])   # 2. Whitley
        self.function_Limit.append([[-1.3, 0.6],    [-1.3, 0.6]])       # 3. Rosenbrock Modification
        self.function_Limit.append([[480, 550],     [380, 450]])        # 4. Egg Holder
        self.function_Limit.append([[-5, 5],        [-5, 5]])           # 5. Xin-She Yang N.4
        self.function_Limit.append([[-1, 1],        [-1, 1]])           # 6. Salomon
        self.function_Limit.append([[-1.0, 1.0],    [-1.0, 1.0]])       # 7. Drop-Wave
        self.function_Limit.append([[-1.0, 1.0],    [-1.0, 1.0]])       # 8. Powell
        self.function_Limit.append([[-4, 4],        [-4, 4]])           # 9. Schaffel N. 2

    # -----------------------------------------------------
    # Arguments
    # -----------------------------------------------------
    def Set_Argument_String(self, _init_pattern_id, _qparam=3):
        _q_str = "-qm 2 -q %d" %_qparam + " "
        self._arg_str.append("-s 1 -t 12000 -qt 1 ")
        self._arg_str.append(_q_str)
        self._arg_str.append(self._init_pattern[_init_pattern_id])

    def Set_Control_Parameter(self, _num_attemp, _quantize, _appendinit):
        self._ctr_list.clear()
        self._ctr_list.append(_num_attemp)
        self._ctr_list.append(_quantize)
        self._ctr_list.append(_appendinit)

    def Generate_Report(self, _Report_FileName, _Result_FileName, q_param):
        _method_str = ["General NLP Operation", "Quantized NLP Operation"]
        _remove_str = "| AVG Iteration  | AVG Cost DIff | AVG Loca Diff | Success Ratio |\n|----------------|---------------|---------------|---------------|\n"

        # Generation Report msg
        for _j, _tf_id in enumerate(self._Test_function):
            _test_func  = "Test Function : %s " %self.TestObj.Get_Test_Function_Name(_tf_id) + "\n"
            self._msg_list.append(_test_func)
            for _k, _result in enumerate(self._result_list):
                self._msg_list.append(_method_str[_k] +"\n") 
                for _i, _data in enumerate(_result):
                    if _i > 0:
                        _data = _data.replace(_remove_str, '')
                    else: 
                        pass
                    self._msg_list.append(_data)

        # Make Report File
        with open(_Report_FileName, 'w') as f:
            for _k, _msg in enumerate(self._msg_list):
                f.write(_msg)
                print(_msg, end='')

            _msg = "Q-Param : %2d" %q_param + "\n"
            f.write(_msg)
            print(_msg, end='')

        # Make Result File 
        with open(_Result_FileName, 'w') as f:
            for _i, _proc_msg in enumerate(self._total_filemsg):
                for _j, _msg in enumerate(_proc_msg):
                    f.write(_msg)

    # It is just a test code for mutiple algorithm
    def Generate_plot(self, c_ns, _trace_data, _func_id, _proc=False):
        if _proc :
            _conv_data, _quan_data = None, None
            # Analyze data
            for _k, _cq_data in enumerate(_trace_data):
                if _k == 0 :
                    for _j, _per_func_data in enumerate(_cq_data):
                        _conv_data = _per_func_data
                else       :
                    for _j, _per_func_data in enumerate(_cq_data):
                        _quan_data = _per_func_data

            # extract initial point and dimension
            _init   = self.TestObj.NLPobj.Get_Init_point()
            _dim    = np.size(_init)
            # _init_info[0] fix (True) or variance _init_info[1] Given or Not
            _init_info  = [True, True]
            # Number of algorithm
            _num_algo   = len(_conv_data)
            # nlp class 2 object
            c_nlp   = self.TestObj.NLPobj.nlp

            # Plot trace to optimization algorithms
            _test_data = []
            for _a_idx in range(_num_algo):
                _cdata = _conv_data[_a_idx]
                _cdata = c_ns.Conv_strvec_to_floatvec(_cdata, _dim=_dim)
                _cdata.insert(0, _init)
                
                _qdata = _quan_data[_a_idx]
                _qdata = c_ns.Conv_strvec_to_floatvec(_qdata, _dim=_dim)
                _qdata.insert(0, _init) 
                
                _test_data.append([_cdata, _qdata])

            # _FunctionId=_func_id, _AlgorithmIndex=_test_idx, _init_vec=_init, _init_info=_init_info
            for _algo_idx in range(_num_algo):
                _param_func = [_func_id, _algo_idx, _init, _init_info]
                c_ns.plot_trace_for_test_nlp(c_nlp=c_nlp, _ResultData=_test_data[_algo_idx], _param_func=_param_func)
        else:
            print("No Generation plot")
        return 0

    # it is plot just an objective function
    def draw_objective_function (self, c_ns, _func_id, _proc=False):
        if _proc :
            # extract initial point and dimension
            _init   = self.TestObj.NLPobj.Get_Init_point()
            _dim    = np.size(_init)
            # _init_info[0] fix (True) or variance _init_info[1] Given or Not
            _init_info  = [True, True]
            # nlp class 2 object
            c_nlp   = self.TestObj.NLPobj.nlp

            # Operation
            # _SideData = [_FunctionId, _Limit_X, _Limit_Y, _init_vec, _initial_info]
            _limit_data = self.function_Limit[_func_id]
            _param_func = [_func_id, _limit_data[0], _limit_data[1], _init, _init_info]
            c_ns.plot_objective_function(c_nlp=c_nlp, _SideData=_param_func)
        else:
            pass

    def Main_Processing(self):
        # Benchmark Test 
        Test = Bechmark_Test(self._arg_str, self._ctr_list)
        Test.Get_Benchmnark_function_List(self._Test_function)
        Test(self._init_vector)

        # Analyze the result 
        Analyze = Analysis_Test_Results(Test._res_func)
        Analyze(self._ctr_list)
        
        # Store middle result 
        self._result_list.append(Analyze.Get_result_msg())
        self._total_filemsg.append(Analyze.Get_total_msg())
        self._init_vector = Analyze.Get_initial_point()

        self.TestObj    = Test
        self.AnalyzeObj = Analyze

# =================================================================
# Class 4 : Other Service Function
# =================================================================
class Service_class:

    def __init__(self, _dimension, _additional_args = ''):
        # [0] : Fixed initial point with a same default initial point
        # [1] : Random initial point with a 2 dimensional initial point
        # [2] : Fixed initial point for some variation
        # [3] : Random initial point for some variation
        # [4] : No assign 
        #_init_pattern   = ['-rf 1 -d 2 ', '-rf 0 -d 100 ', '-rf 1 ', '-rf 0 ', '']
        self._init_pattern  = []
        self._dimension     = _dimension

        # Inital operation for parameter set
        self.Generate_init_pattern(_additional_args)

    def Generate_init_pattern(self, _additional_args):
        _dim_str    = "-d %d " %self._dimension
        _base_str   = ['-rf 1 ', '-rf 0 ']
        
        _op_str     = _base_str[0] + _dim_str
        self._init_pattern.append(_op_str)
        _op_str     = _base_str[1] + _dim_str
        self._init_pattern.append(_op_str)
        _op_str     = _base_str[0]
        self._init_pattern.append(_op_str)
        _op_str     = _base_str[1]
        self._init_pattern.append(_op_str)
        _op_str     = ''
        self._init_pattern.append(_op_str)

        if len(_additional_args) > 0:
            for _k, _args in enumerate(self._init_pattern):
                self._init_pattern[_k] = _args + _additional_args
        else:
            pass            

    def get_init_pattern(self):
        return self._init_pattern

# =================================================================
# Class 5 : Worker Class : It is just a example code
# =================================================================
'''
class Worker_class:
    def __init__(self, _TCobj):
        self.TC = _TCobj

    def main_Proc(self, _num_attemp, _q_value, _func_id, _dimension, _additional_args=''):
        # Set Benchmark Function
        self.TC._Test_function.append(_func_id)
        # -----------------------------------------------------
        # Benchmark Test : 1 using random Initial Point (1)
        # -----------------------------------------------------
        self.TC.Set_Argument_String(_init_pattern_id=1)
        self.TC.Set_Control_Parameter(_num_attemp, _quantize=False, _appendinit=False)
        self.TC.Main_Processing()
        # -----------------------------------------------------
        # Benchmark Test : 2 using same Initial Point (4)
        # -----------------------------------------------------
        self.TC.Set_Argument_String(_init_pattern_id=4, _qparam=_q_value)
        self.TC.Set_Control_Parameter(_num_attemp, _quantize=True, _appendinit=True)
        self.TC.Main_Processing()
        # -----------------------------------------------------
        # Final Processing
        # -----------------------------------------------------
        _Function_Name   = self.TC.TestObj.Get_Test_Function_Name(_func_id)
        _Report_FileName = "Report_NLP_" + _Function_Name + "-d%d" %_dimension + ".txt"
        _Result_FileName = "Full_Result_NLP_" + _Function_Name + "-d%d" %_dimension + ".txt"

        self.TC.Generate_Report(_Report_FileName, _Result_FileName, _q_value)
'''
