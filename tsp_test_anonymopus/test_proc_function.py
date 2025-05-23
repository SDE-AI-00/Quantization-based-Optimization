#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################
# test_proc_function.py
# It contains various classes
# It is just a library for plotting 
###########################################################
_description = '''\
====================================================
test_proc_function.py : Librray
                    using import class in test_proc_function.py 
====================================================
'''
_guide_line = "----------------------------------------------------"

import argparse
import textwrap
from numpy import sin
import numpy as np
import math
import random
from datetime import datetime
from cec_2022_testfunction import get_cec2022_test
#from plot_function import Arbitrary_function
import my_debug as DBG
#import glob

class guide_class:
    def __init__(self, _intro_msg, L_Param, bUseParam=False):
        self._intro_msg = _intro_msg
        self.L_Param    = L_Param
        self.bUseParam  = bUseParam
        self.now        = datetime.now()
        self.str_nowtime= self.now.strftime('%Y-%m-%d_%H_%M_%S')

        self.args       = self.ArgumentParse()

    def ArgumentParse(self):
        parser = argparse.ArgumentParser(
            prog='test_proc.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent(self._intro_msg))

        parser.add_argument('-f', '--test_fuinction_id',
                            help="Test Function ID [0] default_test_func [1] parabolic_washboard_potential \
                                  [2] Gamma-Sin [3~10] CEC 2022 Test Functions",
                            type=int, default=0)
        parser.add_argument('-iqp', '--initial_quantization_power', help="Initial Quantization Power",
                            type=int, default=-2)
        # Search mode Parameter
        parser.add_argument('-i',   '--initial_point', help="Initial Point",
                            type=float, default=-2.0)
        parser.add_argument('-it',  '--iteration', help="Number of Iteration",
                            type=int, default=100)
        parser.add_argument('-ic',  '--increments', help="Unit Incremets default value is 0.1",
                            type=float, default=0.1)
        # Operation Parameter
        parser.add_argument('-om',  '--operation_mode',
                            help="Operation Mode [0] General Search mode [1] Plotting mode [2] Manual Operation mode",
                            type=int, default=0)
        parser.add_argument('-vb',  '--verbose',
                            help="Verbose of Operation [0] All information [1] Only update with dot [2] Only update",
                            type=int, default=0)
        # Service Parameter
        parser.add_argument('-log','--logfile',  help="Name of the Lof file [Default] Log_xxxx_xx_xx....txt",
                            type=str, default="log_")

        if self.bUseParam:
            args = parser.parse_args(self.L_Param)
        else:
            args = parser.parse_args()

        args.logfile += self.str_nowtime+".txt"
        #args.stopping_temp      = 0.1 / (1.0 * args.stopping_iter)
        #args.loaddatafromfile   = True if args.loaddatafromfile == 1 and glob.glob("*.pkl") else False
        #args.active_animation   = True if args.active_animation else False
        print(self._intro_msg)
        print(_guide_line)
        return args

#============================================================================
# Operational Library
#============================================================================
class test_function_list:
    def __init__(self):
        self.l_test_functions = []
        # For Local 3 functions
        self.l_test_functions.append(default_test_func(b_msg=False))
        self.l_test_functions.append(parabolic_washboard_potential(b_msg=False))
        self.l_test_functions.append(Gamma_function(b_msg=False))

        # For plot local and CEC functions
        self.c_tf02     = Arbitrary_function()
        self.l_test_functions.append(Xin_She_Yang_N4(c_tf=self.c_tf02, b_msg=False))
        self.l_test_functions.append(Salomon(c_tf=self.c_tf02, b_msg=False))
        self.l_test_functions.append(Drop_Wave(c_tf=self.c_tf02, b_msg=False))
        self.l_test_functions.append(Schaffel_N2(c_tf=self.c_tf02, b_msg=False))

        self.c_cec = get_cec2022_test()
        self.l_test_functions.append(Bent_Cigar_Function(c_cec=self.c_cec, b_msg=False))
        self.l_test_functions.append(Rastrigins_Function(c_cec=self.c_cec, b_msg=False))
        self.l_test_functions.append(High_Conditioned_Elliptic_Function(c_cec=self.c_cec, b_msg=False))
        self.l_test_functions.append(HGBat_Function(c_cec=self.c_cec, b_msg=False))
        self.l_test_functions.append(Rosenbrocks_Function(c_cec=self.c_cec, b_msg=False))
        self.l_test_functions.append(Griewank_Function(c_cec=self.c_cec, b_msg=False))
        self.l_test_functions.append(Ackleys_Function(c_cec=self.c_cec, b_msg=False))

    def __call__(self, _func_ID):
        return self.l_test_functions[_func_ID]

class function_library:
    def __init__(self, r_min, r_max, _increments):
        # sample input range uniformly at 0.1 increments
        self.r_min      = r_min
        self.r_max      = r_max
        self.increments = _increments
        self.inputs     = np.arange(self.r_min, self.r_max, self.increments)
        # State information for global
        self.c_state    = None
        # functions
        self._obj_func  = None
        self._qfunction = None
        # For Transition Probability
        self._correct   = 0
        self._total     = 0

    def set_state(self, _state):
        self.c_state = _state

    def get_state(self):
        return self.c_state

    def find_location(self, _arg):
        _ret = self.r_min + self.increments * _arg
        return _ret

    def find_min(self, _result):
        _min_value  = np.min(_result)
        _min_arg    = self.find_location(np.argmin(_result))
        print("minimum value : %f  @ %f " %(_min_value, _min_arg))
        return [_min_value, _min_arg]

    def find_max(self, _result):
        _max_value  = np.max(_result)
        _max_arg    = self.find_location(np.argmax(_result))
        print("maximum value : %f  @ %f " %(_max_value, _max_arg))
        return [_max_value, _max_arg]

    def set_params(self, _x, _obj_func, _qfunction, b_init):
        if b_init:
            self._obj_func  = _obj_func
            self._qfunction = _qfunction
            self.c_state, _y= self.compute_state(_x)

            _tr_prob        = self.get_transition_probability()
            print("Initial State")
            print("Initial Point : %f " %_x)
            print("Initial Value : %f " %_y)
            print("Initial State : %f " %self.c_state)
            print("Transition Probability : %f " % _tr_prob)
            print(_guide_line)
        else:
            pass

    #-----------------------------------------------------
    # f(x)와 f(x)의 Quantization f^Q(x)값을 계산한다.
    # -----------------------------------------------------
    def compute_state(self, _x):
        _y      = self._obj_func(_x)
        _cstate = self._qfunction(_y)
        return _cstate, _y

    #-----------------------------------------------------
    # Object function의 Transition Probability를 계산한다. : state<= f^Q(_x)
    # -----------------------------------------------------
    def get_atom_of_transition_probability(self):
        return self._correct, self._total

    def get_transition_probability(self, _verbose=True):
        # self.c_state로만 구성된 np 배열 생성
        self._total     = np.size(self.inputs)
        n_c_state       = np.array([self.c_state for _k in range(self._total)])
        # 현재 QP에서 전체 Space에서 Quantization state, 함수값
        n_Qf, n_f       = self.compute_state(self.inputs)
        # Condition
        n_coorect       = (n_Qf <= n_c_state)
        # sum of True Condition
        self._correct   = int(np.sum(n_coorect))
        p_transition    = (1.0 * self._correct)/(1.0 * self._total)
        # Print Result
        if _verbose:
            print("correct:%d  total:%d" %(self._correct, self._total))
            print(_guide_line)
        else: pass
        return p_transition
    #-----------------------------------------------------
    # Object function의 state Probability를 계산한다. state== f^Q(_x)
    # -----------------------------------------------------
    def get_state_probability(self, _all_state, _x):
        # initilization
        _total      = np.size(_all_state)
        # make state vector of _x
        _fQ,_       = self.compute_state(_x)
        _x_state    = np.array([_fQ for _k in range(_total)])
        # Condition
        n_correct   = (_x_state == _all_state)
        # sum of True Condition
        _correct    = int(np.sum(n_correct))
        p_transition= (1.0 * _correct)/(1.0 * _total)

        print("State Probability of %f  state=%f" %(_x, _fQ))
        print("correct:%d  total:%d" %(_correct, _total))
        print(_guide_line)
        return p_transition

    def one_step_process(self):
        _x          = random.uniform(self.r_min, self.r_max)
        _cstate, _y = self.compute_state(_x)

        return _x, _y, _cstate
#============================================================================
# Quantization
#============================================================================
class quantization:
    def __init__(self, _iqp=-2):
        self._eta   = 1
        self._base  = 2
        self._power = _iqp
        self._QP    = self.set_QP(_powidx=_iqp)

        print("QP         : %f " %self._QP)
        if _iqp < 0 :
            print("Inverse QP : %f " %(1.0/self._QP))
        print(_guide_line)

    def __call__(self, _x):
        _ret = 1/self._QP * np.floor(self._QP * (_x + 0.5/self._QP))
        return _ret

    def set_QP(self, _powidx=-2):
        _ret = self._eta * math.pow(self._base, _powidx)
        return _ret

    def get_QP(self):
        return self._QP

    def increse_QP(self):
        self._power += 1.0
        self._QP = self.set_QP(_powidx=self._power)

#============================================================================
# Test Function 1
#============================================================================
class default_test_func:
    def __init__(self, b_msg=True):
        self.x_optima   = 5.145735
        self.r_min      = -2.7
        self.r_max      = 7.5
        self._name      = "Default test function"
        self.equation   = "$f(x) = \\sin(x) + \\sin(\\frac{10.0}{3.0}x) + 2.0$"
        if b_msg :
            print("%s %s" %(self._name, self.equation))
            print("Optimal value : %f  @ %f" %(self.x_optima, self._function(self.x_optima)))
            print(_guide_line)
        else : pass
    def _function(self, x):
        return sin(x) + sin((10.0 / 3.0) * x) + 2.0

    def get_optima(self):
        return self.x_optima

    def get_range(self):
        return [self.r_min, self.r_max]

#============================================================================
# Test Function 2
#============================================================================
class parabolic_washboard_potential:
    def __init__(self, b_msg=True):
        self.x_optima   = -0.157
        self.r_min      = -6.0
        self.r_max      = 6.0
        self._name      = "Parabolic Washboard Potential"
        self.equation   = "$f(x) = 0.125 x^2 + 2 \\sin(10x) + 2$"
        if b_msg :
            print("%s %s" %(self._name, self.equation))
            print("Optimal value : %f  @ %f" %(self.x_optima, self._function(self.x_optima)))
            print(_guide_line)
        else : pass
    def _function(self, x):
        return 0.125 * pow(x, 2) + 2.0 * sin(10 * x) + 2.0

    def get_optima(self):
        return self.x_optima

    def get_range(self):
        return [self.r_min, self.r_max]

#============================================================================
# Test Function 3
#============================================================================
from scipy.special import gamma
class Gamma_function:
    def __init__(self, b_msg=True):
        self.x_optima   = 1.5
        self.r_min      = 0.2
        self.r_max      = 4.0
        self._name      = "Gamma Function Based Simple Multimodal Problem"
        self.equation   = "$f(x) = \Gamma(x) + \sin( \pi x)$"
        if b_msg :
            print("%s %s" %(self._name, self.equation))
            print("Optimal value : %f  @ %f" %(self.x_optima, self._function(self.x_optima)))
            print(_guide_line)
        else : pass
    def _function(self, x):
        _result = gamma(x) + np.sin( np.pi * x)
        return _result

    def get_optima(self):
        return self.x_optima

    def get_range(self):
        return [self.r_min, self.r_max]

#============================================================================
# Test Function 3~6 (ID 3~6)
#============================================================================
class Xin_She_Yang_N4:
    def __init__(self, c_tf, r_min=-6.0, r_max=6.0, b_msg=True):
        _ID             = 0
        self.c_function = c_tf.L_op_func[_ID ]
        self.x_optima   = c_tf.L_x_optima[_ID ]
        self.r_min      = r_min
        self.r_max      = r_max
        self._name      = c_tf.L_f_name[_ID ]
        self.equation   = c_tf.L_equation[_ID]
        if b_msg :
            print("%s %s" %(self._name, self.equation))
            print("Optimal value : %f  @ %f" %(self.x_optima, self._function(self.x_optima)))
            print(_guide_line)
        else : pass
    def _function(self, x):
        _result = self.c_function(x)
        return _result

    def get_optima(self):
        return self.x_optima

    def get_range(self):
        return [self.r_min, self.r_max]

class Salomon:
    def __init__(self, c_tf, r_min=-6.0, r_max=6.0, b_msg=True):
        _ID             = 1
        self.c_function = c_tf.L_op_func[_ID ]
        self.x_optima   = c_tf.L_x_optima[_ID ]
        self.r_min      = r_min
        self.r_max      = r_max
        self._name      = c_tf.L_f_name[_ID ]
        self.equation   = c_tf.L_equation[_ID]
        if b_msg :
            print("%s %s" %(self._name, self.equation))
            print("Optimal value : %f  @ %f" %(self.x_optima, self._function(self.x_optima)))
            print(_guide_line)
        else : pass
    def _function(self, x):
        _result = self.c_function(x)
        return _result

    def get_optima(self):
        return self.x_optima

    def get_range(self):
        return [self.r_min, self.r_max]

class Drop_Wave:
    def __init__(self, c_tf, r_min=-6.0, r_max=6.0, b_msg=True):
        _ID             = 2
        self.c_function = c_tf.L_op_func[_ID ]
        self.x_optima   = c_tf.L_x_optima[_ID ]
        self.r_min      = r_min
        self.r_max      = r_max
        self._name      = c_tf.L_f_name[_ID ]
        self.equation   = c_tf.L_equation[_ID]
        if b_msg :
            print("%s %s" %(self._name, self.equation))
            print("Optimal value : %f  @ %f" %(self.x_optima, self._function(self.x_optima)))
            print(_guide_line)
        else : pass
    def _function(self, x):
        _result = self.c_function(x)
        return _result

    def get_optima(self):
        return self.x_optima

    def get_range(self):
        return [self.r_min, self.r_max]

class Schaffel_N2:
    def __init__(self, c_tf, r_min=-6.0, r_max=6.0, b_msg=True):
        _ID             = 3
        self.c_function = c_tf.L_op_func[_ID ]
        self.x_optima   = c_tf.L_x_optima[_ID ]
        self.r_min      = r_min
        self.r_max      = r_max
        self._name      = c_tf.L_f_name[_ID ]
        self.equation   = c_tf.L_equation[_ID]
        if b_msg :
            print("%s %s" %(self._name, self.equation))
            print("Optimal value : %f  @ %f" %(self.x_optima, self._function(self.x_optima)))
            print(_guide_line)
        else : pass
    def _function(self, x):
        _result = self.c_function(x)
        return _result

    def get_optima(self):
        return self.x_optima

    def get_range(self):
        return [self.r_min, self.r_max]

#============================================================================
# Test Function 12~19 (ID 3~10)
#============================================================================
class Bent_Cigar_Function:
    def __init__(self, c_cec, _id=0, b_msg=True):
        self.c_function = c_cec.test_functions[_id]
        self.x_optima = self.c_function.minimum_info['min_point']
        self.r_min = self.c_function.plot_range[0]
        self.r_max = self.c_function.plot_range[1]
        self._name = self.c_function.name
        self.equation = self.c_function.equation
        if b_msg:
            print("%s %s" % (self._name, self.equation))
            print("Optimal value : %f  @ %f" % (self.x_optima, self._function(self.x_optima)))
            print(_guide_line)
        else:
            pass

    def function_evaluate_for_point_input(self, x):
        l_x = [x, 0]
        _x = np.array(l_x)
        return self.c_function(_x)

    def function_evaluate_for_ndarray(self, x):
        _y = np.zeros(len(x))
        mesh_x = np.dstack([x, _y])
        mesh_out = np.apply_along_axis(self.c_function, 2, mesh_x)
        return mesh_out.reshape(len(x))

    def _function(self, x):
        _chk_param = str(type(x))
        if _chk_param == "<class 'float'>" or _chk_param == "<class 'int'>":
            _result = self.function_evaluate_for_point_input(x)
        elif _chk_param == "<class 'numpy.ndarray'>":
            _result = self.function_evaluate_for_ndarray(x)
        else:
            DBG.dbg("Type is incompatible!! ")
            DBG.dbg("Type Info : %s" % _chk_param)
            exit()
        return _result

    def get_optima(self):
        return self.x_optima

    def get_range(self):
        return [self.r_min, self.r_max]

class Rastrigins_Function:
    def __init__(self, c_cec, _id=1, b_msg=True):
        self.c_function = c_cec.test_functions[_id]
        self.x_optima   = self.c_function.minimum_info['min_point']
        self.r_min      = self.c_function.plot_range[0]
        self.r_max      = self.c_function.plot_range[1]
        self._name      = self.c_function.name
        self.equation   = self.c_function.equation
        if b_msg :
            print("%s %s" %(self._name, self.equation))
            print("Optimal value : %f  @ %f" %(self.x_optima, self._function(self.x_optima)))
            print(_guide_line)
        else : pass
    def function_evaluate_for_point_input(self, x):
        l_x = [x, 0]
        _x = np.array(l_x)
        return self.c_function(_x)
    def function_evaluate_for_ndarray(self, x):
        _y = np.zeros(len(x))
        mesh_x = np.dstack([x, _y])
        mesh_out = np.apply_along_axis(self.c_function, 2, mesh_x)
        return mesh_out.reshape(len(x))
    def _function(self, x):
        _chk_param = str(type(x))
        if _chk_param == "<class 'float'>" or _chk_param == "<class 'int'>":
            _result = self.function_evaluate_for_point_input(x)
        elif _chk_param == "<class 'numpy.ndarray'>":
            _result = self.function_evaluate_for_ndarray(x)
        else:
            DBG.dbg("Type is incompatible!! ")
            DBG.dbg("Type Info : %s" %_chk_param)
            exit()
        return _result

    def get_optima(self):
        return self.x_optima

    def get_range(self):
        return [self.r_min, self.r_max]

class High_Conditioned_Elliptic_Function:
    def __init__(self, c_cec, _id=2, b_msg=True):
        self.c_function = c_cec.test_functions[_id]
        self.x_optima = self.c_function.minimum_info['min_point']
        self.r_min = self.c_function.plot_range[0]
        self.r_max = self.c_function.plot_range[1]
        self._name = self.c_function.name
        self.equation = self.c_function.equation
        if b_msg:
            print("%s %s" % (self._name, self.equation))
            print("Optimal value : %f  @ %f" % (self.x_optima, self._function(self.x_optima)))
            print(_guide_line)
        else:
            pass

    def function_evaluate_for_point_input(self, x):
        l_x = [x, 0]
        _x = np.array(l_x)
        return self.c_function(_x)

    def function_evaluate_for_ndarray(self, x):
        _y = np.zeros(len(x))
        mesh_x = np.dstack([x, _y])
        mesh_out = np.apply_along_axis(self.c_function, 2, mesh_x)
        return mesh_out.reshape(len(x))

    def _function(self, x):
        _chk_param = str(type(x))
        if _chk_param == "<class 'float'>" or _chk_param == "<class 'int'>":
            _result = self.function_evaluate_for_point_input(x)
        elif _chk_param == "<class 'numpy.ndarray'>":
            _result = self.function_evaluate_for_ndarray(x)
        else:
            DBG.dbg("Type is incompatible!! ")
            DBG.dbg("Type Info : %s" % _chk_param)
            exit()
        return _result

    def get_optima(self):
        return self.x_optima

    def get_range(self):
        return [self.r_min, self.r_max]

class HGBat_Function:
    def __init__(self, c_cec, _id=3, b_msg=True):
        self.c_function = c_cec.test_functions[_id]
        self.x_optima = self.c_function.minimum_info['min_point']
        self.r_min = self.c_function.plot_range[0]
        self.r_max = self.c_function.plot_range[1]
        self._name = self.c_function.name
        self.equation = self.c_function.equation
        if b_msg:
            print("%s %s" % (self._name, self.equation))
            print("Optimal value : %f  @ %f" % (self.x_optima, self._function(self.x_optima)))
            print(_guide_line)
        else:
            pass

    def function_evaluate_for_point_input(self, x):
        l_x = [x, 0]
        _x = np.array(l_x)
        return self.c_function(_x)

    def function_evaluate_for_ndarray(self, x):
        _y = np.zeros(len(x))
        mesh_x = np.dstack([x, _y])
        mesh_out = np.apply_along_axis(self.c_function, 2, mesh_x)
        return mesh_out.reshape(len(x))

    def _function(self, x):
        _chk_param = str(type(x))
        if _chk_param == "<class 'float'>" or _chk_param == "<class 'int'>":
            _result = self.function_evaluate_for_point_input(x)
        elif _chk_param == "<class 'numpy.ndarray'>":
            _result = self.function_evaluate_for_ndarray(x)
        else:
            DBG.dbg("Type is incompatible!! ")
            DBG.dbg("Type Info : %s" % _chk_param)
            exit()
        return _result

    def get_optima(self):
        return self.x_optima

    def get_range(self):
        return [self.r_min, self.r_max]

class Rosenbrocks_Function:
    def __init__(self, c_cec, _id=4, b_msg=True):
        self.c_function = c_cec.test_functions[_id]
        self.x_optima = self.c_function.minimum_info['min_point']
        self.r_min = self.c_function.plot_range[0]
        self.r_max = self.c_function.plot_range[1]
        self._name = self.c_function.name
        self.equation = self.c_function.equation
        if b_msg:
            print("%s %s" % (self._name, self.equation))
            print("Optimal value : %f  @ %f" % (self.x_optima, self._function(self.x_optima)))
            print(_guide_line)
        else:
            pass
        # in test_proc_function.py,  1 Dimension and suppose y=0 
        self.x_optima = self.x_optima[0]
    def function_evaluate_for_point_input(self, x):
        l_x = [x, 0]
        _x = np.array(l_x)
        return self.c_function(_x)

    def function_evaluate_for_ndarray(self, x):
        _y = np.zeros(len(x))
        mesh_x = np.dstack([x, _y])
        mesh_out = np.apply_along_axis(self.c_function, 2, mesh_x)
        return mesh_out.reshape(len(x))

    def _function(self, x):
        _chk_param = str(type(x))
        if _chk_param == "<class 'float'>" or _chk_param == "<class 'int'>":
            _result = self.function_evaluate_for_point_input(x)
        elif _chk_param == "<class 'numpy.ndarray'>":
            _result = self.function_evaluate_for_ndarray(x)
        else:
            DBG.dbg("Type is incompatible!! ")
            DBG.dbg("Type Info : %s" % _chk_param)
            exit()
        return _result

    def get_optima(self):
        return self.x_optima

    def get_range(self):
        return [self.r_min, self.r_max]

class Griewank_Function:
    def __init__(self, c_cec, _id=5, b_msg=True):
        self.c_function = c_cec.test_functions[_id]
        self.x_optima = self.c_function.minimum_info['min_point']
        self.r_min = self.c_function.plot_range[0]
        self.r_max = self.c_function.plot_range[1]
        self._name = self.c_function.name
        self.equation = self.c_function.equation
        if b_msg:
            print("%s %s" % (self._name, self.equation))
            print("Optimal value : %f  @ %f" % (self.x_optima, self._function(self.x_optima)))
            print(_guide_line)
        else:
            pass

    def function_evaluate_for_point_input(self, x):
        l_x = [x, 0]
        _x = np.array(l_x)
        return self.c_function(_x)

    def function_evaluate_for_ndarray(self, x):
        _y = np.zeros(len(x))
        mesh_x = np.dstack([x, _y])
        mesh_out = np.apply_along_axis(self.c_function, 2, mesh_x)
        return mesh_out.reshape(len(x))

    def _function(self, x):
        _chk_param = str(type(x))
        if _chk_param == "<class 'float'>" or _chk_param == "<class 'int'>":
            _result = self.function_evaluate_for_point_input(x)
        elif _chk_param == "<class 'numpy.ndarray'>":
            _result = self.function_evaluate_for_ndarray(x)
        else:
            DBG.dbg("Type is incompatible!! ")
            DBG.dbg("Type Info : %s" % _chk_param)
            exit()
        return _result

    def get_optima(self):
        return self.x_optima

    def get_range(self):
        return [self.r_min, self.r_max]

class Ackleys_Function:
    def __init__(self, c_cec, _id=6, b_msg=True):
        self.c_function = c_cec.test_functions[_id]
        self.x_optima = self.c_function.minimum_info['min_point']
        self.r_min = self.c_function.plot_range[0]
        self.r_max = self.c_function.plot_range[1]
        self._name = self.c_function.name
        self.equation = self.c_function.equation
        if b_msg:
            print("%s %s" % (self._name, self.equation))
            print("Optimal value : %f  @ %f" % (self.x_optima, self._function(self.x_optima)))
            print(_guide_line)
        else:
            pass

    def function_evaluate_for_point_input(self, x):
        l_x = [x, 0]
        _x = np.array(l_x)
        return self.c_function(_x)

    def function_evaluate_for_ndarray(self, x):
        _y = np.zeros(len(x))
        mesh_x = np.dstack([x, _y])
        mesh_out = np.apply_along_axis(self.c_function, 2, mesh_x)
        return mesh_out.reshape(len(x))

    def _function(self, x):
        _chk_param = str(type(x))
        if _chk_param == "<class 'float'>" or _chk_param == "<class 'int'>":
            _result = self.function_evaluate_for_point_input(x)
        elif _chk_param == "<class 'numpy.ndarray'>":
            _result = self.function_evaluate_for_ndarray(x)
        else:
            DBG.dbg("Type is incompatible!! ")
            DBG.dbg("Type Info : %s" % _chk_param)
            exit()
        return _result

    def get_optima(self):
        return self.x_optima

    def get_range(self):
        return [self.r_min, self.r_max]
#============================================================================
# Test Function 3
#============================================================================
'''
from simulated_annealing import SimulatedAnnealing
class travelling_salesman_problem():
    def __init__(self):
        self.obj = SimulatedAnnealing()

        self.a = 0



#============================================================================
# Test Function 4
#============================================================================
class continuous_functions():
    def __init__(self, _function_id=-1):
        self.function_id = _function_id
'''




