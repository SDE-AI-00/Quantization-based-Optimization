#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Plotting Test Function
# Base URL     : https://wikidocs.net/63565
# Simple CNN for MNIST test (Truth test)
###########################################################################
_description = '''\
====================================================
plot_function.py : function plot 
====================================================
Example : python plot_function.py -f 1 -d -6 6
'''
import numpy as np
import matplotlib.pyplot as plt
import argparse, textwrap
from scipy.special import gamma
from scipy.misc import derivative
#import autograd.numpy as anp
#from autograd import hessian
from cec_2022_testfunction import cec_2022_test as c_cecf
import test_proc_function as c_Lib
import my_debug as DBG

def ArgumentParse(_intro_msg, L_Param, bUseParam=False):
    parser = argparse.ArgumentParser(
        prog='plot_function.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(_intro_msg))

    parser.add_argument('-f', '--function', help="Function [0] test_multimodal [1] parabolic_wash_board [2] Arbitraty Function ", default=0, type=int)
    parser.add_argument('-d', '--domain',   help="Domain", nargs='+', default=None, type=float)
    parser.add_argument('-p', '--use_param',help="Use Parameter", default=0, type=int)
    parser.add_argument('-fp','--print_function_list', help="Print provided Function List [Default : False]", default=0, type=int)
    parser.add_argument('-lg','--represent_legend', help="Represent Legend of Graph [Default : True]",
                        default=1, type=int)
    parser.add_argument('-iqp', '--initial_quantization_power', help="Initial Quantization Power",
                        type=int, default=-2)
    parser.add_argument('-i', '--initial_point', help="Initial Point",
                        type=float, default=4.0)
    parser.add_argument('-ts','--test_code', help="Test state [Default:False]",
                        type=int, default=0)

    args = parser.parse_args()

    args.use_param              = True if args.use_param > 0 else False
    args.print_function_list    = True if args.print_function_list > 0 else False
    args.represent_legend       = True if args.represent_legend > 0 else False
    args.test_code              = True if args.test_code == 1 else False

    print(_intro_msg)
    print("Parameter:")
    print("   use_param : %s" %("True" if args.use_param else "False"))
    print("   function  : %s" %args.function)
    if args.domain != None:
        print("   domain    : ", args.domain)
    else: pass
    print("====================================================")

    return args

class test_multimodal:
    # f(x) = \sin(x) + \sin(\frac{10.0}{3.0} x) + 2.0
    def __init__(self, _param=3.0):
        self.band   = 10.0/_param
        self.f_name = "test_multimodal"

    def __call__(self, sol):
        _ret    = np.sin(sol) + np.sin(self.band * sol) + 2.0
        return _ret

class parabolic_wash_board:
    # f(x) = 0.125 * pow(x, 2) + 2.0 * np.sin(self.band * x) + 2.0
    def __init__(self, _param=3.0):
        self.band = _param
        self.f_name = "parabolic_wash_board"

    def __call__(self, sol):
        _ret    = self.H_0(sol) + self.H_1p(sol)
        return _ret

    def H_0(self, sol):
        _ret = 0.125 * pow(sol, 2)
        return _ret

    def H_1p(self, sol):
        _ret = 2.0 * np.sin(self.band * sol) + 2.0
        return _ret

class Arbitrary_function:
    def __init__(self, _param=0):
        self._func_id   = _param

        # Set Local Functions
        _prefix         = "Arbitrary_function"
        L_f_name        = ["Xin_She_Yang_N4", "Salomon", "Drop_Wave", "Schaffel_N2", "Gamma_sin01"]
        L_x_optima      = [0.0, 0.0, 0.0, 0.0, 1.6]
        L_equation      = []
        L_equation.append("$f(x) = 2.0 + (\sum_{i=1}^d \sim^2(x_i) - \exp(-\sum_{i=1}^d x_i^2) \exp(-\sum_{i=1}^d \sin^2 \sqrt{|x_i|})$")
        L_equation.append("$f(x) = 1 - \cos \left(2 \pi \sqrt{\sum_{i=1}^d x_i^2} \\right) + 0.1 \sqrt{\sum_{i=1}^d x_i^2}$")
        L_equation.append("$f(x) = 1 - \\frac{1 = \cos \left( 12 + \sqrt{x^2 + y^2} \\right)}{0.5 (x^2 + y^2) + 2}")
        L_equation.append("$0.5 + \\frac{\sin^2 (x^2 - y^2) - 0.5}{(1 + 0.001(x^2 + y^2)^2}$")
        L_equation.append("$\Gamma(x) + \sin( \pi x)$")

        self.L_op_func  = []
        self.L_op_func.append(self.Xin_She_Yang_N4)
        self.L_op_func.append(self.Salomon)
        self.L_op_func.append(self.Drop_Wave)
        self.L_op_func.append(self.Schaffel_N2)
        self.L_op_func.append(self.Gamma_sin01)

        # For Interface to other files
        self.L_f_name   = L_f_name.copy()
        self.L_x_optima = L_x_optima.copy()
        self.L_equation = L_equation.copy()

        # Set CEC_2022 Functions
        cecf_id = self._func_id - len(L_f_name)
        if cecf_id < 0: pass
        else:
            self.c_cecf = c_cecf(cecf_id)
            for _k, _func in enumerate(self.c_cecf.test_functions):
                L_f_name.append(_func.name)
                L_equation.append(_func.equation)
                L_x_optima.append(_func.minimum_info['min_point'])
                self.L_op_func.append(_func)

        try :
            self.x_optima   = L_x_optima[self._func_id]
            self.equation   = L_equation[self._func_id]
            self.f_name     = _prefix + " : " + L_f_name[self._func_id]
            self._name      = self.f_name
        except Exception as e:
            _msg = '''
                   We bound the parameter self._func_id between 0 to 5 
                   If you want more test functions, 
                   please add the function in the class Arbitrary_function, line 80, plot_function.py  
                   Program Terminated 
                   '''
            print(_msg)
            DBG.dbg(e)
            exit()

        self.op_func    = self.L_op_func[self._func_id]

    def __call__(self, sol):
        _ret = self.op_func(sol)
        return _ret

    def Xin_She_Yang_N4(self, sol):
        x = sol; y = 0
        _ret = 2.0 + (pow(np.sin(x), 2) - np.exp(-pow(x, 2))) * np.exp(-pow(np.sin(np.sqrt(np.abs(x))), 2.0))
        return _ret

    def Salomon(self, sol):
        x = sol; y = 0
        _ret = 1 - np.cos(2 * np.pi * np.abs(x)) + 0.1 * np.abs(x)
        return _ret

    def Drop_Wave(self, sol):
        x = sol; y = 0
        _ret = 1.0 - (1.0 + np.cos(12.0 * np.sqrt(pow(x, 2) + pow(y, 2))))/(0.5 * (pow(x, 2) + pow(y, 2)) + 2.0)
        return _ret

    def Schaffel_N2(self, sol):
        x = sol; y = 0
        _ret = 0.5 + (pow(np.sin(pow(x, 2) - pow(y, 2)), 2) - 0.5)/pow((1 + 0.001 * (pow(x, 2) + pow(y, 2))), 2)
        return _ret

    def Gamma_sin01(self, sol):
        x = sol; y=0
        _ret = gamma(x) + 2.0 * np.sin(np.pi * x) + 1.0
        return _ret


class Quantized_function:
    def __init__(self, c_plot_function):
        self.args   = c_plot_function.args
        self.r_min  = c_plot_function.r_min
        self.r_max  = c_plot_function.r_max
        self.increments     = c_plot_function.increments
        self.test_function  = c_plot_function.op_function

        # Set Quantization
        self.Q_obj = c_Lib.quantization(_iqp=self.args.initial_quantization_power)
        self.c_lib = c_Lib.function_library(r_min=self.r_min, r_max=self.r_max, _increments=self.increments)
        self.c_lib.set_params(_x        =self.args.initial_point,
                              _obj_func =self.test_function,
                              _qfunction=self.Q_obj,
                              b_init    =True)
        # For Virtual function : H(x -a)^2 + c(x-a) + b
        self._H = 0.125
        self._a = 0.0
        self._b = 0.0
        self._c = 0.0
    def __call__(self, _f_x):
        Q_result = self.Q_obj(_f_x)
        return Q_result

    # For Virtual Function for Quantization Band
    # f(x) = H(x - a)^2 + b
    def V_0(self, sol):
        _x = sol - self._a
        _ret    = self._H * pow(_x, 2) + self._c * _x + self._b
        return _ret
    def set_Virtual_Function(self, _H, _a, _b, _c):
        self._H = _H
        self._a = _a
        self._b = _b
        self._c = _c

    def first_derivative(self, _func, _x):
        # Check the https://github.com/maroba/findiff for finddiff
        # example of sci.misc.derivative
        _out = derivative(func=_func, x0=_x, dx=1e-6, n=1)
        return _out

class plot_function:
    def __init__(self, args=None):
        self.args       = args
        self.func_id = self.args.function

        self.increments = 0.01
        # Function Parameter
        self.band       = 3.0
        # Function Setting
        _local_func_id  = self.func_id - 2
        self.function   = []
        self.function.append(test_multimodal())
        self.function.append(parabolic_wash_board())
        self.function.append(Arbitrary_function(_local_func_id))

        # Domain Parameter
        if args.domain is not None:
            self.r_min      = self.args.domain[0]
            self.r_max      = self.args.domain[1]
        else:
            # For CEC 2022 Test functions
            if self.func_id > 6:
                self.r_min = self.function[2].op_func.plot_range[0]
                self.r_max = self.function[2].op_func.plot_range[1]
                self.increments = self.function[2].op_func.plot_unit
            # For Local Test functions
            else:
                self.r_min      = -6.0
                self.r_max      = 6.0
                #self.func_id    = 0
                print("Warning : Domain is not arranged !!!")
                if self.func_id == 6:
                    self.r_min = 0.2
                    self.r_max = 4.0
                else: pass

        # Plotting Parameter
        self.op_function= self.function[self.func_id if self.func_id <= 2 else 2]

        self.range      = self.gen_funcIO()

        self.ymin       = np.min(self.range)
        self.ymax       = np.max(self.range)
        self.xmin       = np.argmin(self.range) * self.increments + self.r_min
        self.xmax       = np.argmax(self.range) * self.increments + self.r_min
        print("Function : ", self.op_function.f_name)
        print("min : %f @%5.2f max : %f @%5.2f " %(self.ymin, self.xmin, self.ymax, self.xmax))

        # Set Quantization
        self.c_Qf = Quantized_function(c_plot_function=self)

    def H_0(self, sol):
        _ret = self.function[self.func_id].H_0(sol) if self.func_id == 1 else 0.0
        return _ret
    def gen_funcIO(self):
        self.domain = np.arange(self.r_min, self.r_max, self.increments)
        # For CEC 2022 Test functions
        if self.func_id > 6:
            X, Y = np.meshgrid(self.domain, self.domain)
            mesh_x   = np.dstack([X, Y])
            mesh_out = np.apply_along_axis(self.op_function, 2, mesh_x)
            main_idx = int(np.shape(mesh_out)[0]/2)
            ret_range= mesh_out[main_idx][:]
        else:
            ret_range = self.op_function(self.domain)
        # For Local Test functions
        return ret_range
    '''
    def H_1p(self, sol):
        _ret = 2.0 * np.sin(self.band * sol) + 2.0
        return _ret

    def function(self, sol):
        #_ret = 0.125 * pow(sol, 2) + 2.0 * np.sin(self.band * sol) + 2.0
        _ret    = self.H_0(sol) + self.H_1p(sol)
        return _ret
    '''
    # Input is a point
    def get_point(self, _x):
        _function   = self.op_function
        _out        = _function(_x)
        return _out

    # 2024-0925
    def set_virtual_function_for_quantization(self, _H=0.5):
        # self.c_Qf
        print("Virtual Function : H(x-a)^2 + c(x-a) + b")
        # We recommned _a=3.347
        _a = self.args.initial_point
        _b = self.get_point(_x=_a)
        _c = -1.0 * self.c_Qf.first_derivative(_func=self.op_function, _x=_a)
        self.c_Qf.set_Virtual_Function(_H=_H, _a=_a, _b=_b, _c=_c)
        print("  a : %f  b : %f  c : %f  H : %f" % (_a, _b, _c, _H))
    def derivative_plot(self):
        h_level = [[3.544, 5.0], [4.632, 5.0], [5.906, 5.0]]
        l_level = [2.57,  1.0]
        q_point = [3.084, 3.0]

        # high level
        plt.axhline(y=h_level[0][1], color='r', linestyle='--')
        plt.axvline(x=h_level[0][0], color='r', linestyle='--')
        plt.axvline(x=h_level[1][0], color='r', linestyle='--')
        plt.axvline(x=h_level[2][0], color='r', linestyle='--')
        # low level
        plt.axhline(y=l_level[1], color='r', linestyle='--')
        plt.axvline(x=l_level[0], color='r', linestyle='--')
        # quantization level
        plt.axhline(y=3, color='g', linestyle='--')
        y_pt    = 0.45 #q_point[1]/(self.ymax - self.ymin)
        plt.axvline(x=q_point[0], ymax=y_pt, color='b', linestyle='--')

        plt.plot(self.domain, self.range)
        plt.ylabel('$f(x) \in \mathbf{R}$')
        plt.xlabel('$x \in \mathbf{R}$')
        plt.show()

    def print_function_list(self):
        for _k, _func in enumerate(self.function):
            _func_name = _func.__class__.__name__
            print("[%2d] %s" %(_k, _func_name))

            if _k == 2:
                l_sub_func_name = _func.L_f_name
                for _j, _sub_func_name in enumerate(l_sub_func_name):
                    _sub_index = _k + _j
                    print("[%2d] %s" %(_sub_index, _sub_func_name))
            else:
                pass

        return 0
    def plot_function(self, _legend=True):
        _x = self.domain
        _f = self.range

        if self.func_id == 1:
            _h0 = self.H_0(_x)
            plt.plot(_x, _h0, 'r--', label='Approximated Hamiltonian $H_0$')
        elif self.func_id == 6:
            _h0 = self.c_Qf.V_0(sol=_x)
            _Q0 = self.c_Qf(_f_x=_f)
            plt.plot(_x, _h0, 'r--', label='Virtual Objective Function $\\tilde{f}$')
            plt.plot(_x, _Q0, 'C5--', label='Quantization $Q_p=$%.2f' %self.c_Qf.Q_obj._QP)
        else:
            pass

        plt.plot(_x, _f, label='Hamiltonian $H$')
        if _legend:
            # plt.legend([_f, _h0], ['Hamiltonian $H$', 'Approximated Hamiltonian $H_0$'])
            plt.legend()
        else : pass

    def operation(self, _simple_function=False):
        if self.args.print_function_list:
            self.print_function_list()
        else:
            if self.args.test_code :
                print("No test code. Please checj -ts option")
                pass
            else:
                self.set_virtual_function_for_quantization()
                self.plot_function(_legend=self.args.represent_legend)
            plt.show()

# =================================================================
# Main Routine
# =================================================================
if __name__ == "__main__":
    args    = ArgumentParse(_description, L_Param=None)
    try:
        c_plt   = plot_function(args)
    except Exception as e:
        print("Error Occur!! Program is terminated by exit : Error Msg : %s" %e)
        exit()

    c_plt.operation()

    print("===================================================")
    print("Process Finished ")
    print("Plotting function for NIPS 2025")
    print("===================================================")