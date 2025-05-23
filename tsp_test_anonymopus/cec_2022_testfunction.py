#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - CEC 2022 Test Functions
# Working Directory : ..\tsp_test
# Base URL     : https://www.kaggle.com/code/kooaslansefat/cec-2022-benchmark
###########################################################################
_description = '''\
====================================================
cec_2022_testfunction.py : Only Functions 
====================================================
Example : python cec_2022_testfunction.py -f 2 -d 1
'''
import numpy as np
import matplotlib.pyplot as plt
import argparse, textwrap
import my_debug as DBG


class get_cec2022_test:
    def __init__(self):
        self.test_functions = []
        self.test_functions.append(Bent_Cigar_Function())
        self.test_functions.append(Rastrigins_Function())
        self.test_functions.append(High_Conditioned_Elliptic_Function())
        self.test_functions.append(HGBat_Function())
        self.test_functions.append(Rosenbrocks_Function())
        self.test_functions.append(Griewank_Function())
        self.test_functions.append(Ackleys_Function())
    def __call__(self):
        return self.test_functions

class cec_2022_test:
    def __init__(self, _func_ID):
        # Construct the Dictionary of Functions
        self.test_functions = []
        c_cec2022_testfunc  = get_cec2022_test()
        self.test_functions = c_cec2022_testfunc()

        # Set CEC_test function
        self.cec_functions  = self.test_functions[_func_ID]

        # Set Plotting Range
        self.plot_range_min = self.cec_functions.plot_range[0]
        self.plot_range_max = self.cec_functions.plot_range[1]

        # Plotting Default Parameter
        self.input_dimension= 2
        self.plot_projection= '3d'
        self.cmap           ='plasma'
        self.colorbar_shrink= 0.5
        self.colorbar_aspect= 5

        # Print Summary
        print("---------------------------------------------------")
        print("Test Function : [%2d] %s" %(_func_ID, self.cec_functions.name))
        print("  Equation    : ", self.cec_functions.equation )
        print("  Minimum     : [$f(x^*) = %f$] at x^* = " %self.cec_functions.minimum_info['min_value'],
                                                           self.cec_functions.minimum_info['min_point'])
        print("===================================================")

    def Plot_Result_for_2D_Input(self, _unit=0.1):
        t_f = self.cec_functions

        x_min, x_max = self.plot_range_min, self.plot_range_max
        y_min, y_max = self.plot_range_min, self.plot_range_max

        # Create a meshgrid of points within the specified ranges
        X, Y = np.meshgrid(np.arange(x_min, x_max, _unit), np.arange(y_min, y_max, _unit))

        # Make 2-Dimension Input Vector (x, y) on the plane generated by np.meshgrid.
        # Each coordinate of the plane represents the input vector (x,y)
        mesh_x = np.dstack([X, Y])

        # Calculate the values of the function at each point in the meshgrid
        Z = np.apply_along_axis(t_f, self.input_dimension, mesh_x)

        # The above code is equivalent to the following :
        '''
        Z = np.zeros(np.shape(X))
        for _i, _X in enumerate(X):
            for _j, _Y in enumerate(Y):
                _x = mesh_x[_i][_j]
                _z = t_f(_x)
                Z[_i][_j] = _z
        '''
        # Create a 3D plot of the function
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=self.plot_projection)
        surf = ax.plot_surface(X, Y, Z, cmap=self.cmap)
        fig.colorbar(surf, shrink=self.colorbar_shrink, aspect=self.colorbar_aspect)

        _title = t_f.name + " : " + t_f.equation
        plt.title(_title, size=10)
        plt.show()

# =================================================================
# CEC Test Functions
# =================================================================
class Bent_Cigar_Function:
    def __init__(self):
        self.name       = self.__class__.__name__
        self.equation   = "$f_{1}(x) = x_0^2 + 10^6 * \sum_{i=1}^{n-1} x_i^2$"
        self.plot_range = [-2, 2]
        self.plot_unit  = 0.1
        self.minimum_info = {'min_point': 0, 'min_value': 0}

    def __call__(self, _x):
        x_0 = _x[0]
        x_k = _x[1:]
        _result = x_0 + 1e6 * np.sum(x_k ** 2)
        return _result

class Rastrigins_Function:
    def __init__(self):
        self.name       = self.__class__.__name__
        self.equation   = "$f(x) = 10n + \sum_{i=1}^n (x_i^2 - 10 \cos (2 \pi x_i))$"
        self.plot_range = [-5.12, 5.12]
        self.plot_unit  = 0.1
        self.minimum_info = {'min_point': 0, 'min_value': 0}

        self.param_A    = 10

    def __call__(self, _x):
        A   = self.param_A
        d   = len(_x)
        # Old code
        #_sum= np.sum(_x[i]**2 - A * np.cos(2 * np.pi * _x[i]) for i in range(d))
        # New code
        gen = (x**2 - A * np.cos(2 * np.pi * x) for x in _x)
        _sum= np.sum(np.fromiter(gen, float))

        _result = d * A + _sum
        return _result

class High_Conditioned_Elliptic_Function:
    def __init__(self):
        self.name       = self.__class__.__name__
        self.equation   = "$f_{3}(\mathbf{x}) = \sum_{i=1}^n a^{\\frac{i-1}{d-1} } x_{i}^2$"
        self.plot_range = [-2, 2]
        self.plot_unit  = 0.01
        self.minimum_info = {'min_point': 0, 'min_value': 0}

        self.param_A    = 1e6

    def __call__(self, _x):
        a   = self.param_A
        d   = len(_x)
        _result = np.sum([a**((i-1)/(d-1))*(_x[i]**2) for i in range(d)])
        return _result

class HGBat_Function:
    def __init__(self):
        self.name       = self.__class__.__name__
        self.equation   = "$f(x_1, x_2) = \left(\sum_{i=1}^2(x_i-a_i)^2\\right)^2 \
                            + \left(\sum_{i=1}^2(x_i-a_i)\\right)^2 / 10^6$"
        self.plot_range = [-20, 20]
        self.plot_unit  = 1.0
        self.minimum_info = {'min_point': 0, 'min_value': 0}

        self.param_A    = [-10, -5]

    def __call__(self, _x):
        a = self.param_A
        term1 = np.sum((_x - a) ** 2) ** 2
        term2 = np.sum(_x - a) ** 2 / 1e6
        return term1 + term2

class Rosenbrocks_Function:
    def __init__(self):
        self.name       = self.__class__.__name__
        self.equation   = "$f(x,y) = 100(y-x^2)^2 + (1-x)^2$"
        self.plot_range = [-2, 2]
        self.plot_unit  = 0.15
        self.minimum_info = {'min_point': [1, 1], 'min_value': 0}

        self.param_A    = 100

    def __call__(self, _x):
        a = self.param_A
        return (a * (_x[1] - _x[0]**2)**2 + (1 - _x[0])**2)

class Griewank_Function:
    def __init__(self):
        self.name       = self.__class__.__name__
        self.equation   = "$f(x) = 1 + (\\frac{1}{4000}) \cdot \sum_{i=1}^{d}x_i^2 \
                                     - \prod_{i=1}^{d} \cos \left( \\frac{x_i}{\sqrt{i}} \\right)$"
        self.plot_range = [-20, 20]
        self.plot_unit  = 0.4
        self.minimum_info = {'min_point': 0, 'min_value': 0}

        self.param_A    = 4000.0

    def __call__(self, _x):
        a = self.param_A
        _gen_1 = (x**2 for x in _x)
        _sum_1 = np.sum(np.fromiter(_gen_1, float))
        _gen_2 = (np.cos(x/np.sqrt(_i+1)) for _i, x in enumerate(_x))
        _prod  = np.prod(np.fromiter(_gen_2, float))
        return (1.0 + 1.0/a * _sum_1 - _prod)

class Ackleys_Function:
    def __init__(self):
        self.name       = self.__class__.__name__
        self.equation   = "$f(x) = -20 \exp \left( -0.2 \sqrt{\\frac{1}{d} \sum_{i=1}^d x_i^2} \\right) \
                                   - \exp \left(\\frac{1}{d} \sum_{i=1}^d \cos( 2 \pi x_i) \\right) + 20 + e$"
        self.plot_range = [-5, 5]
        self.plot_unit  = 0.1
        self.minimum_info = {'min_point': 0, 'min_value': 0}

        self.param_A    = 20.0

    def __call__(self, _x):
        a = self.param_A
        d = float(len(_x))
        _gen_1 = (x**2 for x in _x)
        _sum_1 = np.sum(np.fromiter(_gen_1, float))
        _gen_2 = (np.cos(2 * np.pi * x) for x in _x)
        _sum_2  = np.sum(np.fromiter(_gen_2, float))
        _result= -a * np.exp(-0.2 * np.sqrt(1.0/d * _sum_1)) - np.exp(1.0/d * _sum_2) + a + np.exp(1)
        return _result


# =================================================================
# Main Test Routine
# =================================================================
def ArgumentParse(_intro_msg, L_Param, bUseParam=False):
    parser = argparse.ArgumentParser(
        prog='cec_2022_testfunction.py ',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(_intro_msg))

    parser.add_argument('-f', '--function', help="CEC Test Functions \\ \
                                                    [0] Bent_Cigar_Function \\ \
                                                    [1] Rastrigins_Function \\  \
                                                    [2] High_Conditioned_Elliptic_Function \\ \
                                                    [3] HGBat_Function \\ \
                                                    [4] Rosenbrocks_Function \\ \
                                                    [5] Griewank_Function \\ \
                                                    [6] Ackleys_Function", default=0, type=int)

    parser.add_argument('-d', '--domain',   help="Domain", nargs='+', default=None, type=float)
    parser.add_argument('-p', '--use_param',help="Use Parameter", default=0, type=int)
    args = parser.parse_args()

    args.use_param = True if args.use_param > 0 else False

    print(_intro_msg)
    print("Parameter:")
    print("   use_param : %s" %("True" if args.use_param else "False"))
    print("   function  : %s" %args.function)
    if args.domain != None:
        print("   domain    :", args.domain)
    else: pass
    print("====================================================")

    return args

if __name__ == "__main__":
    _args = ArgumentParse(_intro_msg=_description, L_Param=[], bUseParam=False)

    try :
        c_cec = cec_2022_test(_func_ID=_args.function)
    except Exception as e:
        print("Error Occur!! Function Index is not available. Check the number of -f option")
        print(e)
        exit()

    c_cec.Plot_Result_for_2D_Input()

    print("===================================================")
    print("Process Finished ")
    print("Plotting function for NIPS 2022")
    print("===================================================")


