###########################################################
# objectfunction.py
# Class Name : nlp_functions
###########################################################
import numpy as np
import scipy.optimize as opt
from autograd import grad

import pybenchfunction as bench
import random as rd

class nlp_functions :

    # initial_info[0] : _fix    : fixed initial or random initial (True or False)
    # initial_info[1] : _given  : given initial or generated initial
    def __init__(self, x, _initial_info, _func_id):    
        # function parameter
        self.function_name   = []
        self.objectfunction  = []   # Function itself 
        self.Diff_ob_function= []   # Differentiation of the function (= Gradient)
        self.functionobject  = []   # Object itself 
        self._func_id        = _func_id        
        
        # Operation Stability
        self.bImmediateBreak = False
        self.minvalue        = -99999999999999.0
        self.maxvalue        = 99999999999999.0
        self.input_dim       = np.shape(x)[0]
        self.initial_point   = x
        self._optimal_point  = []
        self.bound           = 0  

        # functions
        self.Init_Rosenblatt()
        self.Init_Ackley()
        self.Init_Whitley()
        self.Init_Rosen_M()
        self.Init_EggHolder()
        self.Init_XinSheYangN4()
        self.Init_Salomon()
        self.Init_DropWave()
        self.Init_Powell()
        self.Init_SchaffelN2()
        
        # Setting 
        _fix    = _initial_info[0]
        _given  = _initial_info[1]
        self.Set_Initial_Point(_initial_info)
        self.num_functions  = len(self.function_name)

#==========================================================
# ID : 0  [Rosenblatt Function]  
# $f(x) = \sum_{i=1}^{n-1} (b(x_{i+1} - x_i^2)^2 + (a - x_i)^2)$ b=100, a=1
#==========================================================
    def Init_Rosenblatt(self):
        # Set Function 
        self.Rosenbrock_function = bench.function.Rosenbrock(self.input_dim)

        # Set Optimal Point and Value
        _optimal_pt     = np.ones(self.input_dim, dtype=np.float)
        _optimal_val    = self.Rosenblatt(_optimal_pt)
        self._optimal_point.append((_optimal_pt, _optimal_val))

        # Set Function Parameter
        self.function_name.append('Rosenblatt')
        self.objectfunction.append(self.Rosenblatt)
        self.Diff_ob_function.append(self.Diff_Rosenblatt)
        #self.functionobject.append(opt.rosen)
        self.functionobject.append(self.Rosenbrock_function)

    def Rosenblatt(self, x):
        #_res = opt.rosen(x)
        _res = self.Rosenbrock_function(x)
        return _res

    def Diff_Rosenblatt(self, X):
        _g = grad(self.Rosenblatt)
        return _g(X)

#==========================================================
# ID : 1  [Ackley Function]
#==========================================================
    def Init_Ackley(self):
        # Set Function 
        self.Ackley_function = bench.function.Ackley(self.input_dim)

        # Set Optimal Point and Value
        _gminimum = self.Ackley_function.get_global_minimum(self.input_dim)
        self._optimal_point.append(_gminimum)

        # Set Function Parameter
        self.function_name.append(self.Ackley_function.name)
        self.objectfunction.append(self.Ackley)
        self.Diff_ob_function.append(self.Diff_Ackley)
        self.functionobject.append(self.Ackley_function)

    def Ackley(self, x):
        _res      = self.Ackley_function(x)
        return _res

    def Diff_Ackley(self, X):
        _g = grad(self.Ackley)
        return _g(X)

#==========================================================
# ID : 2  [Whitley Function]
#==========================================================
    def Init_Whitley(self):
        # Set Function 
        self.Whitley_function = bench.function.Whitley(self.input_dim)

        # Set Optimal Point and Value
        _gminimum = self.Whitley_function.get_global_minimum(self.input_dim)
        self._optimal_point.append(_gminimum)

        # Set Function Parameter
        self.function_name.append(self.Whitley_function.name)
        self.objectfunction.append(self.Whitley)
        self.Diff_ob_function.append(self.Diff_Whitley)
        self.functionobject.append(self.Whitley_function)

    def Whitley(self, x):
        _res      = self.Whitley_function(x)
        return _res

    def Diff_Whitley(self, X):
        _g = grad(self.Whitley)
        return _g(X)

#==========================================================
# ID : 3  [Eggholder Function]
#==========================================================
    def Init_EggHolder(self):
        # Set Function 
        self.EggHolder_function = bench.function.EggHolder(self.input_dim)

        # Set Optimal Point and Value
        _gminimum = self.EggHolder_function.get_global_minimum(self.input_dim)
        self._optimal_point.append(_gminimum)

        # Set Function Parameter
        self.function_name.append(self.EggHolder_function.name)
        self.objectfunction.append(self.EggHolder)
        self.Diff_ob_function.append(self.Diff_EggHolder)
        self.functionobject.append(self.EggHolder_function)

    def EggHolder(self, x):
        _res      = self.EggHolder_function(x)
        return _res

    def Diff_EggHolder(self, X):
        _g = grad(self.EggHolder)
        return _g(X)

#==========================================================
# ID : 4  [Rosenblatt Modification Function]
#==========================================================
    def Init_Rosen_M(self):
        # Force to Dimension for Rosen_M 
        _input_dim = 2
        self.Rosen_M_function = bench.function.Rosenbrock_Modification(_input_dim)

        # Set Optimal Point and Value
        _gminimum = self.Rosen_M_function.get_global_minimum(_input_dim)
        self._optimal_point.append(_gminimum)

        # Set Function Parameter
        self.function_name.append(self.Rosen_M_function.name)
        self.objectfunction.append(self.Rosen_M)
        self.Diff_ob_function.append(self.Diff_Rosen_M)
        self.functionobject.append(self.Rosen_M_function)

    def Rosen_M(self, x):
        _res = self.Rosen_M_function(x)
        return _res

    def Diff_Rosen_M(self, X):
        _g = grad(self.Rosen_M)
        return _g(X)

#==========================================================
# ID : 5  [XinSheYangN4 Function]
#==========================================================
    def Init_XinSheYangN4(self):
        # Get the function
        self.XinSheYangN4 = bench.function.XinSheYangN4(self.input_dim)

        # Set Optimal Point and Value
        _gminimum = self.XinSheYangN4.get_global_minimum(self.input_dim)
        self._optimal_point.append(_gminimum)

        # Set Function Parameter
        self.function_name.append(self.XinSheYangN4.name)
        self.objectfunction.append(self.XinSheYangN4)
        self.Diff_ob_function.append(self.Diff_XinSheYangN4)
        self.functionobject.append(self.XinSheYangN4)

    def XinSheYangN4(self, x):
        _res = self.XinSheYangN4(x)
        return _res

    def Diff_XinSheYangN4(self, X):
        _g = grad(self.XinSheYangN4)
        return _g(X)

#==========================================================
# ID : 6  [Salomon Function]
#==========================================================
    def Init_Salomon(self):
        # Get the function
        self.Salomon = bench.function.Salomon(self.input_dim)

        # Set Optimal Point and Value
        _gminimum = self.Salomon.get_global_minimum(self.input_dim)
        self._optimal_point.append(_gminimum)

        # Set Function Parameter
        self.function_name.append(self.Salomon.name)
        self.objectfunction.append(self.Salomon)
        self.Diff_ob_function.append(self.Diff_Salomon)
        self.functionobject.append(self.Salomon)

    def Salomon(self, x):
        _res = self.Salomon(x)
        return _res

    def Diff_Salomon(self, X):
        _g = grad(self.Salomon)
        return _g(X)

#==========================================================
# ID : 7  [DropWave Function]
#==========================================================
    def Init_DropWave(self):
        # Get the function
        self.DropWave = bench.function.DropWave(self.input_dim)

        # Set Optimal Point and Value
        _gminimum = self.DropWave.get_global_minimum(self.input_dim)
        self._optimal_point.append(_gminimum)

        # Set Function Parameter
        self.function_name.append(self.DropWave.name)
        self.objectfunction.append(self.DropWave)
        self.Diff_ob_function.append(self.Diff_DropWave)
        self.functionobject.append(self.DropWave)

    def DropWave(self, x):
        _res = self.DropWave(x)
        return _res

    def Diff_DropWave(self, X):
        _g = grad(self.DropWave)
        return _g(X)

#==========================================================
# ID : 8  [Powell Function]
#==========================================================
    def Init_Powell(self):
        # Get the function
        self.Powell = bench.function.Powell(self.input_dim)

        # Set Optimal Point and Value
        _gminimum = self.Powell.get_global_minimum(self.input_dim)
        self._optimal_point.append(_gminimum)

        # Set Function Parameter
        self.function_name.append(self.Powell.name)
        self.objectfunction.append(self.Powell)
        self.Diff_ob_function.append(self.Diff_Powell)
        self.functionobject.append(self.Powell)

    def Powell(self, x):
        _res = self.Powell(x)
        return _res

    def Diff_Powell(self, X):
        _g = grad(self.Powell)
        return _g(X)

#==========================================================
# ID : 9  [SchaffelN2 Function]
#==========================================================
    def Init_SchaffelN2(self):
        # Get the function
        self.SchaffelN2 = bench.function.SchaffelN2(self.input_dim)

        # Set Optimal Point and Value
        _gminimum = self.SchaffelN2.get_global_minimum(self.input_dim)
        self._optimal_point.append(_gminimum)

        # Set Function Parameter
        self.function_name.append(self.SchaffelN2.name)
        self.objectfunction.append(self.SchaffelN2)
        self.Diff_ob_function.append(self.Diff_SchaffelN2)
        self.functionobject.append(self.SchaffelN2)

    def SchaffelN2(self, x):
        _res = self.SchaffelN2(x)
        return _res

    def Diff_SchaffelN2(self, X):
        _g = grad(self.SchaffelN2)
        return _g(X)

#==========================================================
# Service Functions 
#==========================================================
    def Set_Function_ID(self, Function_ID):
        self._func_id    = Function_ID

    def Get_Function_ID(self):
        return self._func_id

    def Set_Initial_Point(self, _initial_info):
        _fix    = _initial_info[0]
        _given  = _initial_info[1] 
        _res = self._fix_initial_point(_given) if _fix else self._random_initial_point()
        return _res

    def Get_Initial_Point(self):
        return self.initial_point

    def _fix_initial_point(self, _given_init):
        # As for 2-Dimension, we set the initial point outside of the function.
        if self.input_dim < 2:
            print("Dimension is less than 1. It is abnormal case!! We have to go outside of the program")
            exit()
        else:
            if _given_init is False :
                self.initial_point = np.zeros(self.input_dim, dtype=np.float)
                for _k in range(self.input_dim):
                    self.initial_point[_k] = -1.212 if (_k % 2)==0 else 1.321
            else:                    
                pass

    def _random_initial_point(self):
        self.initial_point = np.zeros(self.input_dim, dtype=np.float)
        _IdomainVec = self.Get_Function_Input_Domain()
        for _k in range(self.input_dim):
            _Idomain= _IdomainVec[_k]
            # r = rd.random() returns  0.0 ~ 1.0  and x = min(_Idomain),  y = max(_Idomain) then o = x + r(y-x)
            self.initial_point[_k] = min(_Idomain) + rd.random() * (max(_Idomain) - min(_Idomain))

#==========================================================
# Common Function
#==========================================================
    def Optimal_Point(self):
        _optimal_pt = self._optimal_point[self._func_id]
        return _optimal_pt[0]

    def Objective_Function(self, x):
        _function_value = self.objectfunction[self._func_id](x)
        self.bImmediateBreak = abs(_function_value) > self.maxvalue
        return _function_value

    def Diff_Objective_Function(self, x):
        return self.Diff_ob_function[self._func_id](x)

    def _get_break(self):
        if self.bImmediateBreak:
            print("**********************************************************")
            print("Algorithm is Broken !!!! Objective function is Diverge !!!")
            print("**********************************************************")
        return self.bImmediateBreak

    def Get_Input_Dimension(self):
        return self.input_dim

    def Get_Function_Name(self):
        return self.function_name[self._func_id]

    def Get_All_Function_Name(self):
        return self.function_name

    def Get_Optimal_Value_and_Point(self):
        return self._optimal_point[self._func_id]

    def Get_Function_Object(self):
        return self.functionobject[self._func_id]        

    def Get_Function_Input_Domain(self):
        _fo = self.Get_Function_Object()
        return _fo.input_domain

    def Get_num_Functions(self):
        return self.num_functions

# =================================================================
# Test Processing
# =================================================================
if __name__ == "__main__":
    from nlp_class02 import NLP_Class02
    #from nlp_functions import nlp_functions as nf
    import nlp_service as ns
    # =================================================================
    # Service Function 
    # =================================================================
    def object_function(X):
        return cFunctions.Objective_Function(X)

    def Calculate_Gradient(X):
        return cFunctions.Diff_Objective_Function(X)

    # =================================================================
    # Pre-Setting
    # =================================================================
    _dimension      = 2  # 2 : Default Dimension
    _Function_ID    = 1  # 0 : Rosenblatt Function
    
    # =================================================================
    # Dummy Setting
    # =================================================================
    Initial_point   = np.array(np.zeros(_dimension), dtype=np.float64)
    init_X          = np.array(Initial_point, dtype=np.float)  # Main Training Parameter
    initial_info    = []
    initial_info.append(True)           # Fixed Initial
    initial_info.append(Initial_point)  # Fixed Initial point
    # =================================================================
    # Set Object Function and Differentiation 
    # =================================================================
    # Definition of Class
    #cFunctions  = nf(init_X)
    cFunctions  = nlp_functions(init_X, _initial_info=Initial_point, _func_id=_Function_ID)

    # Set Inital Point for test function 
    init_X      = cFunctions.Get_Initial_Point()

    # Description of Object_Function() and Gradient Parameter
    inference    = object_function
    GlobalMin    = inference(cFunctions.Optimal_Point())
    initial_cost = inference(init_X)
    init_gr      = Calculate_Gradient(init_X)

    print("=================================================================")
    print("Input Dimension    : ", cFunctions.Get_Input_Dimension())
    print("Benchmark Function : ", cFunctions.Get_Function_Name())
    print("Minimum point value: ", cFunctions.Optimal_Point(), GlobalMin)
    print("Initial point value: ", init_X, initial_cost)
    print("Initial Gradient   : ", init_gr)
    print("=================================================================")


    print("=================================================================")
    print("Processing is finished")
    print("=================================================================")