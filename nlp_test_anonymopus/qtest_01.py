###########################################################
# qtest_01.py
# Quantization Test 
###########################################################
_description = '''\
====================================================
qtest-01.py : Based on torch module
Function ID 
0 : Rosenblatt Function
1 : Ackey Function
====================================================
Example : python qtest-01.py
'''
import numpy as np
import matplotlib.pyplot as plt

from nlp_functions import nlp_functions as nf
import nlp_service as ns

class qTest_Function:
    def __init__(self, initX, Function_ID=0, q_step=0.05):
        self.cFunctions     = nf(initX)
        self._Function_ID   = Function_ID

        self._inference     = self.object_function
        self._Dufferentual  = self.cFunctions.Diff_ob_function[self._Function_ID]
        self.Optimal_Point  = self.cFunctions.Optimal_Point(self._Function_ID)

        self.q_step_limit   = 0.00048828125
        self.q_step         = q_step
        self.lLimit         = [-2.0, 3.0, -2.0, 3.0]
        self.level_list     = [ \
            [2e-8, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384], \
            [2e-8, 2e-4, 2e-2, 2e-1, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.65, 2.75, 3, 4, 5, 6]]

        # Line Search (Golden Search)
        self.F = 0.618
        self.LimitofLineSearch = 20
        self.stop_condition = 0.0000003

    def object_function(self, X):
        return self.cFunctions.Objective_Function(X, self._Function_ID)

    def directional_derivative(self, X):
        _res = self.cFunctions.Diff_Objective_Function(X, self._Function_ID)
        return -1 * _res

    def get_objectFunction_name(self):
        return self.cFunctions.function_name[self._Function_ID]

    def LineSearch(self, x, h):
        a, b = x, x + h
        chk_cnt = 0
        while True:
            L           = b - a
            ai, bi      = a + (1 - self.F) * L, b - (1 - self.F) * L
            Pai, Pbi    = self._inference(ai), self._inference(bi)
            (a, b)      = (ai, b) if Pbi <= Pai else (a, bi)
            _dLenth     = ai - bi
            _epsilon    = np.linalg.norm(_dLenth)

            _bStopCondition = (chk_cnt >= self.LimitofLineSearch) or (_epsilon < self.stop_condition)
            if _bStopCondition: break
            else: chk_cnt = chk_cnt + 1

        xn = bi if Pbi <= Pai else ai

        #xn = self.Qunatization(xn)
        return xn, self._inference(xn)

    def Description_of_Object_Function(self):
        lLimit = self.lLimit
        
        _q_step = self.q_step if self.q_step > self.q_step_limit else self.q_step_limit
        print("q_step for Description_of_Object_Function : ", _q_step)
        x = np.arange(lLimit[0], lLimit[1], _q_step)
        y = np.arange(lLimit[2], lLimit[3], _q_step)

        X, Y = np.meshgrid(x, y)
        Z = self._inference(np.array([X, Y]))
        return X, Y, Z

    def plot_Qresult(self):
        levels = self.level_list[self._Function_ID]
        X, Y, Z     = self.Description_of_Object_Function()
        ymin, ymax  = np.min(Y), np.max(Y)
        xmin, xmax  = np.min(X), np.max(X)

        plt.contour(X, Y, Z, levels)
        plt.plot(self.Optimal_Point[0], self.Optimal_Point[1], 'r*', label="Optimal Point")

        plt.ylim((ymin, ymax))
        plt.xlim((xmin, xmax))
        plt.ylabel('Y-Coordinate')
        plt.xlabel('X-Coordinate')

        return plt 

    def plot_vector(self, _x):

        return plt     

# end of qTest_Function

class Quantization:
    def __init__(self):
        self.Q_base  = 2
        self.Q_eta   = 1
        self.Q_index = 0
        self.Q_param = self.eval_QP(self.Q_index)

    def eval_QP(self, index):
        _index = index if index > -1 else 0 
        _res = self.Q_eta * pow(self.Q_base, _index)
        return _res

    def set_QP(self, index=-1):        
        _index = self.Q_index if index==-1 else index
        self.Q_param = self.eval_QP(_index)

    def get_QP(self):        
        return self.Q_param 
            
    def get_Quantization(self, X):
        _X      = X if isinstance(X, np.ndarray) else np.array(X)
        _X1     = self.Q_param * _X + 0.5
        _X2     = np.floor(_X1)
        _res    = (1.0/self.Q_param) * _X2

        return _res

class Temperature:
    def __init__(self, base, index):
        self._C     = pow(base, index)   # Hyper parameter 
        self._alpha = 20.0           # Speed control 
        self._dim   = 1             # Dimension of input/data
        self._eta   = 1             # must be 1

    def inf_sigma(self, t):
        _res = self._C/np.log(t + 2)
        return _res 

    def T(self, t):
        _pow = self._alpha/(t + 2) 
        _res = np.power(2, 2.0 * _pow) * self.inf_sigma(t)
        return _res

    def obj_function(self, x, _func):
        _beta = self._dim/(24.0 * pow(self._eta, 2))
        _res  = 0.5 * np.log2(_beta/_func(x))
        return _res



# --------------------------------------------------------
# Parsing the Argument
# --------------------------------------------------------
import argparse
import textwrap

def _ArgumentParse(_intro_msg):
    parser = argparse.ArgumentParser(
        prog='qtest_01.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(_intro_msg))

    parser.add_argument('-f',  '--function', help="[0] Rosenblatt [1]Ackley", type=int, default=0)
    parser.add_argument('-i',  '--initial_point',help="Initial Point", nargs='+', default=None, type=float)
    parser.add_argument('-qi', '--QIndex', help="[default] 0  and Integer values must larger than 0", type=int, default=0)

    args = parser.parse_args()
    args.QIndex = 0  if args.QIndex < 1 else args.QIndex

    print(_intro_msg)
    return args

# =============================================================
# Test Processing
# =============================================================

if __name__ == '__main__':
    _args = _ArgumentParse(_description)

    c_qtz   = Quantization()
    c_qtz.set_QP(_args.QIndex)
    _q_step = 1.0/c_qtz.Q_param
    c_qtf   = qTest_Function(initX=_args.initial_point, Function_ID=_args.function, q_step=_q_step)

    print("==========================================")
    print("Function Name : {}".format(c_qtf.get_objectFunction_name()))
    print("Q_step        : {}".format(_q_step))    
    print("Qparameter(Qp): ", c_qtz.get_QP())

    plt     = c_qtf.plot_Qresult()

    _x      = np.array([_args.initial_point[0], _args.initial_point[1]])
    _xq     = c_qtz.get_Quantization(_x)
    gd      = c_qtf.directional_derivative(_xq)
    bp, _f  = c_qtf.LineSearch(_x, gd)
    bpq     = c_qtz.get_Quantization(bp)
    _fq     = c_qtf._inference(bpq)    

    print("Initial Point           : ", _x)
    print("Quantized Initial Point : ", _xq)
    print("Gradient                : ", gd)
    print("Best Point              : ", bp,  "f : ", _f)
    print("Quantized Best Point    : ", bpq, "f : ", _fq)
    print("==========================================")
    
    xline   = [_xq[0], gd[0]]
    yline   = [_xq[1], gd[1]]
    plt.plot(xline, yline, 'b', label="Trace")
    plt.plot(bp[0],  bp[1],  'g*', label="Optimal Point on Direction")
    plt.plot(bpq[0], bpq[1], 'y*', label="Quantized Optimal Point on Direction")
    plt.legend(loc=2)

    plt.show()
    f2 = plt.gcf()
    f2.savefig("result-001.pdf", format="pdf", dpi=600)

    print("==========================================")
    print("Processing is finished")
    print("==========================================")
