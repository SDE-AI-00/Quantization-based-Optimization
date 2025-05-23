###########################################################
# func_plot.py
# plotting of function Test 
###########################################################
_description = '''\
====================================================
func_plot.py
====================================================
Example : python func_plot.py
'''
import numpy as np
import matplotlib.pyplot as plt

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

# For Plotting 
_q_step = 1.0/1024.0
lLimit  = [-0.05, 20.0]

if __name__ == '__main__':
    c_T = Temperature(base=10, index=-6)
    x = np.arange(lLimit[0], lLimit[1], _q_step)
    
    _inf = c_T.obj_function(x, c_T.T)
    _sup = c_T.obj_function(x, c_T.inf_sigma)
    
    plt.plot(x, _sup, 'r', label="sup \\bar{h}(t)")
    plt.plot(x, _inf, 'b', label="inf \\bar{h}(t)")
    plt.grid()
    plt.legend(loc=4)
    plt.show()
    
    print("==========================================")
    print("Processing is finished")
    print("==========================================")
