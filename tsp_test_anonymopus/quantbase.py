#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################
# quantbase.py
# Quantization Class for TSP
###########################################################
_description = '''\
====================================================
quantbase.py : Base class and functions for Quantized relation for TSP 
====================================================
Example : python qtest-01.py
'''
import numpy as np

class Quantization:
    def __init__(self):
        self.Q_base = 2
        self.Q_eta = 1
        self.Q_index = 0
        self.Q_param = self.eval_QP(self.Q_index)

    def eval_QP(self, index):
        _index = index if index > -1 else 0
        _res = self.Q_eta * pow(self.Q_base, _index)
        return _res

    def set_QP(self, index=-1):
        _index = self.Q_index if index == -1 else index
        self.Q_param = self.eval_QP(_index)

    def get_QP(self):
        return self.Q_param

    def get_Quantization(self, X):
        _X = X if isinstance(X, np.ndarray) else np.array(X)
        _X1 = self.Q_param * _X + 0.5
        _X2 = np.floor(_X1)
        _res = (1.0 / self.Q_param) * _X2

        return _res


class Temperature:
    def __init__(self, base, index):
        self._C = pow(base, index)  # Hyper parameter
        self._alpha = 20.0  # Speed control
        self._dim = 1  # Dimension of input/data
        self._eta = 1  # must be 1

    def inf_sigma(self, t):
        _res = self._C / np.log(t + 2)
        return _res

    def T(self, t):
        _pow = self._alpha / (t + 2)
        _res = np.power(2, 2.0 * _pow) * self.inf_sigma(t)
        return _res

    def obj_function(self, x, _func):
        _beta = self._dim / (24.0 * pow(self._eta, 2))
        _res = 0.5 * np.log2(_beta / _func(x))
        return _res


class Quantized_func:
    # ================================================================================
    # Quantization
    # ================================================================================
    def __init__(self, args):
        # Qunatrization Init value
        self.bQuantization   = False
        self.c_qtz           = Quantization()
        self.c_tmp           = Temperature(base=10, index=-6)
        self.l_index_trend   = []
        self.index_limit     = 16
        self.QuantMethod     = args.quantize_method
        self.l_infindextrend = []

    def Quantization(self, x):
        if self.bQuantization:
            Xq = self.c_qtz.get_Quantization(x)
        else:
            Xq = x
        return Xq, x

    def Increment_Index(self, _Xq, _Xf, _index):
        _res = _Xq
        while True:
            if _res.sum() == 0:
                if _index > self.index_limit:
                    break
                else:
                    _index += 1
                    self.c_qtz.set_QP(index=_index)
                    _res = self.c_qtz.get_Quantization(_Xf)
            else:
                break

        return _res, _index

    def Limited_Index(self, step, _index, _infindex):
        if self.QuantMethod > 1:
            _infindex = self.c_tmp.obj_function(step, self.c_tmp.T)
            _infindex = 0 if _infindex < 0 else _infindex
            while True:
                if _infindex > _index:
                    _index += 1
                else:
                    break
        else:
            pass
        return _index, _infindex

    # Xq : [DX]     Quantized dx^Q = [lm * h]
    # Xf : [dXf]    Floating  dx   = lm * h
    def Adv_quantize(self, Xq, Xf, step):
        _res = Xq
        if self.bQuantization and self.QuantMethod > 0:
            _index = self.l_index_trend[-1]
            _infindex = self.l_infindextrend[-1]
            if step == 0:
                pass
            else:
                if self.QuantMethod > 0:
                    _res, _index = self.Increment_Index(_res, Xf, _index)
                    _index, _infindex = self.Limited_Index(step, _index, _infindex)
                else:
                    pass
            self.l_index_trend.append(_index)
            self.l_infindextrend.append(_infindex)
        else:
            pass

        return _res

    '''
    # 2021-08-30 Jinwuk
    # Line Search 에 의해 Step Size가 적절하지 않게 구해지는 경우 보ㅓ상하기 위한 것
    # 그러나 잘 작동하지 않는다. 나중에 Trust Region 이나 다른 방법론에 의해 해결하는 것이 옳다.
    def Emergency_Solution(self, _h, DX, _Active=False):
        _res = DX
        if self.bQuantization and self.QuantMethod > 0:
            _index = self.l_index_trend[-1] if len(self.l_index_trend) > 0 else 0
            _norm_h = np.linalg.norm(_h)
            _norm_dxq = np.linalg.norm(DX)
            _condition = _index >= self.index_limit \
                         and _norm_h > self.stop_condition * 10.0 \
                         and _norm_dxq < self.stop_condition \
                         and _Active
            if _condition:
                DBG.dbg("Emegrncy Problem", _active=False)
                _QP = self.c_qtz.get_QP()
                _atomic_h = _h * 0.45
                _res = self.c_qtz.get_Quantization(_atomic_h)
            else:
                pass
        else:
            pass

        return _res
    '''
