###########################################################
# nlpclass.py
# Class Name : NLP_Class02
###########################################################
import numpy as np
import math
import nlp_service as ns

import scipy.optimize as sop

#For Quantization 
from qtest_01 import Quantization
from qtest_01 import Temperature
import my_debug as DBG

class NLP_Class02 :

    def __init__(self, args):
        # Basic Function pointer
        self.stepfunc   = []
        self.searchfunc = []
        self.update_rule= []
        self.SearchAlgorithm = 0
        self.StepSizeRule= 0

        # Constant Step Size
        self.Constant_StepSize = 0.00212      ## Default 0.002

        # Time Decaying Step Size
        self.Epoch          = 0                 ## EPOCH Index
        self.AbsoluteTime   = 0                 ## Iteration including Inner Loop
        self.DecayingRate   = 1.0

        # For evaluation of Armijo'rule
        self.alpha = 0.55
        self.beta  = 0.612
        self.Initial_StepSize   = 0.9
        self.ArmijoDebugFreq    = 1.0
        self.ArmijoOneStepCnt   = 0
        self.ArmijoPrevCost     = 0.0
        self.VariableBeta       = 1.0
        self.BestBeta           = 0.0       ## For Moment Algorithm
        self.FastIdleGoControl  = 0.5       ## For Fast Algorithm Control SGD 0.7, Others are 0.5

        # Method of Conjugate Gradient
        self.CGmethod = 0                  # 0 : Polak - Riebel  1 : Fletcher-Reeves

        # Line Search (Golden Search)
        self.F = 0.618
        self.LimitofLineSearch = 20
        self.stop_condition = 0.0000003

        # Quasi Newton
        self.AlgorithmWeight = 0.0  # Weight of DFP if zero, it means that BFGS

        # AdaGrad
        self.sum_of_gradient = 0.0

        # AdaDelta/RMSProp
        self.expect_gradient = 0.0
        self.h_previous      = 1.0
        self.auxiliary_e     = 0.00002
        self.delta           = 0.9

        # ADAM
        self.beta_1          = 0.1          # Default 0.9
        self.beta_2          = 0.9          # default 0.999
        self.variable_beta_1 = 1.0
        self.variable_beta_2 = 1.0
        self.ADAM_epsilon    = 0.000000001  # 1e-8
        self.ADAM_FirstMoment= 0.0
        self.ADAM_SecondMoment= 0.0

        # SGD with Moment & Nestrov Acceleration
        self.moment_v        = 0.0
        self.moment_gamma    = 0.9                  # Default 0.9

        # Qunatrization
        self.bQuantization   = False
        self.c_qtz           = Quantization()
        self.c_tmp           = Temperature(base=10, index=-6)
        self.l_index_trend   = []
        self.index_limit     = 16
        self.QuantMethod     = args.quantize_method
        self.l_infindextrend = []
        
        # Operation Stability and Control
        self.bImmediateBreak    = False
        self.ArmijoInternalCounterLimit = 100
        self.minvalue           = -99999999999999.0
        self.maxvalue           = 99999999999999.0
        self._update            = False

        # Function Pointer
        self.stepfunc.append(self.ConstantStepSize)
        self.stepfunc.append(self.Armijo_Lambda)
        self.stepfunc.append(self.LineSearchStepsize)
        self.stepfunc.append(self.TimeDecayingStepsize)
        self.stepfunc.append(self.Armijo_onestep)
        self.stepfunc.append(self.Armijo_onestep)           ## Fast Armijo Idle and Go
        self.stepfunc.append(self.LineSearch_onestep)
        self.stepfunc.append(self.WolfLineSearch)

        self.searchfunc.append(self.Armijo_Gradient)
        self.searchfunc.append(self.Conjugate_Gradient)     ## Assume that PR CG
        self.searchfunc.append(self.Conjugate_Gradient)     ## Assume that FR CG 
        self.searchfunc.append(self.Quasi_Newton)
        self.searchfunc.append(self.AdaGrad)
        self.searchfunc.append(self.AdaDelta)
        self.searchfunc.append(self.RMSProp)
        self.searchfunc.append(self.ADAM)
        self.searchfunc.append(self.SGD_Moment)     # Original SGD_Moment
        self.searchfunc.append(self.SGD_Moment)     # Nestrov Acceleration : Code is shared

        # System Interface
        self.BenchmarkClass     = None              # Class for Benchmark
        self._inference         = None              # For Other Service 


    def _parameter_initialization(self, args, cost_function, grad_objf, Quasi_DFP_Weight, ConjugateMethod, SearchAlgorithm, StepSizeRule):
        """_parameter_initialization for Quasi Newton and Conjugate Method.
            :param Quasi_DFP_Weight: Weight of DFP method. 0 <= param <= 1.0
            :type  Quasi_DFP_Weight: float
            :param ConjugateMethod : Method of Conjugate Gradient
            :type  ConjugateMethod : int
        """
        self.AlgorithmWeight = Quasi_DFP_Weight
        bArmijoStepSize = (StepSizeRule == 1 or StepSizeRule == 4 or StepSizeRule == 5)

        if ConjugateMethod != None:
            self.CGmethod   = ConjugateMethod

        # Conjugate Gradient and Armio Rule
        if (SearchAlgorithm == 1 or SearchAlgorithm == 2) and bArmijoStepSize:
            self.alpha      = (lambda _y: 0.5 if _y == 0 else 0.4)(self.CGmethod)

        # Quasi Newton and Armijo Rule
        if SearchAlgorithm == 3 and bArmijoStepSize:
            self.alpha      = 0.000095

        # Gamma for SGD Moment and Nestrov 
        if (SearchAlgorithm == 7 or SearchAlgorithm == 8) and bArmijoStepSize:
            self.moment_gamma = args.MomentGamma

        if StepSizeRule == 5:
            self.FastIdleGoControl = args.FastParameter

        self.inference  = cost_function
        self.gradient   = grad_objf
        self.SearchAlgorithm = SearchAlgorithm
        self.StepSizeRule   = StepSizeRule

        return 0

    def _get_break(self):
        if self.bImmediateBreak:
            print("***********************************")
            print("Immediate STOP !!! Check the codes")
            print("***********************************")
        return self.bImmediateBreak

    def _parameter_tuning(self, X):
        # When the Nestrov Acceleration is applied :: [0 0 1 1 2 3 4 5 6 7 8] => 10번은 8이 됨
        if self.SearchAlgorithm == 8:
            Ga = self.moment_gamma
            _v = self.moment_v
            Xn = X - Ga * _v
        else:
            Xn = X

        return Xn
    #================================================================================
    # Cost Update Function (Very Important) ns.Algorithm_Info_buf[4]/[5]
    #================================================================================
    def _cost_update_onestep(self, _update, _pcost, _ccost, X, Xn, h, inference, Algorithm_Info):
        if Algorithm_Info == ns.Algorithm_Info_buf[4] or Algorithm_Info == ns.Algorithm_Info_buf[5]:
            if _update:
                Xe = X - self.Initial_StepSize * h
                r_pcost = _ccost
                r_ccost = inference(Xe)
            else:
                r_pcost = _pcost
                r_ccost = inference(Xn)
        else:
            r_pcost = _pcost
            r_ccost = _ccost

        _epsilon = r_pcost - r_ccost

        return _epsilon, r_pcost, r_ccost

    def _cost_update_general(self, _update, _pcost, _ccost, X, Xn, h, inference, Algorithm_Info):
        if Algorithm_Info != ns.Algorithm_Info_buf[4] and Algorithm_Info != ns.Algorithm_Info_buf[5]:
            r_pcost = _ccost
            r_ccost = inference(Xn)
        else:
            r_pcost = _pcost
            r_ccost = _ccost
            # Re-Check the immediate break condition 
            inference(X)

        _epsilon = r_pcost - r_ccost

        return _epsilon, r_pcost, r_ccost

    #================================================================================
    # Step Size Rules
    #================================================================================
    def ConstantStepSize(self, x, g, h, _cost, debugOn=True):
        return self.Constant_StepSize, True

    def Armijo_Core(self, beta_k, _Difference, grad, chk_cnt, chk_freq, _X, debugOn=True):
        Psi = - beta_k * grad  # -beta^k \cdot \alpha \| \nabla f(x_i) \|^2
        # Debug Code
        if (chk_cnt % chk_freq == 0) and debugOn:
            print("Armijo count: %4d Beta: %4.8f Diff: %4.8f  Psi: %4.8f  beta_k: %4.8f" %(chk_cnt, beta_k, _Difference, Psi, beta_k), " _X:", _X)
        # Main Processing
        if _Difference <= Psi:
            self.BestBeta   = beta_k          ## For Moment Algorithm
            bUpdate= True
        else:
            beta_k = beta_k * self.beta
            chk_cnt = chk_cnt + 1
            bUpdate = False

        return beta_k, chk_cnt, bUpdate
    #-------------------------------------------------------------
    #   param _cost : Differnce : \varepsilon_n = f_{n-1} - f_{n}
    # -------------------------------------------------------------
    def Armijo_Lambda(self, _x, _g, _h, _cost, debugOn=True):
        chk_freq    = self.ArmijoDebugFreq  # Set Debugging Frequency
        chk_cnt     = 0
        beta_k      = self.Initial_StepSize
        grad        = self.alpha * np.inner(_g, _h)
        _current    = self.ArmijoPrevCost - _cost

        while True:
            xe      = _x - beta_k * _h
            _next   = self.inference(xe)
            _Difference = _next - _current

            beta_k, chk_cnt, bUpdate = self.Armijo_Core(beta_k, _Difference, grad, chk_cnt, chk_freq, xe, debugOn)
            if bUpdate:
                Lambda = beta_k
                break
            else:
                if chk_cnt >= self.ArmijoInternalCounterLimit:
                    print("==========================================")
                    DBG.dbg("Armijo Rule is broken Some Problem Occur!!", _active=True)
                    print("==========================================")
                    beta_k, chk_cnt, bUpdate = self.Armijo_Core(beta_k, _Difference, grad, chk_cnt, chk_freq, xe, True)
                    self.bImmediateBreak = True
                    Lambda = beta_k
                    break

        self.ArmijoPrevCost = _current
        self.ArmijoOneStepCnt = chk_cnt

        return Lambda, True

    def Fast_Armijo_Initialize(self, chk_cnt, _inner_prod, _Df, _h, debugOn=True) :
        # This function is called only if the self.StepSizeRule == 4, 5
        # self.StepSizeRule == 4 is General Idle and Go
        # self.StepSizeRule == 5 is Fast Idle and Go
        if self.StepSizeRule == 5 and chk_cnt == 0:
            _Beta_0 = self.Initial_StepSize
            _alpha  = self.alpha
            _beta   = self.beta
            # Calculate M
            _Nomin  = _Df - _Beta_0 * _inner_prod
            _Denom  = np.inner(_h, _h)
            _maxEig =  2.0 * _Nomin/(_Beta_0 * _Beta_0 * _Denom )
            # Calculate Estimation of Learning rate
            _EstLR  = (2.0 * (1.0 - _alpha) * _inner_prod)/(_Denom * _maxEig)
            _Estcnt = (lambda _c : int(-self.FastIdleGoControl * math.log(_EstLR)) if _c > 0 else 0)(_EstLR)
            _EstIniLR = _Beta_0 * pow(_beta,  _Estcnt)
            self.VariableBeta = _EstIniLR

            # Debug Code
            if debugOn:
                print("Df : %4.8f   <h, gr> : %4.8f  Df - B<h, gr>: %4.8f  |h|^2 : %4.8f _minEig: %4.8f  _EstLR: %4.8f" \
                        %(_Df, _inner_prod, _Nomin, _Denom, _maxEig, _EstLR))
                print("Estimation k : %d  Init LR : %4.8f" %(_Estcnt, _EstIniLR))

        else:
            _Estcnt = chk_cnt

        return  _Estcnt

    def Armijo_onestep(self, _x, _g, _h, _cost, debugOn=True):
        chk_freq    = self.ArmijoDebugFreq  # Set Debugging Frequency
        chk_cnt     = self.ArmijoOneStepCnt

        # _h 의 경우 -h(x) 로 들어온다
        inner_prod  = np.inner(_g, _h)
        grad        = self.alpha * inner_prod      # \beta^0
        # 부호가 반대이다. next - current 이어야 하는데, 입력은 current - next 이므로
        _Difference = -1.0 * _cost
        _current    = self.ArmijoPrevCost - _cost

        # 1월 20일 02:38 일단 새로운 방법을 적용하기 위해 이렇게 해 놓았다.
        chk_cnt = self.Fast_Armijo_Initialize(chk_cnt, inner_prod, _Difference, _h, debugOn)

        # Learning rate Setting
        beta_k      = (lambda cnt : self.VariableBeta if cnt > 0 else self.Initial_StepSize)(chk_cnt)

        beta_k, chk_cnt, bUpdate = self.Armijo_Core(beta_k, _Difference, grad, chk_cnt, chk_freq, _x, debugOn)

        if chk_cnt >= self.ArmijoInternalCounterLimit and not bUpdate:
            print("==========================================")
            DBG.dbg("Armijo Rule is broken Some Problem Occur!!", _active=True)
            print("==========================================")
            self.bImmediateBreak = True

        chk_cnt = (lambda _y: 0 if _y else chk_cnt)(bUpdate)
        self.VariableBeta       = beta_k
        self.ArmijoOneStepCnt   = chk_cnt

        if bUpdate:
            self.ArmijoPrevCost = _current

        return self.VariableBeta, bUpdate


    def LineSearch(self, x, h, debug=True):
        a = x
        b = x - h
        chk_cnt = 0

        if debug:
            print("-------------------------- Line Search Debug Info ---------------------------------------")

        while True:
            L = b - a
            ai = a + (1 - self.F) * L
            bi = b - (1 - self.F) * L

            Pai = self.inference(ai)
            Pbi = self.inference(bi)

            if Pbi <= Pai:
                a = ai
            else:
                b = bi

            if debug:
                print("[Step: %2d" %chk_cnt, "]", "a:", a, " ai:", ai, "f(ai)=", Pai, " b:", b, " bi:", bi, "f(bi)=", Pbi)

            _dLenth = ai - bi
            _epsilon = np.linalg.norm(_dLenth)

            _bStopCondition = (chk_cnt >= self.LimitofLineSearch) or (_epsilon < self.stop_condition)
            if _bStopCondition:
                break
            chk_cnt = chk_cnt + 1

        if Pbi <= Pai:
            xn = bi
        else:
            xn = ai

        xn, _ = self.Quantization(xn)
        return xn

    def LineSearchStepsize(self, x, g, h, _cost, debugOn=True):
        xn = self.LineSearch(x, h, debugOn)

        test_01 = np.asscalar(np.dot((x - xn), h))
        test_02 = np.asscalar(np.dot(h, h))

        try:
            Lambda = test_01 / test_02
        except Exception as ex:
            print ("LineSearchStepsize Error!!! Error code :", ex)
            Lambda = 0.0
            self.bImmediateBreak = True
        return Lambda, True


    def LineSearch_onestep(self, x, g, h, _cost, debugOn=True):
        xn = self.LineSearch(x, h, debugOn)

        test_01 = np.asscalar(np.dot((x - xn), h))
        test_02 = np.asscalar(np.dot(h, h))

        try:
            Lambda = test_01 / test_02
        except Exception as ex:
            print ("LineSearchStepsize Error!!! Error code :", ex)
            Lambda = 0.0
            self.bImmediateBreak = True
        return Lambda, True

    def TimeDecayingStepsize(self, x, g, h, _cost, debugOn=True):
        c = self.Constant_StepSize
        t = self.AbsoluteTime
        d = self.DecayingRate
        Lambda = c/(t * d + 1)
        self.AbsoluteTime += 1
        return Lambda, True


    def WolfLineSearch(self, x, g, h, _cost, debugOn=True):
        _f      = self.inference
        _gf     = self.gradient 
        _res    = sop.line_search(_f, _gf, x, h)

        return _res(0)

    #================================================================================
    # Quantization
    #================================================================================
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
                if _index > self.index_limit: break
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
        else : pass
        return _index, _infindex

    # Xq : [DX]     Quantized dx^Q = [lm * h]
    # Xf : [dXf]    Floating  dx   = lm * h
    def Adv_quantize(self, Xq, Xf, step):
        _res = Xq
        if self.bQuantization and self.QuantMethod > 0: 
            _index      = self.l_index_trend[-1]
            _infindex   = self.l_infindextrend[-1]
            if step == 0 :
                pass
            else:
                if self.QuantMethod > 0:
                    _res,   _index      = self.Increment_Index(_res, Xf, _index)
                    _index, _infindex   = self.Limited_Index(step, _index, _infindex)
                else: pass                    
            self.l_index_trend.append(_index)
            self.l_infindextrend.append(_infindex)
        else: 
            pass 
        
        return _res

    # To compensate for cases where the Step Size is not properly obtained by Line Search
    # However, it does not work well. It is right to solve it later by Trust Region or other methodologies.
    def Emergency_Solution(self, _h, DX, _Active=False):
        _res = DX
        if self.bQuantization and self.QuantMethod > 0:
            _index      = self.l_index_trend[-1] if len(self.l_index_trend) > 0 else 0
            _norm_h     = np.linalg.norm(_h) 
            _norm_dxq   = np.linalg.norm(DX)
            _condition  = _index >= self.index_limit \
                        and _norm_h     > self.stop_condition * 10.0\
                        and _norm_dxq   < self.stop_condition \
                        and _Active    
            if _condition:
                DBG.dbg("Emegrncy Problem", _active=False)
                _QP         = self.c_qtz.get_QP()
                _atomic_h   = _h * 0.45
                _res        = self.c_qtz.get_Quantization(_atomic_h)
            else:
                pass
        else:
            pass

        return _res 


    #-----------------------------------------------------------
    # Annealing Effect
    # -----------------------------------------------------------
    def Simulated_Annealing(self, c, t):
        T = c/np.log(t + 2)
        return T

    #================================================================================
    # Search Algorithms
    #================================================================================
    # Algorithm ID : 0, 1
    def Armijo_Gradient(self,gr, gr_n, h, _H, Xn, X):
        gr = gr_n
        h = gr_n
        return gr, h, _H

    # Algorithm ID : 2 (Polyak-Reeves), 3(Fletcher-Reeves)
    def Conjugate_Gradient(self, gr, gr_n, h, _H, Xn, X):
        if self._update:
            # Comjugate Gradient
            if self.CGmethod == 0:
                rc = np.inner(gr_n - gr, gr_n) / np.inner(gr, gr)
            else:
                if self.Epoch > 0 :
                    # 정상 동작
                    rc = np.inner(gr_n, gr_n) / np.inner(gr, gr)
                else:
                    # t=0 에서 h = gr_n 이 되도록 한다.
                    rc = 0.0

            gr = gr_n
            hn = gr_n - rc * h
        else:
            #gr = gr
            hn = h

        return gr, hn, _H

    # Algorithm ID : 4
    def Quasi_Newton(self, gr, gr_n, h, B, Xn, DX):
        # In the current algorithm, the case when EPoch = 0 must be handled separately. Also, the case in the Step algorithm must be considered.
        if self.Epoch == 0 or (self.Epoch > 0 and not self._update):
            return gr, h, B

        # Quasi Newton
        basicDim = (np.size(Xn), 1)
        dX = np.reshape(DX, basicDim)  # General 2x1 Vector (tuple) : DX = Xn - X
        dG = np.reshape(gr_n - gr, basicDim)

        # Check Abnormal condition for Quasi Newton
        _bcond1 = (lambda x: x > 0)(np.linalg.norm(dX))
        _bcond2 = (lambda x: x > 0)(np.linalg.norm(dG))
        _bcond3 = (lambda x: x != 0)(np.asscalar(np.dot(dG.T, dX)))

        # DFP Method
        BdG = np.dot(B, dG)           # B : 2x2 to 2x1
        dXdXt = np.dot(dX, dX.T)      # 2x2 Matrix
        BdGBdGt = np.dot(BdG, BdG.T)  # 2x2 Matrix

        if self.AlgorithmWeight == 0:
            Bn_DFP = np.zeros(np.shape(B), dtype=np.float)
        else:                
            if _bcond1 and _bcond2:      # Default and Stanard Case
                test_0 = np.asscalar(np.dot(dX.T, gr))
                test_1 = np.asscalar(np.dot(BdG.T, dG))
                Bn_DFP = B + (dXdXt / test_0) - (BdGBdGt / test_1)
            elif _bcond1:
                test_1 = np.asscalar(np.dot(BdG.T, dG))
                Bn_DFP = B - (BdGBdGt / test_1)
            elif _bcond2:
                test_0 = np.asscalar(np.dot(dX.T, gr))
                Bn_DFP = B + (dXdXt / test_0)
            else:
                Bn_DFP = B

        # BFGS Method
        Ieye = np.eye(np.size(Xn))
        if _bcond3:                  # Default and Stanard Case
            p = 1.0 /np.asscalar(np.dot(dX.T, dG))
            LeftP = Ieye - p * np.dot(dX, dG.T)
            RighP = Ieye - p * np.dot(dG, dX.T)
            Bn_BFGS = np.dot(np.dot(LeftP, B), RighP) + p * dXdXt
        else:
            # When BFGS cindition is broken (dG = 0, dX=0)
            Bn_BFGS = self.BFGS_Broken(B, dG, dX, Ieye)

        _pr = self.AlgorithmWeight
        B = _pr * Bn_DFP + (1 - _pr) * Bn_BFGS
        gr = gr_n
        h = np.matmul(B, gr_n)

        return gr, h, B

    # For Auxiliary Function for Quasi Newton 
    def BFGS_Broken(self, B, dG, dX, Ieye):
        DBG.dbg("Quasi Newton : BFGS meet abnormal Condition : <dG, dX> = 0")
        DBG.dbg("dG.T : %s" %str(dG.T))
        DBG.dbg("dX.T : %s" %str(dX.T))
        self.bImmediateBreak = True
        return B

    # Algorithm ID : 5 (AdaGrad)
    def AdaGrad(self, gr, gr_n, h, B, Xn, DX):
        # In the current algorithm, the case in the Step algorithm must be considered..
        if self.Epoch > 0 and not self._update:
            return gr, h, B

        gr = gr_n
        self.sum_of_gradient += np.dot(gr, gr)
        h = gr/np.sqrt(self.sum_of_gradient)

        return gr, h, B

    # Algorithm ID : 6 (AdaDelta)
    def AdaDelta(self, gr, gr_n, h, B, Xn, DX):
        # In the current algorithm, the case in the Step algorithm must be considered..
        if self.Epoch > 0 and not self._update:
            return gr, h, B

        gr  = gr_n
        g2  = np.dot(gr, gr)
        Eg  = self.expect_gradient      # When Epoch (pr step) is 0, Eg = 0
        _d  = self.delta

        Eg  = _d * Eg + (1.0 - _d) * g2
        v   = self.h_previous

        _RMS_G = np.sqrt(Eg + self.auxiliary_e)
        _RMS_H = np.sqrt(v  + self.auxiliary_e)

        h = (_RMS_H/_RMS_G) * gr
        self.expect_gradient = Eg
        self.h_previous      = np.dot(gr, gr)

        return gr, h, B

    # Algorithm ID : 7 (AdaDelta)
    def RMSProp(self, gr, gr_n, h, B, Xn, DX):
        # In the current algorithm, the case in the Step algorithm must be considered..
        if self.Epoch > 0 and not self._update:
            return gr, h, B

        gr  = gr_n
        g2  = np.dot(gr, gr)
        Eg  = self.expect_gradient      # When Epoch (pr step) is 0, Eg = 0
        _d  = self.delta

        Eg  = _d * Eg + (1.0 - _d) * g2

        _RMS_G = np.sqrt(Eg + self.auxiliary_e)

        h = (1.0/_RMS_G) * gr
        self.expect_gradient = Eg

        return gr, h, B

    # Algorithm ID : 8 (ADAM)
    def ADAM(self, gr, gr_n, h, B, Xn, DX):
        # In the current algorithm, the case in the Step algorithm must be considered..
        if self.Epoch > 0 and not self._update:
            return gr, h, B

        gr  = gr_n
        g2  = np.dot(gr, gr)
        b1  = self.beta_1
        b2  = self.beta_2
        vb1 = b1 * self.variable_beta_1
        vb2 = b2 * self.variable_beta_2
        _e  = self.ADAM_epsilon

        _m  = (lambda _c : self.ADAM_FirstMoment if _c > 0 else gr)(self.Epoch)
        _v  = self.ADAM_SecondMoment

        _m  = b1 * _m + (1 - b1) * gr
        _v  = b2 * _v + (1 - b2) * g2

        m_  = _m/(1 - vb1)
        v_  = _v/(1 - vb2)

        h = (1.0/(np.sqrt(v_) + _e)) * m_

        self.variable_beta_1 = vb1
        self.variable_beta_2 = vb2
        self.ADAM_FirstMoment = _m
        self.ADAM_SecondMoment= _v

        return gr, h, B

    # Algorithm ID : 9 (SGD with Moemntum)
    def SGD_Moment(self, gr, gr_n, h, B, Xn, DX):
        # In the case of Moment, the case of t=0 is the same as SGD.
        #if self.Epoch == 0 or (self.Epoch > 0 and not self._update):
        if self.Epoch > 0 and not self._update:
            return gr, h, B

        gr  = gr_n
        if (self.StepSizeRule == 1 or self.StepSizeRule == 4 or self.StepSizeRule == 5) and self.Epoch > 0:
            lm = self.BestBeta
        else:
            lm = self.Constant_StepSize
        b   = self.moment_gamma
        _v  = self.moment_v

        _v  = b * _v + lm * gr

        # Since Learning Equation is implemented as Xn = X - lm * h,
        # We should multiply the inverse of lm to make Xn = X - h
        h   = 1/lm * _v
        self.moment_v = _v

        '''
        if self.StepSizeRule == 4:
            print("==========================================")
            print("Armijo Rule is not appropriate to the ")
            print("Momentum Algorithm so that Algorithm is broken !!")
            print("==========================================")
            self.bImmediateBreak = True
        '''
        return gr, h, B

    #================================================================================
    # Parameter Update Rule
    #================================================================================
    def DirectlyUpdate(self, Xn, X, Dfunc):    return Xn

    def MonotoneDecrease(self, Xn, X, Dfunc):
        t = self.AbsoluteTime
        if Dfunc <= 0:
            self.AbsoluteTime = t - 1
            Xnew = Xn
        else:
            self.AbsoluteTime = t + 1
            Xnew = X
        return Xnew

    def AnnealingDecrease(self, Xn, X, Dfunc):
        t = self.AbsoluteTime
        if Dfunc <= 0:
            self.AbsoluteTime = t - 1
            Xnew = Xn
        else:
            self.AbsoluteTime = t + 1
            Xnew = X
        return Xnew




