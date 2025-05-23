###########################################################
# nlpclass.py
# Class Name : NLP_Class
###########################################################
import numpy as np

class NLP_Class :

    def __init__(self):
        # Basic Function pointer
        self.stepfunc   = []
        self.searchfunc = []
        self.update_rule= []

        # Constant Step Size
        self.Constant_StepSize = 0.00212      ## Default 0.002

        # Time Decaying Step Size
        self.AbsoluteTime  = 0
        self.DecayingRate  = 1.0

        # For evaluation of Armijo'rule
        self.alpha = 0.5
        self.beta  = 0.612
        self.Initial_StepSize   = 0.9
        self.ArmijoDebugFreq    = 10.0
        self.ArmijoOneStepCnt   = 0
        self.VariableBeta       = 1.0

        # Method of Conjugate Gradient
        self.CGmethod = 0                  # 0 : Polak - Riebel  1 : Fletcher-Reeves

        # Line Search (Golden Search)
        self.F = 0.618
        self.LimitofLineSearch = 20
        self.stop_condition = 0.0000003

        # Quasi Newton
        self.AlgorithmWeight = 0.0  # Weight of DFP if zero, it means that BFGS

        # Qunatrization
        self.bQuantization   = False
        self.iQuantparameter = 10000
        self.k_limit         = 10
        self.qc_param = np.array([[0, 1],[1, 0], [1, 1]], dtype=np.float)

        # Break Occur
        self.bImmediateBreak = False

        # Function Pointer
        self.stepfunc.append(self.ConstantStepSize)
        self.stepfunc.append(self.Armijo_Lambda)
        self.stepfunc.append(self.LineSearchStepsize)
        self.stepfunc.append(self.TimeDecayingStepsize)
        self.stepfunc.append(self.Armijo_onestep)

        self.searchfunc.append(self.Armijo_Gradient)
        self.searchfunc.append(self.Conjugate_Gradient)
        self.searchfunc.append(self.Quasi_Newton)

    def _parameter_initialization(self, cost_function, Quasi_DFP_Weight, ConjugateMethod):
        """_parameter_initialization for Quasi Newton and Conjugate Method.
            :param Quasi_DFP_Weight: Weight of DFP method. 0 <= param <= 1.0
            :type  Quasi_DFP_Weight: float
            :param ConjugateMethod : Method of Conjugate Gradient
            :type  ConjugateMethod : int
        """
        self.AlgorithmWeight = Quasi_DFP_Weight
        if ConjugateMethod != None:
            self.CGmethod        = ConjugateMethod

        self.inference = cost_function
        return 0

    def inference(self, X):
        return X

    #-----------------------------------------------------------
    # Step Size
    # -----------------------------------------------------------
    def ConstantStepSize(self, x, g, h, _cost, debugOn=True):
        return self.Constant_StepSize, True

    def Armijo_Lambda(self, np_x, np_g, np_h, _cost, debugOn=True):
        chk_freq = self.ArmijoDebugFreq  # Set Debugging Frequency
        beta_k = self.Initial_StepSize
        grad = self.alpha * np.inner(np_g, np_g)  # \beta^0
        chk_cnt = 0
        while True:
            xn = np_x - beta_k * np_h  # x_i - \beta^k h_i
            Phi = self.inference(xn) - _cost  # f(x_i + \beta^k h_i) - f(x_i)
            Psi = - beta_k * grad  # -beta^k \cdot \alpha \| \nabla f(x_i) \|^2

            if Phi > Psi:
                beta_k = beta_k * self.beta
                chk_cnt = chk_cnt + 1
                if chk_cnt % chk_freq == 0 :
                    grad = grad * self.alpha  # := alpha^k
                    nr_grad = np.linalg.norm(grad)
                    if debugOn:
                        print("Armijo Info count:", "%4d" % chk_cnt, "beta:", "%4.8f" % beta_k, "alpha * |grad| :",
                                "%4.8f" % nr_grad)
            else:
                Lambda = beta_k
                break

        return Lambda, True

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

        xn = self.Qunatization(xn)
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

    def TimeDecayingStepsize(self, x, g, h, _cost, debugOn=True):
        c = self.Constant_StepSize
        t = self.AbsoluteTime
        d = self.DecayingRate
        Lambda = c/(t * d + 1)
        self.AbsoluteTime += 1
        return Lambda, True

    def Armijo_onestep(self, _x, _g, _h, _cost, debugOn=True):
        chk_freq    = self.ArmijoDebugFreq  # Set Debugging Frequency
        chk_cnt     = self.ArmijoOneStepCnt

        beta_k      = (lambda cnt : self.VariableBeta if cnt > 0 else self.Initial_StepSize)(chk_cnt)
        grad        = self.alpha * np.inner(_g, _g)     # \beta^0
        Psi         = - beta_k * grad                   # -beta^k \cdot \alpha \| \nabla f(x_i) \|^2
        xe          = _x - beta_k * _h
        _next       = self.inference(xe)
        _Difference = _next - _cost

        if _Difference > Psi:
            beta_k = beta_k * self.beta
            chk_cnt = chk_cnt + 1
            Update  = False
            if chk_cnt % chk_freq == 0 and debugOn:
                print("Armijo Info count:", "%4d" % chk_cnt, "beta:", "%4.8f" % beta_k, "beta^k * alpha * |grad| :", "%4.8f" % (-1.0 * Psi))
        else:
            chk_cnt = 0
            Update = True

        self.VariableBeta = beta_k
        self.ArmijoOneStepCnt = chk_cnt

        return self.VariableBeta, Update

    #-----------------------------------------------------------
    # Quantization
    # -----------------------------------------------------------
    def Qunatization(self, _x):

        if self.bQuantization:
            _xt   = _x * self.iQuantparameter
            _xt   = np.round(_xt)
            _ix   = _xt.astype(int)
            _fx   = _ix/self.iQuantparameter
        else:
            _fx   = _x

        return _fx

    def Q_Compensation(self, x, h, _cost, function, bdebug_info):
        xn = x
        epsilon = _cost - self.inference(x)
        if self.bQuantization and epsilon <= 0:
            fbest = _cost
            sh    = np.sign(h)/self.iQuantparameter
            hmin  = np.min(np.abs(h))
            normh = np.linalg.norm(h)

            if bdebug_info:
                print("-------------------------- Qunatization Debug Info ------------------------------------")
                print("<step Ready> x:", x, "h: ", h, "|h|:", normh, "prev_cost: %4.8f" %fbest, "current_cost: %4.8f" %self.inference(x))

            k = -1
            while True :
                g_k   = pow(2, k)
                gQ_c  = np.abs(g_k * h / hmin) + 0.5/self.iQuantparameter
                hQ_ck = self.Qunatization(gQ_c * sh)

                for i in range(x.size + 1):
                    hq_i = self.qc_param[i] * hQ_ck
                    xcc    = x - hq_i
                    fnew  = function(xcc)
                    if fnew < fbest or i==2:
                        xc = xcc
                        break

                if bdebug_info:
                    print("<step %2d" %k, "> x :", x, " xn:", xc, "|hq|:", np.linalg.norm(hQ_ck), "best cost : %4.8f" %fbest, " New cost: %4.8f" %fnew)

                # Stop Condition
                if fnew < fbest or np.linalg.norm(hQ_ck) > normh or k > self.k_limit:
                    if fnew < fbest:
                        xn = xc
                    break
                else:
                    k = k + 1

            print("[Compensation On] Xold :", x, "hQ :", hQ_ck, "Xn : ", xn, "prev_cost: %4.8f" %_cost, "Compensated Cost: %4.8f" %self.inference(xn))
        return xn

    #-----------------------------------------------------------
    # Annealing Effect
    # -----------------------------------------------------------
    def Simulated_Annealing(self, c, t):
        T = c/np.log(t + 2)
        return T

    #-----------------------------------------------------------
    # Search Algorithms
    # -----------------------------------------------------------
    def Armijo_Gradient(self,gr, gr_n, h, _H, Xn, X, _bUpdate):
        gr = gr_n
        h = gr_n
        return gr, h, _H

    def Conjugate_Gradient(self, gr, gr_n, h, _H, Xn, X, _bUpdate):
        if _bUpdate == False: return gr, h, _H
        # Comjugate Gradient
        if self.CGmethod == 0:
            rc = np.inner(gr_n - gr, gr_n) / np.inner(gr, gr)
        else:
            rc = np.inner(gr_n, gr_n) / np.inner(gr, gr)

        gr = gr_n
        h  = gr_n - rc * h

        return gr, h, _H

    def Quasi_Newton(self, gr, gr_n, h, B, Xn, X, __bUpdate):
        # Quasi Newton
        basicDim = (np.size(X), 1)
        dX = np.reshape(Xn - X, basicDim)  # General 2x1 Vector (tuple)
        dG = np.reshape(gr_n - gr, basicDim)

        # Check Abnormal condition for Quasi Newton
        _bcond1 = (lambda x: x > 0)(np.linalg.norm(dX))
        _bcond2 = (lambda x: x > 0)(np.linalg.norm(dG))
        _bcond3 = (lambda x: x != 0)(np.asscalar(np.dot(dG.T, dX)))

        # DFP Method
        BdG = np.dot(B, dG)           # B : 2x2 to 2x1
        dXdXt = np.dot(dX, dX.T)      # 2x2 Matrix
        BdGBdGt = np.dot(BdG, BdG.T)  # 2x2 Matrix

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
        Ieye = np.eye(np.size(X))
        if _bcond3:                  # Default and Stanard Case
            p = 1.0 /np.asscalar(np.dot(dX.T, dG))
            LeftP = Ieye - p * np.dot(dX, dG.T)
            RighP = Ieye - p * np.dot(dG, dX.T)
            Bn_BFGS = np.dot(np.dot(LeftP, B), RighP) + p * dXdXt
        else:
            Bn_BFGS = B
            print("Quasi Newton : BFGS meet abnormal Condition : <dG, dX> = 0")

        _pr = self.AlgorithmWeight
        B = _pr * Bn_DFP + (1 - _pr) * Bn_BFGS
        gr = gr_n
        h = np.matmul(B, gr_n)

        return gr, h, B

    #-----------------------------------------------------------
    # Parameter Update Rule
    # -----------------------------------------------------------

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




