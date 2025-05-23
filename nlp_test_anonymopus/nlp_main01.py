###########################################################
# nlp_main01.py
# Main NLP Class
###########################################################
# import seaborn as sns
import numpy as np
import time
import copy
import math

from nlp_class02 import NLP_Class02
from nlp_functions import nlp_functions as nf
import nlp_service as ns
import my_debug as DBG

class NLP_Main :

    def __init__(self):
        # -----------------------------------------------------------------
        # Set Search Point
        # -----------------------------------------------------------------
        self.Initial_point   = np.array([-0.25548896, 0.0705816])
        self.init_X          = np.array(self.Initial_point, dtype=float)  # Main Training Parameter

        # -----------------------------------------------------------------
        # Algorithm Control
        # -----------------------------------------------------------------
        self.stop_condition = 0.0000003
        self.training_steps = 2000      # Maximum Number of Training Steps
        self.debugfrequency = 10        # print the information per self.debugfrequency for  each step

        # Debug parameter for algorithm 
        self.Nan_gradient   = False
        # -----------------------------------------------------------------
        # Argument Parsing and Parameter Restting
        # -----------------------------------------------------------------
        self.args = ns.ParseArgument(self.debugfrequency, self.training_steps)
        _ret = ns.ParameterSet(self.args, self.Initial_point)
        self.SearchAlgorithm = _ret[0]
        self.CGmethod        = _ret[1]
        self.AlgorithmWeight = _ret[2]
        self.debugfrequency  = _ret[3]
        self.StepSizeRule    = _ret[4] 
        self.Qparam          = _ret[5]
        self.bQuantize       = _ret[6]
        self.init_X          = _ret[7]
        self.training_steps  = _ret[8]
        self.Given_Initial_pt= _ret[9]
        # -----------------------------------------------------------------
        # Set Object Function and Service Functions
        # -----------------------------------------------------------------
        # Definition of Class (if args.random_fix=True, Fixed Initilized)
        _init_point_info    = [self.args.random_fix, self.Given_Initial_pt]
        self.cFunctions     = nf(self.init_X, _init_point_info, _func_id=self.args.functionID)
        self.nlp            = NLP_Class02(self.args)

        # Set Object Function 
        self.object_function     = self.cFunctions.Objective_Function         # Set Object_function 
        self.nlp.BenchmarkClass  = self.cFunctions

        # Set Inital Point for test function 
        self.init_X         = self.cFunctions.Get_Initial_Point()

        # Main Information Buffer
        self._premsgbuf     = ""
        self._printbuf      = ""
        self._Lmsgbuf       = []

        # object nlp_service
        self.c_ns           = ns

    # =================================================================
    # Service Interface Function for Batch Files 
    # =================================================================
    def Get_Function_Object(self):
        return self.cFunctions.Get_Function_Object()

    def Get_Function_Input_Domain(self):
        _fo = self.cFunctions.Get_Function_Object()
        return _fo.input_domain

    def Get_Init_point(self):
        return self.init_X

    def Get_nlp_service(self):
        return self.c_ns

    # =================================================================
    # Set Service Function 
    # =================================================================
    def Check_Gradient_is_Nan(self, _g):
        _chk_nan = np.isnan(_g)
        if True in _chk_nan:
            DBG.dbg("Gradient includes NAN :", _g)
            self.Nan_gradient = True
        else: 
            self.Nan_gradient = False            
    
    def Calculate_Gradient(self, X):
        X   = self.nlp._parameter_tuning(X)
        _g  = self.cFunctions.Diff_Objective_Function(X)
        return _g

    def print_process(self, step, epsilon, prev_cost, current_cost, np_lambda, X, h, lRecord, printfreq=10, ForcedDbgOn=False):
        return ns.print_process(step, epsilon, prev_cost, current_cost, np_lambda, X, h, lRecord, printfreq, ForcedDbgOn)

    def print_pre_process(self, _cost, X, gr, B, GlobalMin, _Test_Function, _msgbuf=""):
        _alpha = _cost/np.dot(gr, gr)
        _msgbuf += ("======================================")  + "\n"
        _msgbuf += ("Initial f(x) : %f at: %s " %(_cost, str(X))) + "\n"
        _msgbuf += ("Gradient     : %s" %str(gr)) + "\n"
        _msgbuf += ("Pseudo Alpha : %f" %_alpha) + "\n"
        _msgbuf += ("Global Minima: %f" %GlobalMin) + "\n"
        _msgbuf += ("self.StepSizeRule : %s" %ns.Algorithm_Info_buf[self.StepSizeRule]) + "\n"
        _msgbuf += ("TestFunction : %s" %_Test_Function) + "\n"
        _msgbuf += ("======================================") + "\n"

        return _msgbuf

    def Init_Quantization(self, _msgbuf=""):
        _msgbuf += ("======================================") + "\n"
        
        if self.bQuantize:
            self.nlp.bQuantization = True
            self.nlp.c_qtz.set_QP(index=self.Qparam)
            self.nlp.l_index_trend.append(self.Qparam)

            _infindex = self.nlp.c_tmp.obj_function(0, self.nlp.c_tmp.T)
            _infindex = 0 if _infindex < 0 else _infindex
            self.nlp.l_infindextrend.append(_infindex)

            _Qp = self.nlp.c_qtz.get_QP()
            _msgbuf += ("Information of Qunatrization [For Developing codes or Debug]") + "\n"
            _msgbuf += ("Q-index (by arg) : %d" %self.Qparam ) + "\n"
            _msgbuf += ("self.Qparameter (Qp)  : %d" %_Qp) + "\n"
            _msgbuf += ("self.Qparameter^{-1}  : %f" %(1.0/_Qp)) + "\n"

        else:  
            _msgbuf += ("No Quantization") + "\n"   

        _msgbuf += ("======================================") + "\n"
        return _msgbuf

    def Chk_StopCondition(self, step, epsilon, _update, current_cost, GlobalMin, gr):
        _condition = []
        _condition.append((step > 0))
        _condition.append((np.linalg.norm(gr) < self.stop_condition))
        _condition.append((abs(epsilon) < self.stop_condition and _update))
        _condition.append((current_cost == GlobalMin))
        _condition.append(self.nlp._get_break())
        _condition.append(self.cFunctions._get_break())

        # Quantization Active 일때 step==1 인 경우에 Quantization으로 움직이지 않으면 (_condition[2]=True) 이를 수행한다.
        if self.nlp.bQuantization and step < 2 and _condition[2]:
            _condition[2] = False

        _res = _condition[0] and (_condition[1] or _condition[2] or _condition[3] or _condition[4] or _condition[5])
        
        Best_Step = step - (lambda _c: 1 if _c else 0)(_res)
        return _res, Best_Step 


    def Auxiliary_BeginInLoop (self, step, _dbgBeginStep, _dbgFinishStep, _DebugOn):
        # Debug Condition @ for Break Point
        Debug_param =  ((step >= _dbgBeginStep) and (step < _dbgFinishStep)) and _DebugOn
        # Epoch Update
        self.nlp.Epoch = step

        return Debug_param

    # =================================================================
    # Main Routine 
    # =================================================================
    def main_processing(self):    

        args = self.args
        # -----------------------------------------------------------------
        # Ready for Process 
        # -----------------------------------------------------------------
        # For record a simulation
        rEpsilon    = []
        rCost       = []
        rX          = []
        rLambda     = []
        iBestIdx    = []
        lRecord     = [rEpsilon, rCost, rX, rLambda, iBestIdx, self.cFunctions.Get_Function_ID]

        # Definition of NLP_Class
        self.nlp.Constant_StepSize   = args.LearningRate
        
        
        evalstepsize            = self.nlp.stepfunc[self.StepSizeRule]  # Set Step Size Rule : evalstepsize is function
        
        
        SearchFunction          = self.nlp.searchfunc[self.SearchAlgorithm]  # Set Search Function
        self.nlp._parameter_initialization(args, self.object_function, self.Calculate_Gradient, self.AlgorithmWeight, self.CGmethod, self.SearchAlgorithm, self.StepSizeRule)

        # Set Pre information
        self._printbuf    = ns._shline + "\n"
        self._printbuf   += '[' + args.message +']' + " Test Start : " + ("Benchmark Function : %s " %self.cFunctions.Get_Function_Name()) + "\n"
        self._printbuf   += ns._shline + "\n"
        ns.ns_print(self._printbuf, Active=not args.quite_mode)

        # Quantization Parameter Setting if Quantization is active
        self._printbuf    = self.Init_Quantization(_msgbuf=self._printbuf)

        # Description of Object_Function() and Gradient Parameter
        inference    = self.object_function
        GlobalMin    = inference(self.cFunctions.Optimal_Point())
        initial_cost = inference(self.init_X)
        init_gr      = self.Calculate_Gradient(self.init_X)
        init_gr_n    = np.zeros(np.shape(self.init_X), dtype=float)
        init_h       = init_gr
        init_B       = np.eye(np.shape(self.init_X)[0], dtype=float)

        # Set Initial value to Record data set
        rEpsilon.append(0.0)
        rCost.append(initial_cost)
        rLambda.append((lambda y: self.nlp.Initial_StepSize if y > 0 else self.nlp.Constant_StepSize)(self.StepSizeRule))
        rX.append(self.init_X)

        # print Basic Processing Information
        _Benchmark_function = self.cFunctions.Get_Function_Name()
        self._printbuf = self.print_pre_process(initial_cost, self.init_X, init_gr, init_B, GlobalMin, _Benchmark_function, _msgbuf=self._printbuf)
        self._premsgbuf = copy.deepcopy(self._printbuf)
        self._Lmsgbuf.append(self._premsgbuf)
        self._printbuf = ""; 

        # Set Print Parameters for Step 
        ns._nsPrintActive = not args.quite_mode     #_nsPrintActive is inverse of args.quite_mode

        # -----------------------------------------------------------------
        #if step == 2320:
        #    print(ns._filename_(), ns._lineno_(), " Debug :: Step at ", step)
        #
        # Auxiliary Function in Main Routine
        # param _dbgBeginStep : Debug Begin step
        # param _dbgFinishStep: Debug finish step
        # -----------------------------------------------------------------
        # Debug Parameter
        DebugOn = False
        #DebugOn = (args.stepsize == 1)

        # -----------------------------------------------------------------
        # Search Routine
        # -----------------------------------------------------------------
        # Constant Parameter
        Algorithm_Info = ns.Algorithm_Info_buf[self.StepSizeRule]

        # Initial Point Processing : If it is needed
        X   = self.init_X
        X,_ = self.nlp.Quantization(X)   # Quantization (if it is applied)
        Xn  = X
        DX  = Xn - X

        prev_cost       = 0
        current_cost    = initial_cost
        epsilon         = prev_cost - current_cost
        gr, gr_n, h, B  = init_gr, init_gr_n, init_h, init_B
        self.nlp.ArmijoPrevCost = prev_cost

        _update         = True
        start_time      = time.time()
        for step in range(self.training_steps):
            # Ready for Each Iteration
            Debug_param = self.Auxiliary_BeginInLoop(step, 0, 100000, DebugOn)

            # Compute Descent Direction : Evaluate the Gradient and Estimated Hessian
            gr_n        = self.Calculate_Gradient(X)
            gr, h, B    = SearchFunction(gr, gr_n, h, B, X, DX)

            # Check Stop Condition 
            Stop_Process_condition, Best_Step = self.Chk_StopCondition(step, epsilon, _update, current_cost, GlobalMin, gr)
            if Stop_Process_condition:  break

            # Cost, Difference Evaluation for One step (Idle and Go type)
            epsilon, prev_cost, current_cost = self.nlp._cost_update_onestep(_update, prev_cost, current_cost, X, Xn, h, inference, Algorithm_Info)

            # Learning or Searching : Main Search Rule
            lm, _update = evalstepsize(X, gr, h, epsilon, Debug_param)
            DX      = lm * h
            DX, dxf = self.nlp.Quantization(DX)                     # If bQuantization is False, it is just a dummy code
            DX      = self.nlp.Adv_quantize(DX, dxf, step)          # If needed, advanced quantization is achieved 
            DX      = self.nlp.Emergency_Solution(h, DX, _Active=False) # 2021-08-30 For very small step size 
            Xn      = X - DX

            # Cost, Difference Evaluation (General case)
            epsilon, prev_cost, current_cost = self.nlp._cost_update_general(_update, prev_cost, current_cost, X, Xn, h, inference, Algorithm_Info)

            # Difference and Update : n <- n+1
            DX  = Xn - X
            X   = (lambda _c: Xn if _c else X)(_update)
            self.nlp._update = _update

            # Record Each Iteration
            Xr = (lambda _y: X if (_y == ns.Algorithm_Info_buf[4] or _y == ns.Algorithm_Info_buf[5]) else Xn)(ns.Algorithm_Info_buf[self.StepSizeRule])
            lRecord = self.print_process(step, epsilon, prev_cost, current_cost, lm, Xr, h, lRecord, self.debugfrequency, _update)


        # -----------------------------------------------------------------
        # Final Routine
        # -----------------------------------------------------------------
        Processing_Time = time.time() - start_time
        Optimal_Point = self.cFunctions.Optimal_Point()
        ns.set_printbuf(self._printbuf)

        self.Qparam = self.nlp.l_index_trend[-1] if self.bQuantize else self.Qparam
        _AlgorithmInfo = [self.SearchAlgorithm, self.StepSizeRule, self.AlgorithmWeight, self.Qparam]
        _DataInfo = [X, h, Processing_Time, Best_Step, epsilon, current_cost, lm]

        self._printbuf += ("====================================") + "\n"
        self._printbuf += ("Stop Condition Parameters [self.stop_condition]: %f" %self.stop_condition) + "\n"
        self._printbuf += (" step               : %d " %step) + "\n"
        self._printbuf += (" np.linalg.norm(gr) : %f " %np.linalg.norm(gr)) + "\n"
        self._printbuf += (" abs(epsilon)       : %f " %abs(epsilon)) + "\n"
        self._printbuf += (" nlp._get_break()   : %s " %str(self.nlp._get_break() )) + "\n"
        self._printbuf += (" cFunctions._get_break() : %s" %str(self.cFunctions._get_break() ))

        self.nlp._inference = inference
        self._Lmsgbuf  = ns.print_Result(self.nlp, _AlgorithmInfo, _DataInfo, self._Lmsgbuf)
        
        ns.plot_result(Best_Step, lRecord, Optimal_Point, args, self.nlp)
        ns.ns_refresh()

        return self._Lmsgbuf, args

# =================================================================
# End of Class
# =================================================================
# Service Function
# =================================================================
# Final Stage : record information 
# 1. print first 
def Print_Buf(_buf, args):
    for _k, _msg in enumerate(_buf):        
        _lmsg = _msg.split()
        if _lmsg[0] == "step:" and args.quite_mode:
            continue
        else:
            pass
        print(_msg)

# 2. Make msgs for File and Write out File 
def File_Print(_buf, args, _File_Name="TXT_Result.txt", _option='a'):
    for _k, _msg in enumerate(_buf):
        _lmsg = _msg.split()
        if _lmsg[0] == "step:":
            _fmsg = _msg + "\n"
            if args.quite_mode:
                continue
            else:
                pass
        else:            
            _fmsg = _msg
        _buf[_k] = _fmsg

    with open(_File_Name , _option) as f:
        for _fmsg in _buf:
            f.write(_fmsg)

# =================================================================
# Main Routine 
# =================================================================
# For Module processing
if __name__ == '__main__':
    NLP = NLP_Main()
    _infobuf, args = NLP.main_processing()

    Print_Buf(_infobuf, args)
    File_Print(_infobuf, args)
