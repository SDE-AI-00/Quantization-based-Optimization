###########################################################
# nlp_service.py
# Class Name : nlp_service
###########################################################
import numpy as np
import matplotlib.pyplot as plt
import argparse
import textwrap

from matplotlib import animation
from mpl_toolkits import mplot3d
import platform
import os
import sys 

# Structurally, nlp_service is read by nlp_class02.
# However, in Service, the variable of nlp_class02 must be referenced, so nlpclass must be read.
from nlpclass import NLP_Class

from nlp_functions import nlp_functions as nf
import my_debug as DBG
#==========================================================
# Line print function for Python file debugging
# Usage Example :
# if (step >= 68):
#         print(ns._filename_(), ns._lineno_(), " Debug :: Step at ", step)
#==========================================================
import inspect
# Returns the line number where this function was called.
def _lineno_():
    return inspect.getlineno(inspect.getouterframes(inspect.currentframe())[-1][0])
# Returns the file name where this function was called along with the Path.
def _filename_():
    return inspect.getfile(inspect.getouterframes(inspect.currentframe())[-1][0])

#==========================================================
# Save and Load Results Data
#==========================================================
import pickle

class nlp_Records:
    def __init__(self, FileName='nlp_results'):
        self.rEpsilon   = []
        self.rCost      = []
        self.rX         = []
        self.rLambda    = []
        self.nDimension = 0
        self.rBestIdx   = []
        self._BestIdx   = 0
        self._FunctionId= 0
        self.filename_  = FileName


    def save_data(self, rEpsilon, rCost, rX, rLambda, iBestIdx, _FunctionID):
        self.rEpsilon   = rEpsilon
        self.rCost      = rCost
        self.rX         = rX
        self.rLambda    = rLambda
        self.rBestIdx   = iBestIdx
        self.nDimension = len(rX)
        self._FunctionId= _FunctionID

    def load_data(self):
        return self.rEpsilon, self.rCost, self.rX, self.rLambda, self.iBestIdx

    def save_data_as_file(self):
        try :
            with open(self.filename_ , 'wb') as f:
                pickle.dump(self, f, protocol=4)
            print("Results Data are written as nlp_results")
        except PermissionError:
            DBG.dbg("PermissionError: [Errno 13] Permission denied: './Results\\nlp_results'")
            DBG.dbg("It is impossible to write the results to a File")
            
    # Load Example
'''
    def load_data_as_file(self):
        with open('nlp_results'), 'rb') as f:
            self = pickle.load(f)
        print("Load Results Data from the file ./Results/nlp_results")
'''
#==========================================================
# Service Function Main
#==========================================================
# Algorithm/Class Environment
strAlgorithmName        = [ 'Gradient_Descent',
                            'Conjugate_Gradient_(PR)',
                            'Conjugate_Gradient_(FR)',
                            'Quasi_Newton',
                            'AdaGrad', 'AdaDelta', 'RMSProp', 'ADAM', 'SGD_Moment', 'Nestrov_Acceleration',
                            'Default Learning']
Algorithm_Info_buf      = [ "Constant Step Size ", 
                            "Armijo ", 
                            "Line Search (Golden Search)", 
                            "Time Decaying", 
                            "Armijo One Step Size(Idle and Go)", 
                            "Armijo One Step Size(Fast)", 
                            "Line Search (Idle and Go)", 
                            "Default MLP"]
ConjugateGradientInfo   = ["Polak-Riebel", "Fletcher-Reeves"]

#_inference              = 0 # Function pointer of Inference Function
# Processing Environment
LocalNLP    = NLP_Class()
LocalNLPRC  = nlp_Records()

_OSList     = ['Windows',   'Linux']
_OSName     = _OSList.index(platform.system())
_printbuf   = ""
_Lstepmsg   = []
_nsPrintActive = False

_shline ="-----------------------------------------------------------------"
_lhline ="================================================================="

def ns_refresh():
    _printbuf = ""
    _nsPrintActive = False
    _Lstepmsg.clear()

def ns_print(_msg, Active=False):
    if      Active or _nsPrintActive: print(_msg) 
    else:   print(".", end='')

def set_printbuf(_msg_buf):
    _printbuf = _msg_buf
    return _printbuf

def print_process(step, epsilon, prev_cost, current_cost, np_lambda, X, h, lRecord, printfreq=10, ForcedDbgOn=False):
    # Programming Result
    if (printfreq > 0 and step % printfreq == 0) or ForcedDbgOn:
        _stepmsg = ("step: %3d epsilon: %4.8f prev_cost: %4.8f current_cost: %4.8f np_lambda : %4.8f " \
                   %(step, epsilon, prev_cost, current_cost, np_lambda)) + \
                   ("h : %s X : %s" %(str(h), str(X)))
        _Lstepmsg.append(_stepmsg)
        ns_print(_stepmsg)
    else: pass

    LocalNLPRC._BestIdx = step if ForcedDbgOn else LocalNLPRC._BestIdx

    rEpsilon    = lRecord[0]
    rCost       = lRecord[1]
    rX          = lRecord[2]
    rLambda     = lRecord[3]
    iBestIdx    = lRecord[4]

    rEpsilon.append(epsilon)
    rCost.append(current_cost)
    rLambda.append(np_lambda)
    rX.append(X)
    iBestIdx.append(LocalNLPRC._BestIdx)

    return [rEpsilon, rCost, rX, rLambda, iBestIdx]

def Parameter_pop(_param):
    rEpsilon    = _param[0]
    rCost       = _param[1]
    rX          = _param[2]
    rLambda     = _param[3]

    rEpsilon.pop()
    rCost.pop()
    rLambda.pop()
    rX.pop()

def print_Result(nlp, AlgorithmInfo, DataInfo, _Lmsgbuf, _printbuf=_printbuf):

    _SearchAlgorithm    = AlgorithmInfo[0]
    _StepSizeRule       = AlgorithmInfo[1]
    _AlgorithmWeight    = AlgorithmInfo[2]
    _Qparam             = AlgorithmInfo[3]

    X                   = DataInfo[0]
    h                   = DataInfo[1]
    Processing_Time     = DataInfo[2]
    Best_Step           = DataInfo[3]
    epsilon             = DataInfo[4]
    current_cost        = DataInfo[5]
    lm                  = DataInfo[6]

    print("")
    _printbuf += "=========== Final Result ===========" + "\n"
    _printbuf += "OS        : %s" %_OSList[_OSName] + "\n"
    _printbuf += "Algorithm : %s" %strAlgorithmName[_SearchAlgorithm] + "\n"


    StepSizeSecondStr   = ["   Learning Rate=", "    alpha=", "   No Supplementary Info", "   No Supplementary Info", "    alpha=", "   No Supplementary Info"]
    StepSizeSecondInfo  = [nlp.Constant_StepSize,  nlp.alpha, False, False, nlp.alpha, False]
    StepSizeRuleInfo    = Algorithm_Info_buf[_StepSizeRule] + StepSizeSecondStr[_StepSizeRule] \
                          + (lambda _c: str(StepSizeSecondInfo[_StepSizeRule]) if _c != False else "")(StepSizeSecondInfo[_StepSizeRule])
    #print("   Step Size Rule           :", Algorithm_Info_buf[_StepSizeRule], "    alpha =", nlp.alpha)
    _printbuf += "   Step Size Rule           : %s" %StepSizeRuleInfo + "\n"

    if _SearchAlgorithm == 0:
        _printbuf += "   Constant Learning rate   : %f" %lm + "\n"
    elif _SearchAlgorithm == 1 or _SearchAlgorithm == 2:
        _printbuf += "   Conjugate Gradient Method: %s" %ConjugateGradientInfo[nlp.CGmethod] + "\n"
    elif _SearchAlgorithm == 3:
        _printbuf += "   Algorithm Weight [DFP]   : %f" %_AlgorithmWeight + "[BFGS] : %f" %(1 - _AlgorithmWeight) + "\n"
    else: 
        pass

    # ------ Test ------
    _printbuf += "   Quantization Level       : %s" %_Qparam 
    if nlp.bQuantization :
        _printbuf += "     Quantization  ON" + "\n"
    else:
        _printbuf += "     Quantization  OFF" + "\n"
    #---------------------
    _printbuf += "====================================" + "\n"
    _printbuf += 'Best step: %3d epsilon: %4.8f  current_cost: %4.8f lambda: %4.8f ' % (Best_Step, epsilon, current_cost, lm)
    _printbuf += "X: %s Final h: %s" %(str(X), str(h)) + "\n"
    _printbuf += "Processing Time : %4.8f" % Processing_Time + "\n"
    _printbuf += "====================================" + "\n"
    #---------------------
    _optimalptval = nlp.BenchmarkClass.Get_Optimal_Value_and_Point()
    _absdifference= np.abs(_optimalptval[1] - current_cost)
    _NormDifferent= np.linalg.norm(X - _optimalptval[0])
    
    _printbuf += "Optimal Cost     : %4.8f  OptPoint : %s" %(_optimalptval[1], _optimalptval[0]) + "\n"
    _printbuf += "current_cost     : %4.8f  CrtPoint : %s" %(current_cost, str(X)) + "\n"
    _printbuf += "------------------------------------" + "\n"
    _printbuf += "|Final step      | Difference(Cost) | Norm of Point |" + "\n"
    _printbuf += "| %3d  | %4.8f  | %4.8f  | " %(Best_Step, _absdifference, _NormDifferent) + "\n"
    _printbuf += "====================================" + "\n"
    # ----- Final Process ----
    for _stepmsg in _Lstepmsg:
        _Lmsgbuf.append(_stepmsg)
    _Lmsgbuf.append(_printbuf)

    return _Lmsgbuf

#==========================================================
# Plotting Objective function and Result
#==========================================================
# Old code : def Description_of_Object_Function(nlp, np_X, lLimit, np_Cost):
def Description_of_Object_Function(nlp, lLimit):
    if len(lLimit) == 0 : lLimit=[-2.0, 3.0, -2.0, 3.0]
    else: pass

    x = np.arange(lLimit[0], lLimit[1], 0.05)
    y = np.arange(lLimit[2], lLimit[3], 0.05)

    X, Y = np.meshgrid(x, y)
    
    #Z = nlp._inference(np.array([X, Y]))
    Z = np.apply_along_axis(nlp._inference, 0, np.array([X, Y]))

    return X, Y, Z

def animate(i, gnp_rX, line):
#    x = gnp_rX[i, 0]
#    y = gnp_rX[i, 1]
#    line.set_data(x, y)
    line.set_data(gnp_rX[:i,0], gnp_rX[:i,1])
    return line,

def GetAlgorithmName(args):
    AlgorithmName = ["GradientConstant", "GradientArmijo", "ConjugatePolak", "ConjugateReevel", "QuasiNewtonBFGS"]
    rtvalue = AlgorithmName[args.algorithm]
    return rtvalue

def Get_LevelList(nlp):
    level_list    = {'Rosenblatt':0, 'Ackley':1, 'Whitley':1, 'Rosenbrock Modification':0, 'Egg Holder':2, 'Xin-She Yang N.4':2, 'Salomon':2, \
                    'Drop-Wave':2, 'Powell':2, 'Schaffel N. 2':2 }
    _key = nlp.BenchmarkClass.Get_Function_Name()
    return level_list[_key]

def plot_result(Best_Step, _ResultData, Optimal_Point, args, nlp):
    # Check Activation
    if args.quite_mode or args.NoShowFigure:
        return
    else:
        pass         
        
    # Data Management
    rEpsilon    = _ResultData[0]
    rCost       = _ResultData[1]
    rX          = _ResultData[2]
    rLambda     = _ResultData[3]
    iBestIdx    = _ResultData[4]

    time_index  = range(Best_Step+1 + 1)
    np_rX       = np.array(rX)
    np_Cost     = np.array(rCost)
    lLimit      = [-2.0, 3.0, -2.0, 3.0]     # Default Value
    X, Y, Z     = Description_of_Object_Function(nlp, lLimit)

    #--------------------------------------------------------------------
    # plot dub-figure (trend of epsilon, cost, Lmabda and trace of X)
    # [2e-8, 2e-4, 2e-2, 2e-1, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.65, 2.75, 3, 4, 5, 6]
    # --------------------------------------------------------------------
    _Level = Get_LevelList(nlp)
    level_formats = [[2e-8, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384], \
                     [2e-8, 2e-4, 2e-2, 2e-1, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.65, 2.75, 3, 4, 5, 6],
                     []]  
    
    levels = level_formats[_Level]
    plt.subplot(221)
    plt.plot(time_index, rEpsilon, 'b')
    plt.ylabel('Epsilon')
    plt.xlabel('Epochs')

    plt.subplot(222)
    plt.plot(time_index, rCost, 'r')
    plt.ylabel('Cost')
    plt.xlabel('Epochs')

    plt.subplot(223)
    plt.plot(time_index, rLambda, 'b')
    plt.ylabel('Lambda')
    plt.xlabel('Epochs')

    plt.subplot(224)
    
    if len(levels) > 0:
        plt.contour(X, Y, Z, levels)
    else:        
        plt.contourf(X, Y, Z, levels=6)
    
    plt.plot(np_rX[:, 0], np_rX[:, 1], 'b')
    plt.plot(Optimal_Point[0], Optimal_Point[1], 'r*', label="Optimal Point")
    plt.ylabel('Y-Coordinate')
    plt.xlabel('X-Coordinate')
    plt.legend(loc=2)

    f1 = plt.gcf()

    if _OSList[_OSName] == 'Windows':
        if not args.NoShowFigure:
            plt.show()
        else:
            print("No Show Figure Option is active for result of Algorithm Performance")
        f1.savefig("Figure_Quad.pdf", format="pdf", dpi=600)
        #f1.savefig("D:\Document\Github\python01\Figure_Quad.eps", format="eps")
        #f1.savefig("C:\\Users\Jinwuk_surface\Downloads\Jinwuk_Work\Github\python01\Figure_Quad.eps", format="eps")
    elif _OSList[_OSName] == 'Linux':
        if not (os.access("Figure_Quad.eps", os.W_OK)):      # Check the file is Writable
            os.chmod ("Figure_Quad.eps", 0o777)              # if the file is not writable, chmod 777

        f1.savefig("Figure_Quad.eps", format="eps")
    else :
        raise ArithmeticError('OS Name is not assigned!! please check the nlp_service.py')

    #--------------------------------------------------------------------
    # Second plot
    # --------------------------------------------------------------------
    xmin = np.amin(np_rX[:,0]) - 0.05
    xmax = np.amax(np_rX[:,0]) + 0.05
    ymin = np.amin(np_rX[:,1]) - 0.05
    ymax = np.amax(np_rX[:,1]) + 0.05

    xmin = (xmin, Optimal_Point[0] - 0.05)[xmin > Optimal_Point[0]]  # A=(False_Result, True_Result)[Condition]
    xmax = (xmax, Optimal_Point[0] + 0.05)[xmax < Optimal_Point[0]]
    ymin = (ymin, Optimal_Point[1] - 0.05)[ymin > Optimal_Point[1]]
    ymax = (ymax, Optimal_Point[1] + 0.05)[ymax < Optimal_Point[1]]

    lLimit = [xmin, xmax, ymin, ymax]

    X, Y, Z = Description_of_Object_Function(nlp, lLimit)

    plt.plot()
    if len(levels) > 0:
        plt.contour(X, Y, Z, levels)
    else:        
        plt.contourf(X, Y, Z, levels=30)

    plt.plot(Optimal_Point[0], Optimal_Point[1], 'r*', label="Optimal Point")

    plt.ylim((ymin, ymax))
    plt.xlim((xmin, xmax))
    plt.ylabel('Y-Coordinate')
    plt.xlabel('X-Coordinate')
    plt.legend(loc=2)

    f2 = plt.gcf()
    plt.plot(np_rX[:, 0], np_rX[:, 1], 'b', label="Trace")

    if _OSList[_OSName] == 'Windows':
        if not args.NoShowFigure:
            plt.show()
        else:
            print("No Show Figure Option is active for result of Optimization")

        f2.savefig("Figure_trace.pdf", format="pdf", dpi=600)
        #f2.savefig("D:\Document\Github\python01\Figure_trace.eps", format="eps")
    elif _OSList[_OSName] == 'Linux':
        if not (os.access("Figure_Quad.eps", os.W_OK)):      # Check the file is Writable
            os.chmod ("Figure_trace.eps", 0o777)              # if the file is not writable, chmod 777

        f1.savefig("Figure_trace.eps", format="eps")
    else :
        raise ArithmeticError('OS Name is not assigned!! please check the nlp_service.py')


    #--------------------------------------------------------------------
    # Third plot : Index for QP
    # --------------------------------------------------------------------
    if args.quantize_method > 0 : 
        z = nlp.l_infindextrend
        y = nlp.l_index_trend
        x = np.arange(0, len(y), 1)
        plt.plot(x, y, 'r', label="proc_index")
        plt.plot(x, z, 'b', label="inf_index")
        plt.grid()
        plt.show()

    #--------------------------------------------------------------------
    # Fourth plot :: Animation
    # --------------------------------------------------------------------

    plt.contour(X, Y, Z, levels)
    plt.plot(Optimal_Point[0], Optimal_Point[1], 'r*', label="Optimal Point")

    plt.ylim((ymin, ymax))
    plt.xlim((xmin, xmax))
    plt.ylabel('Y-Coordinate')
    plt.xlabel('X-Coordinate')
    plt.legend(loc=2)
    f2 = plt.gcf()

    _framnes = (2000, Best_Step + 2)[Best_Step < 1998]
    if Best_Step < 50:
        _dpi = 300
    elif Best_Step < 300:
        _dpi = 150
    else:
        _dpi = 80

    print("_framnes : ", _framnes, "dpi : ", _dpi )
    line, = plt.plot([], [], 'bo-', label="Trace")
    anim = animation.FuncAnimation(f2, animate, frames = _framnes, fargs=(np_rX, line), interval=20, blit=True)

    if args.PrintoutGIF == 1:
        FileName = GetAlgorithmName(args) + '.gif'
        anim.save(FileName, dpi=_dpi, writer='imagemagick')

        if _OSList[_OSName] == 'Windows':
            plt.show()
        else:
            print('No available animated pictures on window. Check the folder and find the file %s' %(FileName))

    Nr = nlp_Records()
    Nr.save_data(rEpsilon, rCost, rX, rLambda, iBestIdx, args.functionID)
    Nr.save_data_as_file()


# --------------------------------------------------------------------
# plot for multiple test
# --------------------------------------------------------------------
def Conv_strvec_to_floatvec(_invec, _dim=2):
    _outvec = []
    
    for _k, _data in enumerate(_invec):
        _temp = np.delete(_data, np.where(_data ==''))
        _temp = _temp.astype(np.float64)
        _outvec.append(_temp)
    
    return _outvec

def Set_data_for_plot(_init_vec, _ResultData):
    _dim = np.shape(_init_vec)[0]
    _num = len(_ResultData)
    board = [[0 for i in range(_dim)] for j in range(_num)]

    for _k, _rx in enumerate(_ResultData):
        board[_k][0] = _rx[0]
        board[_k][1] = _rx[1]

    np_rX = np.array(board, dtype=np.float64)
    return np_rX

def Set_Limitation(np_rX, Optimal_Point, lLimit):
    [gxmin, gxmax, gymin, gymax] = lLimit

    _min_vec = np.min(np_rX, axis=0) - 0.05
    _max_vec = np.max(np_rX, axis=0) + 0.05

    [xmin, ymin] = _min_vec
    [xmax, ymax] = _max_vec

    if xmin < gxmin: gxmin = xmin
    if xmax > gxmax: gxmax = xmax
    if ymin < gymin: gymin = ymin
    if ymax > gymax: gymax = ymax

    gxmin = (gxmin, Optimal_Point[0] - 0.05)[gxmin > Optimal_Point[0]]  # A=(False_Result, True_Result)[Condition]
    gxmax = (gxmax, Optimal_Point[0] + 0.05)[gxmax < Optimal_Point[0]]
    gymin = (gymin, Optimal_Point[1] - 0.05)[gymin > Optimal_Point[1]]
    gymax = (gymax, Optimal_Point[1] + 0.05)[gymax < Optimal_Point[1]]

    return [gxmin, gxmax, gymin, gymax]

def plot_trace_data(plt, np_rX, _AlgorithmName, _Quantize=False):
    _FinalPoint = 'o' if _Quantize else 'd'
    
    plt.plot(np_rX[:, 0], np_rX[:, 1], label = _AlgorithmName)
    plt.plot(np_rX[-1, 0],np_rX[-1, 0], _FinalPoint)

    plt.tight_layout()
    plt.legend(loc='upper left')
    return plt

def plot_trace_map(np_rX, _SideData, nlp):
    #  _SideData = [_FunctionId, _AlgorithmName, _AlgorithmIndex]
    # Data Management
    _Function_ID    = _SideData[0]
    _AlgorithmName  = _SideData[1]
    _AlgorithmIndex = _SideData[2]
    _init_vec       = _SideData[3]
    _initial_info   = _SideData[4]

    # Function Setting
    cFunctions  = nf(_init_vec, _initial_info, _func_id=_Function_ID)
    Optimal_Point = cFunctions.Optimal_Point()
    # --------------------------------------------------------------------
    # plot dub-figure (trend of epsilon, cost, Lmabda and trace of X)
    # --------------------------------------------------------------------
    _Level = Get_LevelList(nlp)
    level_list = [[2e-8, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384], \
                  [2e-8, 2e-4, 2e-2, 2e-1, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.65, 2.75, 3, 4, 5, 6], \
                  []]
    levels = level_list[_Level]

    # --------------------------------------------------------------------
    # make data for plotting
    # --------------------------------------------------------------------
    #np_rX = Set_data_for_plot(_init_vec, _ResultData)

    lLimit = [99999, -99999, 99999, -99999]
    lLimit= Set_Limitation(np_rX, Optimal_Point, lLimit)
    [gxmin, gxmax, gymin, gymax] = lLimit

    X, Y, Z = Description_of_Object_Function(nlp, lLimit)

    # --------------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------------
    plt.contour(X, Y, Z, levels=15)
    plt.plot(Optimal_Point[0], Optimal_Point[1], 'r*', label="Optimal Point")

    plt.ylim((gymin, gymax))
    plt.xlim((gxmin, gxmax))
    plt.ylabel('Y-Coordinate')
    plt.xlabel('X-Coordinate')

    return plt, cFunctions

def plot_trace_for_test_nlp(c_nlp, _ResultData, _param_func):
    # Set Parameters 
    [_FunctionId, _AlgorithmIndex, _init_vec, _init_info] = _param_func
    _AlgorithmName  = strAlgorithmName[_AlgorithmIndex]
    _SideData       = [_FunctionId, _AlgorithmName, _AlgorithmIndex, _init_vec, _init_info]

    # Data Extraction : Data = [_convetiuonal_data[Algorithm], _Quantized_data[Algorithm]]
    l_rX = []
    for _k, _data in enumerate(_ResultData):
        _rX   = Set_data_for_plot(_init_vec, _data)
        l_rX.append(_rX)
    _ref_data = l_rX[0]
    
    # Plotting
    plt, c_nf   = plot_trace_map(_ref_data, _SideData, c_nlp)
    for _k, _data in enumerate(l_rX):
        _bQuant = True if _k==1 else False
        _AlgName= ('Quantized ' if _bQuant else '') + _AlgorithmName
        plt = plot_trace_data(plt, _data, _AlgName, _bQuant)

    #plt.show()

    # Show the plotting and save files
    f1 = plt.gcf()
    _funcname = c_nf.function_name[_FunctionId]
    _descript = _funcname + "_" + _AlgorithmName
    _filename = os.path.join("./Results_and_backup", _descript)
    if _OSList[_OSName] == 'Windows':
        plt.show()
        f1.savefig(_filename + ".pdf", format="pdf", dpi=600)
        #f1.savefig("D:\Document\Github\python01\Figure_Quad.eps", format="eps")
        #f1.savefig("C:\\Users\Jinwuk_surface\Downloads\Jinwuk_Work\Github\python01\Figure_Quad.eps", format="eps")
    elif _OSList[_OSName] == 'Linux':
        if not (os.access(_filename + ".eps", os.W_OK)):      # Check the file is Writable
            os.chmod (_filename + ".eps", 0x777)              # if the file is not writable, chmod 777
        f1.savefig(_filename + ".eps", format="eps")
    else :
        raise ArithmeticError('OS Name is not assigned!! please check the nlp_service.py')

def plot_objective_function(c_nlp, _SideData):
    #  _SideData = [_FunctionId, _Limit_X, _Limit_Y, _init_vec, _initial_info]
    # Data Management
    _Function_ID    = _SideData[0]
    _Limit_X        = _SideData[1]
    _Limit_Y        = _SideData[2]
    _init_vec       = _SideData[3]
    _initial_info   = _SideData[4]

    # Function Setting
    cFunctions  = nf(_init_vec, _initial_info, _func_id=_Function_ID)
    Optimal_Point = cFunctions.Optimal_Point()

    lLimit = [_Limit_X[0], _Limit_X[1], _Limit_Y[0], _Limit_Y[1]]
    [gxmin, gxmax, gymin, gymax] = lLimit

    X, Y, Z = Description_of_Object_Function(c_nlp, lLimit)

    # --------------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------------
    fig     = plt.figure()
    ax3d    = plt.axes(projection="3d")

    ax3d.plot_surface(X, Y, Z, cmap="inferno")
    ax3d.plot(Optimal_Point[0], Optimal_Point[1], 'r*', label="Optimal Point")

    #plt.ylim((gymin, gymax))
    #plt.xlim((gxmin, gxmax))
    ax3d.set_ylabel('Y-Coordinate')
    ax3d.set_xlabel('X-Coordinate')
    ax3d.set_zlabel('Objective Function $f(x)$')

    plt.show()

    print("[Debug] %s : %d " %(_filename_(), _lineno_()))
# ==========================================================
# Parsing argument and setting parameters
# ==========================================================
def ParseArgument(debugfrequency, training_steps):
    # parse command line arguments
    Algorithm_Help_String = "Search Algorithm \n \
                            [0] Gradient Descent   \n \
                            [1] Conjugate Gradient (Polak-Riebel) \n \
                            [2] Conjugate Gradient (Fletcher-Reeves) \n \
                            [3] Quasi Newton \n \
                            [4] Adagrad [5] AdaDelta [6] RMSProp [7] ADAM [8] Momentum [9] Nestrov"

    parser = argparse.ArgumentParser(
        prog='nlp_main01_test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
                                ------------------------------------------------
                                Nonlinear Optimization Testbench
                                                  
                                ------------------------------------------------
                                Example : nlp_main01.py -a 4 -w 0 -s 2
                                Meaning : Using Quasi-Newton BFGS with Step size evaluated by Line Search
                                '''))
    parser.add_argument('-a', '--algorithm', help=Algorithm_Help_String, default=1, type=int)
    parser.add_argument('-w', '--weight',
                        help="Weight for Quasi Newton WEIGHT x (DFP) + (1 - WEIGHT)(BFGS). The lower limit is 0.0 and highest limit is 1.0",
                        type=float, default=0.0)
    parser.add_argument('-df', '--debugfreq',
                        help="Debugging Frequency. Ir means that each time of (time mod df)==0 prints out the debugginf infot",
                        type=int, default=debugfrequency)
    parser.add_argument('-s', '--stepsize',     help="Stepsize Rule [0] Constant [1] Armijo Rule [2] Line Search [3] Time Decaying [4] Trust Region", type=int)
    parser.add_argument('-q', '--quantize',     help="Level to Quantization Parameter for Quantized Algorithm", type=int, default=-1)
    parser.add_argument('-qm','--quantize_method',help="Quantization Method [0] Simple [1] Automatic [2] (My proposed) Annealing Like", type=int, default=-1)
    parser.add_argument('-i', '--initial_point',help="Initial Point", nargs='+', default=None, type=float)
    parser.add_argument('-t', '--iterations',   help="Number of Iterations", default=training_steps, type=int)
    parser.add_argument('-f', '--functionID',   help="Objective Function ID [0] Rosenblatt [1] Ackley ", default=0, type=int)
    parser.add_argument('-p', '--PrintoutGIF',  help="print out GIF animation [0] False [1] True", default=0, type=int)
    parser.add_argument('-l', '--LearningRate', help="Initial Learning Rate", default=LocalNLP.Constant_StepSize, type=float)
    parser.add_argument('-n', '--NoShowFigure', help="Don't show the figure of result [0] Show [1] No Show", default=0, type=int)
    parser.add_argument('-g', '--MomentGamma',  help="Gamma value for SGD moment and Nestrov Acceleration", default=0.9, type=float)
    parser.add_argument('-fo','--FastParameter',help="Parameter for Fast Armijo Algorithm", default=0.5, type=float)
    parser.add_argument('-d', '--dimension',    help="Dimension of Parameter Default: 2", default=0, type=int)
    parser.add_argument('-qt','--quite_mode',   help="Quite Process : defalut = 0", default=0, type=int)
    parser.add_argument('-msg','--message',     help="Special Message for Test : defalut = ''", default='', type=str)
    parser.add_argument('-rf','--random_fix',   help="Initialize as [0] random or [1] fix", default=1, type=int)
    parser.add_argument('-em','--emergency',    help="When some search problems caused by quantization, it should be active", default=0, type=int)
    args = parser.parse_args()

    return args

def ParameterSet(args, _Initial_point):
    # Set Algorithm Type : It should be modified when a new algorithm is added
    #SearchAlgorithm = args.algorithm >> 1
    
    SearchAlgorithm = args.algorithm

    if SearchAlgorithm == 1 or SearchAlgorithm == 2:
        CGmethod = SearchAlgorithm - 1 
    else:
        CGmethod = -1

    AlgorithmWeight = np.clip(args.weight, 0.0, 1.0)
    debugfrequency  = args.debugfreq

    StepSizeRule = args.stepsize
    if args.stepsize != None:
       StepSizeRule = np.clip(args.stepsize, 0, len(LocalNLP.stepfunc))

    '''
    AlgorithmIndex  = [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8]
    SearchAlgorithm = AlgorithmIndex[args.algorithm]
    CGmethod        = ((args.algorithm & 2) >> 1) & (args.algorithm & 1)
    AlgorithmWeight = np.clip(args.weight, 0.0, 1.0)
    debugfrequency  = args.debugfreq

    StepSizeRule = _LStepSizeRule[args.algorithm]   # Default Step Size rule for each Algorithm
    if args.stepsize != None:
       StepSizeRule = np.clip(args.stepsize, 0, len(LocalNLP.stepfunc))
    '''

    # Qunatization Parameter : args.quantize is Initial index of Qp = 2^{args.quantize > 0}
    bQuantize = True
    if args.quantize == -1 and args.quantize_method == -1:  
        bQuantize = False
    elif args.quantize > -1 and args.quantize_method == -1:
        args.quantize_method = 0
    elif args.quantize == -1 and args.quantize_method > -1:
        args.quantize = 0 
    else: pass        
    Qparam    = args.quantize

    # Set Initial Point - 1
    _init_msg = _shline + "\n"
    if args.initial_point != None:
        _bGiven_Initial_point = True
        # The default value of args.initial_point is None. If it exists, it must be written according to it.
        X = np.array(args.initial_point, dtype=np.float64)
        _init_msg += "Set Initial Point by args.initial_point @%s" %str(X) + "\n"
    else:
        _bGiven_Initial_point = False
        if args.dimension > 1:
            X = np.array(np.zeros(args.dimension), dtype=np.float64)
            _init_msg += "Set Initial Point by args.dimension @%s" %str(X) + "\n"
        else:
            X = np.array(_Initial_point, dtype=np.float64)
            _init_msg += "Set Initial Point by System Default @%s" %str(X) + "\n"

    # Emergency Process for Qunatization
    if bQuantize:
        args.emergency = True if args.emergency == 1 else False
    else: 
        args.emergency = False

    # Other Control Parameters
    training_steps = args.iterations
    args.quite_mode     = False if args.quite_mode == 0 else True
    args.NoShowFigure   = True  if (args.quite_mode or args.NoShowFigure==1) else False
    args.random_fix     = True  if args.random_fix == 1 else False 


    # Set Initial Point - 2
    _init_msg += "Use fixed initial point\n" if args.random_fix else "Use random initial point\n"
    _init_msg += _shline

    #print-out the information of operation 
    ns_print(_init_msg, Active=not args.quite_mode)

    return [SearchAlgorithm, CGmethod, AlgorithmWeight, debugfrequency, StepSizeRule, \
            Qparam, bQuantize, X, training_steps, _bGiven_Initial_point]
