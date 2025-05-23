#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################
# test_proc.py
# Plot an arbitrary test function
# Example code is test_nlp.py and read README.md for appropriate usage
###########################################################
import numpy as np

_description = '''\
====================================================
test_proc.py : Plot an arbitrary test function
                    using test_proc.py 
====================================================
example : test_proc.py -iqp -2 -i 2.3 
'''
g_colors    = ["darkred",        "saddlebrown", "darkorange", "darkgoldenrod", "darkolivegreen", "darkgreen",
                "darkslategray", "cadetblue",   "darkblue",   "indigo",        "magenta",        "crimson"]

#from numpy import arange
from matplotlib import pyplot
import test_proc_function as c_Lib
from datetime import datetime

# multimodal function
class plotting_function:
    def __init__(self, c_test_function_ID=None):
        # Guidance
        self.c_args         = c_Lib.guide_class(_intro_msg=_description, L_Param=[])
        self.args           = self.c_args.args
        self.function_ID    = self.args.test_fuinction_id
        # Main Test function Library
        self.c_function_list= c_Lib.test_function_list()
        self.c_function     = self.c_function_list(self.function_ID)
        # define range for input
        [r_min, r_max]      = self.c_function.get_range()
        self.r_min          = r_min
        self.r_max          = r_max
        self.increments     = self.args.increments
        self.test_function  = self.c_function._function
        self.optima         = self.c_function.get_optima()
        self.Q_obj          = c_Lib.quantization(_iqp=self.args.initial_quantization_power)
        self.c_lib          = c_Lib.function_library(r_min=self.r_min, r_max=self.r_max, _increments=self.increments)
        # sample input range uniformly at 0.1 increments
        self.inputs         = self.c_lib.inputs
        self._stop_prob     = 1.0/(1.0 * np.size(self.inputs))
        self.time_idx       = 0
        self.c_lib.set_params(_x        =self.args.initial_point,
                              _obj_func =self.test_function,
                              _qfunction=self.Q_obj,
                              b_init    =True)
        # Colors of plotting
        self.l_colors       = g_colors
        #For operation msg or Log
        self.log_msg        = []
        self._msg           = None
        self._pmsg          = None
        self.b_update_on    = False
        self._update_counter= 0
        # For plotting Data
        self._x             = []
        self._y             = []
        self._loc_text      = []
        self.init_plotting_data()

        # For Generating Final Message L Fundamental
        self.final_msg      =[]
        self.final_msg.append("=================================================================")
        self.final_msg.append("Test Function   [%d] %s" %(self.function_ID, self.c_function._name))
        self.final_msg.append(" Equation        %s" %self.c_function.equation)
        self.final_msg.append(" Optimal value : %f  @ %f" %(self.c_function.x_optima,
                                                            self.c_function._function(self.c_function.x_optima)))
    #--------------------------------------------------------
    # General Operation
    # --------------------------------------------------------
    def general_operation(self):
        # compute targets
        results = self.test_function(self.inputs)
        self.c_lib.find_min(results)
        self.c_lib.find_max(results)
        # create a line plot of input vs result
        pyplot.plot(self.inputs, results, color='black', label='Original Objective Function')

        return results
    #--------------------------------------------------------
    # Quantization
    # --------------------------------------------------------
    def quantization(self, results):
        Q_result = self.Q_obj(results)
        pyplot.plot(self.inputs, Q_result, color='limegreen', linestyle='--', label='Quantization')

        return Q_result
    #--------------------------------------------------------
    # Plot point
    # --------------------------------------------------------
    def plot_optimal(self):
        # define optimal input value
        x_optima = self.optima
        # draw a vertical line at the optimal input
        pyplot.axvline(x=x_optima, ls='--', color='red')
        self.plot_point(_x=x_optima)

    def plot_point(self, _x, _color='ro'):
        _y = self.test_function(_x)
        pyplot.plot(_x, _y, _color)

    def plot_point_seq(self, _x, _f, _fQ):
        pyplot.plot(_x, _f, 'ro')
        pyplot.plot(_x, _fQ, 'go')

    def plot_final_processing(self, _title, _k=0, _offset=0):
        _loc = ['upper left', 'upper left', 'upper right', 'upper left', 'upper left', 'upper left', 'upper left']
        pyplot.title(_title[_k])
        pyplot.legend(loc=_loc[_k+_offset])
        pyplot.xlabel('Input $x \in \mathbf{R}$')
        pyplot.ylabel('$f(x)$')
        pyplot.show()

    def plot_arrow(self, _start, _finish, _scaling = 0.95):
        _sx, _sy = _start[0], _start[1]
        _dx, _dy = _finish[0]*_scaling - _sx, _finish[1]*_scaling - _sy
        pyplot.arrow(_sx, _sy, _dx, _dy, head_width=0.125, fc='black', ec='black')

    #--------------------------------------------------------
    # LOG for Main Operation
    # --------------------------------------------------------
    def init_log(self):
        self._msg   = ""
        self._pmsg  = ""
        _x          = self.args.initial_point
        _f, _fQ     = self.c_lib.compute_state(_x)
        _guide_str  = "t=%d" %self.time_idx
        self.plot_point_seq(_x, _f, _fQ)
        pyplot.text(x=_x + 0.3, y=_f, s=_guide_str)

    def LOG_PreProcessing(self, _k, _x, _f, _fQ, _pfQ):
        self._msg   = '[{0:2d}] x:{1: 3.5f}  f:{2: 2.5f}  fQ:{3: 2.5f}  _pfQ:{4: 2.5f}  '.format(_k, _x, _f, _fQ, _pfQ)
        self._pmsg  = ""
        #self.plot_point_seq(_x, _f, _fQ)

    def LOG_MainProcessing(self, _x, _f, _fQ, _print_time=False):
        self._update_counter += 1
        self._pmsg = "Pb: %2.5f  " % (self.c_lib.get_transition_probability())
        self.plot_point_seq(_x, _f, _fQ)
        if _print_time:
            _guide_str  = "t=%d" %(self.time_idx+1)
            pyplot.text(x=_x+0.3, y=_f, s=_guide_str)
        else: pass

    def LOG_PostProcessing(self, _pQP, _optimal):
        _cQP        = self.Q_obj.get_QP()
        _pb_atom_msg= "correct: %d  total: %d  " %(self.c_lib.get_atom_of_transition_probability())
        self._msg   +="prQP:{0: 6.2f} crQP:{1: 4.2f} Optimal:{2: 6.5f}  ".format(_pQP, _cQP, _optimal)
        self.log_msg.append(self._msg + _pb_atom_msg + self._pmsg)

        if self.args.verbose == 0:
            print(self._msg, self._pmsg)
        else:
            if self.b_update_on:
                print(self._msg, self._pmsg)
            else:
                _msg = '.' if self.args.verbose == 1 else ''
                print(_msg, end='')
    
    def write_out_LOG(self, _final_msg):
        self.final_msg.append(_final_msg)
        self.final_msg.append("=================================================================")
        # For Log File
        with open(self.args.logfile, 'w') as f:
            for _msg in self.log_msg:
                f.write(_msg + '\n')
            for _msg in self.final_msg:
                f.write(str(_msg) + '\n')

        # For command Window
        for _msg in self.final_msg:
            print(_msg)

    #--------------------------------------------------------
    # Main Operation
    # --------------------------------------------------------
    def inner_operation(self, _k, _minfunc, _optimal):
        # The innre-loop for the for-clause
        # common Processing
        _x, _f, _fQ = self.c_lib.one_step_process()
        _pfQ = self.c_lib.get_state()
        # Log Record
        _pQP = self.Q_obj.get_QP()
        self.LOG_PreProcessing(_k=_k, _x=_x, _f=_f, _fQ=_fQ, _pfQ=_pfQ)
        # Main Comparison
        if _fQ <= _pfQ:
            _optimal = _x
            self.Q_obj.increse_QP()
            _nstate, _y = self.c_lib.compute_state(_optimal)
            _minfunc = _y if _minfunc > _y else _minfunc
            self.c_lib.set_state(_nstate)
            # Log Record
            self.b_update_on = True
            self.LOG_MainProcessing(_x, _f, _fQ, _print_time=True)
        else:
            # Log Record
            self.b_update_on = False
        # Log Record
        self.LOG_PostProcessing(_pQP=_pQP, _optimal=_optimal)

        return _minfunc, _optimal

    def stop_condition(self):
        _PB     = self.c_lib.get_transition_probability(_verbose=False)
        # Stop Condition : 즉, stop probability (i.e. 1/np.size(self.input)) 보다 작으면 stop : 더 움직일 수 없다.
        _ret    = (_PB < self._stop_prob)
        return _ret
    def operation(self):
        # initial process
        _results = self.general_operation()
        _optimal = self.args.initial_point
        _iteration=self.args.iteration
        _minfunc = 9999999999999.9
        # For LOG
        self.init_log()
        #Sequential Process
        for _k in range(_iteration):
            _minfunc, _optimal = self.inner_operation(_k=_k, _minfunc=_minfunc, _optimal=_optimal)
            #Stop Condition
            if self.stop_condition(): break
            else: self.time_idx = _k+1

        # Quantization Level
        self.quantization(results=_results)
        # plot maximum point
        #self.plot_point(_x=2.3)
        # plot minimal point
        #self.plot_optimal()

        l_title = ['Result Optimal Point=%f min f(x)=%f, @t=%d ' %(_optimal, _minfunc, self.time_idx)]
        self.plot_final_processing(_title=l_title)
        self.write_out_LOG(_final_msg=l_title)

    #--------------------------------------------------------
    # function plot operation
    # --------------------------------------------------------
    # 그림 그리기를 위한 data 초기화 부분
    def init_plotting_data(self):
        # 총 4번의 update를 가정하자. 위의 경우는 실제 실험결과이다.
        #_x = [self.args.initial_point, -0.24253, 2.71957, 7.39121, 5.14150, -0.64285, -2.47483, , 5.12772, 5.15291]
        self._x  = [self.args.initial_point, -0.24253, 2.71957, 5.14150, 5.14150]
        #_y 의 위치는 Quantization Error str이 위치하는 좌표이다.
        self._y  = [3.51, 0.5, 1.5, 2.25, 2.85, 0.0, 0.0]

        self._loc_text.append([(4.0, 3.1), (6.3, 1.8), (5.3, 4.15)])    #_k=0 False idx=0
        self._loc_text.append([(3.0, 0.7), (0.95, 1.7), (0.95, 0.22)])  #_k=0 True  idx=1
        self._loc_text.append([(3.0, 2.0), (0.95, 1.5), (0.95, 2.5)])   #_k=1 False idx=2
        self._loc_text.append([(5.5, 2.5), (4.0, 3.0), (4.0, 1.5)])     #_k=1 True  idx=3
        self._loc_text.append([(5.5, 3.0), (4.0, 2.5), (4.0, 3.5)])     #_k=2 False idx=4
        self._loc_text.append([(2.0, 0.0), (6.3, 0.1), (6.4, -0.3)])     #_k=2 True  idx=5
        self._loc_text.append([(2.0, 0.65),(6.0, 0.6), (6.0, 0.0)])     #_k=3 False idx=6

    def stage_guidance(self, _x, _f, _fQ, _idx, _offset=0):
        _box1       = {'boxstyle': 'round', 'ec': (0.0, 0.0, 0.0), 'fc': (1.0, 1.0, 1.0)}
        _guide_str  = ["Quantization error $\\varepsilon Q_p^{-1}(t_%d)$" %_idx,
                      "$f(x_%d)$" %_idx,
                      "$f(x_%d)^Q$" %_idx]
        _coidx      = _idx + _offset
        _y          = self._y
        _location   = [(_x, _y[_coidx]), (_x, _f), (_x, _fQ)]
        _loc_text   = self._loc_text[_coidx]

        pyplot.arrow(_x, _f, 0, 0.9 * (_fQ - _f), head_width=0.125, linestyle='--', fc='black', ec='black')
        pyplot.annotate(_guide_str[0], xy=_location[0], xytext=_loc_text[0], fontsize=10, ha='center', bbox=_box1,
                        arrowprops=dict(facecolor='black', width=1, shrink=0.1, headwidth=8))
        pyplot.annotate(_guide_str[1], xy=_location[1], xytext=_loc_text[1], fontsize=10, ha='center', bbox=_box1,
                        arrowprops=dict(facecolor='black', width=1, shrink=0.1, headwidth=8))
        pyplot.annotate(_guide_str[2], xy=_location[2], xytext=_loc_text[2], fontsize=10, ha='center', bbox=_box1,
                        arrowprops=dict(facecolor='black', width=1, shrink=0.1, headwidth=8))

        if (_idx == 1 and _offset == 1):
            pyplot.annotate('Moving $f(t_1)^Q$ according to change of QP',
                            xy=_location[2], xytext=(_x, 0), fontsize=10, ha='left', bbox=_box1,
                            arrowprops=dict(facecolor='black', width=1, shrink=0.1, headwidth=8,
                                            connectionstyle="arc3, rad=-0.3"))
        elif _coidx == 4:
            pyplot.annotate('Moving $f(t_2)^Q$ according to change of QP',
                            xy=_location[2], xytext=(_x, 1.65), fontsize=10, ha='left', bbox=_box1,
                            arrowprops=dict(facecolor='black', width=1, shrink=0.1, headwidth=8,
                                            connectionstyle="arc3, rad=-0.3"))
        else:
            pass
# Plot Quantized data
    def post_processing(self, _k, Q_result, _point, _next=False):
        # General Processing
        l_colors    = ["darkred", "indigo", "darkgreen", "darkslategray"]
        _data       = Q_result[_k]
        _label_msg  = "Quantized Objective Function."

        _point_data = _point[_k]
        _x, _f, _fQ = _point_data[0], _point_data[1], _point_data[2]
        _nx,_nf,_nfQ= _point_data[3], _point_data[4], _point_data[5]

        self.plot_point_seq(_x=_x, _f=_f, _fQ=_fQ)
        if _next:
            _start_point    = [_x, _f]
            _finish_point   = [_nx, _nf]
            self.plot_point_seq(_x=_nx, _f=_nf, _fQ=_nfQ)
            pyplot.annotate(' ', xy=(_nx, _nf), xytext=(_x, _f), fontsize=10, ha='center',
                            arrowprops=dict(facecolor='black', width=1.0, shrink=0.1, headwidth=8))
            lx, lf, lfQ, _offset = _nx, _nf, _nfQ, _k + 1
        else:
            lx, lf, lfQ, _offset = _x, _f, _fQ, _k

        pyplot.plot(self.inputs, _data[1], color=l_colors[_k], linestyle='--', label=_label_msg)
        self.stage_guidance(lx, lf, lfQ, _idx=_k, _offset=_offset)

    def function_plot_operation(self):
        # Initial Setting
        _num_exp = 4
        _point   = []
        Q_result = []
        _title   = []
        # initial process
        _results = self.general_operation()
        _x       = self._x

        for _k in range(_num_exp):
            _qp     = self.Q_obj.get_QP()
            _state  = self.Q_obj(_results)

            _fQ, _f = self.c_lib.compute_state(_x[_k])
            _nfQ,_nf= self.c_lib.compute_state(_x[_k+1])
            _point.append([_x[_k], _f, _fQ, _x[_k+1], _nf, _nfQ])

            _p_tr   = self.c_lib.get_transition_probability()
            _p_st   = self.c_lib.get_state_probability(_all_state=_state, _x=_x[_k])
            _p_st_c = _p_tr - _p_st
            print("P_tr :{0: .5f}  S_tr :{1: .5f}  C_tr :{2: .5f}".format(_p_tr, _p_st, _p_st_c))

            _title.append("QP={0:.2f}".format(_qp))
            Q_result.append([_qp, _state])
            self.Q_obj.increse_QP()

        # manual input : default : _next= False,  Multiple plotting per Epoch : _next=True
        _next = False
        if _next:
            for _k in range(_num_exp):
                self.post_processing(_k=_k, Q_result=Q_result, _point=_point, _next=_next)
                self.plot_final_processing(_title, _k=_k, _offset=_k + (1 if _next else 0))
        else:
            _k, _next = 0, False
            self.post_processing(_k=_k, Q_result=Q_result, _point=_point, _next=_next)
            self.plot_final_processing(_title, _k=_k, _offset=_k + (1 if _next else 0))


    def manual_function_plot_operation(self):
        # initial process
        _results = self.general_operation()
        _optimal = self.args.initial_point
        _iteration=self.args.iteration
        _minfunc = 9999999999999.9
        # For LOG
        self.init_log()
        #Sequential Process
        for _k in range(_iteration):
            _minfunc, _optimal = self.inner_operation(_k=_k, _minfunc=_minfunc, _optimal=_optimal)
            #Stop Condition
            if self.stop_condition(): break
            else: self.time_idx = _k+1

        # Quantization Level
        self.quantization(results=_results)
        # plot maximum point
        #self.plot_point(_x=2.3)
        # plot minimal point
        #self.plot_optimal()

        l_title = ['Result Optimal Point=%f min f(x)=%f, @t=%d ' %(_optimal, _minfunc, self.time_idx)]
        self.plot_final_processing(_title=l_title)
        self.write_out_LOG(_final_msg=l_title)
# =================================================================
# Main Routine
# =================================================================
# For Module processing
if __name__ == '__main__':
    #c_plot_func     = plotting_function()

    try :
        c_plot_func     = plotting_function()
    except Exception as e:
        print("Error Occur!! Error Message : %s" %e)
        exit()

    if c_plot_func.args.operation_mode == 0:
        c_plot_func.operation()
    elif c_plot_func.args.operation_mode == 1:
        c_plot_func.function_plot_operation()
    elif c_plot_func.args.operation_mode == 2:
        c_plot_func.manual_function_plot_operation()
    else:
        print("This mode is not implemented yet : mode %d" %c_plot_func.args.operation_mode)


