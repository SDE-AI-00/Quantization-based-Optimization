#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# NeurIPS Research Test for Quantization Parameters (base, eta, warm-up period and others)
# Working Directory :
#  Service Functions for torch_nn02.py
###########################################################################
_description = '''\
====================================================
service_test.py : Based on torch module
                    Written by 
====================================================
Example : python service_test.py 
'''
import argparse
import textwrap

from service_func import Result_Analysis
from service_func import modify_quantization_param
from service_func import local_operation_class

#=============================================================
# Definitions : parser.parse_args(['--sum', '7', '-1', '42'])
#=============================================================
def _ArgumentParse(_intro_msg, L_Param, bUseParam=False):
    parser = argparse.ArgumentParser(
        prog='service_test.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(_intro_msg))
    parser.add_argument('-bs', '--base', help="[quantization] base (Default:2)",
                        type=int, default=2)
    parser.add_argument('-et', '--eta', help="[quantization] eta (Default:524288.0)",
                        type=float, default=524288.0)
    parser.add_argument('-kp', '--kappa', help="[quantization] kappa (Default:2.0)",
                        type=float, default=2.0)
    parser.add_argument('-wp', '--warmp_up_period', help="[quantization] warmp_up_period (Default:0.2)",
                        type=float, default=0.2)
    parser.add_argument('-db', '--debug', help="[quantization] Debug (Default:0 <False>) ",
                        type=int, default=0)
    parser.add_argument('-ts', '--test', help="[quantization] test (Default:0 <False>) ",
                        type=int, default=0)
    parser.add_argument('-ky', '--key_data', help="[quantization] key_data (Default: 'warm_up_period') ",
                        type=str, default='warm_up_period')
    parser.add_argument('-dir','--directory', help="[Test_Result_Analysis] Target Directory",
                        type=str, default='./result')
    parser.add_argument('-tid','--test_style', help="[Test_Result_Analysis] Test Style [0]Test_QParam_prc [1]Test_warm_up_period [(2)]Test_Result_Analysis",
                        type=int, default=2)
    # -------------------------------------------------------------
    # Not used Parameter but initialization
    # -------------------------------------------------------------
    parser.add_argument('-l_data','--l_data', help="local test data parameters (Not used)",
                        type=float, default=[524288.0, 262144.0, 131072.0, 65536.0, 32768.0, 16384.0, 8192.0, 4096, 2048, 1024])
    parser.add_argument('-l_qparam','--l_qparam', help="local quantization parameters (Not used)",
                        type=dict, default=dict(base='bs', eta='et', kappa='kp', warm_up_period='wp'))

    # -------------------------------------------------------------
    # Parsing and post processing
    # -------------------------------------------------------------
    args = parser.parse_args()
    args.debug   = True if args.debug == 1 else False
    args.test    = True if args.test  == 1 else False

    print("\n", _intro_msg)
    return args

#=============================================================
# Test Processing
#    Initial QP            : 262144.0
#    eta                   : 524288.0
#=============================================================
def Test_QParam_prc(_args):
    l_data      = [524288.0, 262144.0, 131072.0, 65536.0, 32768.0, 16384.0, 8192.0, 4096, 2048, 1024]
    l_qparam    = dict(base='bs', eta='et', kappa='kp', warm_up_period='wp')

    c_op        = local_operation_class(_args)

    c_op._op(_key="eta", l_data=l_data)
    #c_op._op(_key="warmp_up_period", l_data=l_data)
    #c_op.file_rename()

#=============================================================
# Test Processing : To get the std.dev of results for multiple test
#=============================================================
def Test_Result_Analysis(_args):
    RA = Result_Analysis(_args=_args)
    RA.operation()
    RA.close()

#=============================================================
# Test Modifying Quantization Parameter
#=============================================================
def Test_warm_up_period(_args):
    c_mqp = modify_quantization_param(_args=_args)
    c_mqp.operation(_key='warmp_up_period', _data=_args.warmp_up_period)

#=============================================================
# Test Processing Main Body
#=============================================================
if __name__ == "__main__":
    Test_functions = []
    Test_functions.append(Test_QParam_prc)
    Test_functions.append(Test_warm_up_period)
    Test_functions.append(Test_Result_Analysis)
    _args = _ArgumentParse(_intro_msg=_description, L_Param=[])

    Test_functions[_args.test_style](_args)

    #Test_Result_Analysis(_args)
    #Test_warm_up_period(_args)

    print("=============================================================")
    print("Process Finished!!")
    print("=============================================================")
