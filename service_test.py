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
from service_process_board import config_yaml
from service_process_board import operation_class
import my_debug as DBG
import torch_nn02
import os
import shutil
import numpy as np
import torch
import yaml
import glob
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
    parser.add_argument('-ky', '--key_data', help="key_data (Default: 'warm_up_period') ",
                        type=str, default='warm_up_period')
    parser.add_argument('-dir','--directory', help="Target Directory",
                        type=str, default='./result')

    args = parser.parse_args()
    args.debug   = True if args.debug == 1 else False
    args.test    = True if args.test  == 1 else False

    print(_intro_msg)
    return args

class   modify_quantization_param:
    def __init__(self):
        self._args      = _ArgumentParse(_intro_msg=_description, L_Param=[])
        self.cy         = config_yaml()
        self.file_name  = self.cy.get_config_file_name('quantization')
        self.data_sets  = self.cy.read_yaml_file(self.file_name)
        self._Qdata     = self.data_sets['Quantization']
        self._Qdata_keys= list(self._Qdata.keys())

    def set_param(self, _key, _data):
        self._Qdata[_key] = _data
    def get_param(self, _key):
        return self._Qdata[_key]
    def get_Qdata(self):
        return self._Qdata
    def operation(self, _key, _data):
        self.set_param(_key, _data)
        self.cy.write_yaml_file(_data=self.data_sets, _yaml_file_name=self.file_name)

class local_operation_class:
    def __init__(self):
        #self.op_param   = ["-m", "QtAdam", "-d", "FashionMNIST", "-e", "200", "-n", "ResNet", "-g", "1", "-l", "0.01"]
        self.s_arg_data = "argdata.dat"
        self.op_param   = self.read_argdata()
        self.c_mqp      = modify_quantization_param()

    def read_argdata(self):
        with open(self.s_arg_data, 'rt') as f:
            for _k, _line in enumerate(f.readlines()):
                _idx = _line.find("::")
                if _idx < 0:
                    _operation_param = _line.split()
                    if len(_operation_param) == 0:
                        print("There is not any data at Line : {0:3}".format(_k))
                        DBG.dbg("Check the file for this error", _active=True)
                        exit()
                    else:
                        pass
                else:
                    pass

        return _operation_param
    def file_rename(self, _data_id):
        L_param=['-rd', 'result']   # Dummy code
        result_op = operation_class(L_param=L_param, bUseArgParser=False)
        # Result File Gather
        _proc_files = []
        _proc_files.append(result_op.pickle_file_list[0])
        _proc_files.append(result_op.yaml_file_list[0])
        _proc_files.append(result_op.text_file_list[0])
        _proc_files.append(result_op.pt_file_list[0])
        # Make New Directory
        _new_dir = os.path.join(result_op.data_dir, c_op.c_mqp._args.key_data)
        os.makedirs(_new_dir, exist_ok=True)
        # Rename Files
        _key_data = str(l_qparam[c_op.c_mqp._args.key_data])
        _value_data = str(l_data[_data_id])
        _str_data = _key_data + _value_data
        for _file in _proc_files:
            _part = _file.split('_')
            _n = len(_part)
            _part.insert(_n - 1, _str_data)
            _new_file = '_'.join(_part)

            old_file = os.path.join(result_op.data_dir, _file)
            new_file = os.path.join(result_op.data_dir, _new_file)
            os.rename(old_file, new_file)

            _dst_file = os.path.join(_new_dir,_new_file)
            if os.path.exists(_dst_file):
                os.remove(_dst_file)
            else: pass
            shutil.move(src=new_file, dst=_dst_file)

    def _op(self, _key, l_data, bUseParam=True):
        torch_nn02.clean_result_directory()

        for _k, _data in enumerate(l_data):
            self.c_mqp.operation(_key=_key, _data=_data)
            torch_nn02.training(self.op_param, bUseParam=bUseParam)
            self.file_rename(_data_id=_k)

        torch_nn02.generate_notify(_target_file="config_quantization.yaml")
#=============================================================
# YAML Result Analysis
#=============================================================
class   Result_Analysis:
    def __init__(self, _args, yaml_ext="*.yaml", txt_ext="*.txt", _yaml_loc=10, _txt_def="[Epoch :   99]"):
        self.yaml_ext   = yaml_ext
        self.txt_ext    = txt_ext
        self.base_dir   = os.getcwd()
        self.target_dir = _args.directory
        self.work_dir   = os.path.join(self.base_dir, self.target_dir)
        self.yaml_files = []
        self.txt_files  = []
        self.TE_accuracy= []
        self.TR_error   = []
        print("Work Directory : ", self.work_dir)

        self.target_yaml_data_location  = _yaml_loc
        self.target_txt_data_def_str    = _txt_def
        self.get_yaml_files()
        self.get_txt_files()
        os.chdir(self.work_dir)

        #self.data_sets  = self.cy.read_yaml_file(self.file_name)
    def close(self):
        os.chdir(self.base_dir)
    def get_yaml_files(self):
        _target = os.path.join(self.work_dir, self.yaml_ext)
        self.yaml_files = glob.glob(_target)

    def get_txt_files(self):
        _target = os.path.join(self.work_dir, self.txt_ext)
        self.txt_files = glob.glob(_target)

    def read_yaml_file(self, file_path):
        with open(file_path, "r") as stream:
            _linedata = stream.readlines()
        _target_data_str = _linedata[self.target_yaml_data_location]
        _target_data = _target_data_str.split()

        return _target_data[-1]

    def read_txt_file(self, file_path):
        with open(file_path, "r") as stream:
            _linedata = stream.readlines()

        _target_data = -1
        for _k, _line in enumerate(_linedata):
            if self.target_txt_data_def_str in _line:
                _atom_data  = _line.split()
                _target_data= _atom_data[5]
            else: pass
        return _target_data

    def operation(self):
        for _k, _yaml_file in enumerate(self.yaml_files):
            _test_str = os.path.basename(_yaml_file)
            self.TE_accuracy.append(float(self.read_yaml_file(_yaml_file)))
            print("%f     %s" %(self.TE_accuracy[-1], _test_str))

        for _k, _txt_file in enumerate(self.txt_files):
            _test_str = os.path.basename(_txt_file)
            self.TR_error.append(float(self.read_txt_file(_txt_file)))
            print("%f     %s" %(self.TR_error[-1], _test_str))

#=============================================================
# Test Processing
#    Initial QP            : 262144.0
#    eta                   : 524288.0
#=============================================================
'''
if __name__ == "__main__":
    l_data      = [524288.0, 262144.0, 131072.0, 65536.0, 32768.0, 16384.0, 8192.0, 4096, 2048, 1024]
    l_qparam    = dict(base='bs', eta='et', kappa='kp', warm_up_period='wp')

    c_op        = local_operation_class()

    c_op._op(_key="eta", l_data=l_data)
    #c_op._op(_key="warmp_up_period", l_data=l_data)
    #c_op.file_rename()

    print("=============================================================")
    print("Process Finished!!")
    print("=============================================================")
#=============================================================
# Test Processing
#=============================================================
'''
if __name__ == "__main__":
    _args = _ArgumentParse(_intro_msg=_description, L_Param=[])
    RA = Result_Analysis(_args=_args)

    RA.operation()

    RA.close()
    print("=============================================================")
    print("Process Finished!!")
    print("=============================================================")
