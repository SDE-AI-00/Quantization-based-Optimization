#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# NeurIPS Research Test
# Working Directory :
###########################################################################
import shutil

_description = '''\
====================================================
service_func.py : 
                    Written by 
====================================================
Example : service_func.py
'''
#=============================================================
# Definitions
#=============================================================
import argparse
import textwrap
import torch
import os
import gc
import glob

from service_process_board import config_yaml
from service_process_board import operation_class
import torch_nn02
import my_debug as DBG
#=============================================================
# Local Service
#=============================================================
def _ArgumentParse(_intro_msg):
    parser = argparse.ArgumentParser(
        prog='test pytorch_inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(_intro_msg))

    parser.add_argument('-g', '--device', help="Using [(0)]CPU or [1]GPU",
                        type=int, default=0)
    parser.add_argument('-gn','--gpu_index', nargs='+', help="GPU index, Default value is =[0] (List), ex) -gn 0 1 ..",
                        type=int, default=[0])
    parser.add_argument('-qm','--quiet_memory', nargs='+', help="[0] Quiet GPU Memory information [1:Default] Print Memory Information",
                        type=int, default=[0])
    ## Cleaning GPU : for debug
    parser.add_argument('-clgpu', '--cleaning_gpu',
                        help="[0:Default] no cleaning [1] cleaning the GPU and exit()",
                        type=int, default=0)

    args = parser.parse_args()
    args.quiet_memory = True if args.quiet_memory == 0 else False
    args.cleaning_gpu = True if args.cleaning_gpu == 1 else False
    return args

#=============================================================
# GPU Service
#=============================================================
class GPU_service:
    def __init__(self):
        self.args       = _ArgumentParse(_intro_msg=_description)
        # self._args    = _ArgumentParse(_description, L_Param, bUseParam)
        self._device    = 'cuda' if self.args.device == 1 and torch.cuda.is_available() else 'cpu'
        self.GPU_DEVICE = torch.device(self._device)

        self.device_list = torch.cuda.list_gpu_processes(device=None)
        #self.gpu_device_id = self.args.gpu_test

        DBG.dbg("Initilization finished")

    def GPU_info_summary(self):
        if self.args.quiet_memory: return
        else : pass

        if self._device == 'cuda':
            print(torch.cuda.memory_summary(self.device))
        else:
            print("=============================================================")
            print(" This Application works on CPU not GPU ")
            print("=============================================================")

    # Ref : https://artiiicy.tistory.com/61
    def GPU_setting_func(self, GPU_DEVICE_ID=0):
        if self._device == 'cpu' : return
        else: pass

        # Set GPU Parallel Processing
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        _device_count  = torch.cuda.device_count()

        DBG.dbg("_device_count : ", _device_count)
        DBG.dbg("self.args.gpu_index : ", self.args.gpu_index)



        if _device_count > 1 and len(self.args.gpu_index) > 1:
            b_parallel_GPU = True
            _msg = ' '.join(list(map(str, self.args.gpu_index)))
        else :
            b_parallel_GPU = False
            _msg = str(self.args.gpu_index[0])
        os.environ["CUDA_VISIBLE_DEVICES"] = _msg
        print('os.environ["CUDA_VISIBLE_DEVICES"] = %s' %_msg)


        if len(self.args.gpu_index) == 1 and self.args.gpu_index[0] > 0:
            self.device = self._device + ":" + str(self.args.gpu_index[0])
        else:
            self.device = self._device

        gc.collect()
        if self.args.cleaning_gpu :
            print("Cleaning the GPU : ", self.device)
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        else:
            print("No Cleaning the GPU : ", self.device)

        print('Count of using GPUs:', _device_count)
        print('Current cuda device:', torch.cuda.current_device())
        print('Process cuda device:', self.device)
        print('Parallel Processing: %s' %('True' if b_parallel_GPU else 'False'))

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
        self.TR_accuracy= []
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
        print("=============================================================")
        print(" TEST ACCURACY ")
        print("-------------------------------------------------------------")
        for _k, _yaml_file in enumerate(self.yaml_files):
            _test_str = os.path.basename(_yaml_file)
            self.TE_accuracy.append(float(self.read_yaml_file(_yaml_file)))
            print("%f     %s" %(self.TE_accuracy[-1], _test_str))

        print("-------------------------------------------------------------")
        print(" TRAINING ACCURACY ")
        print("-------------------------------------------------------------")
        self.target_yaml_data_location = 6
        for _k, _yaml_file in enumerate(self.yaml_files):
            _test_str = os.path.basename(_yaml_file)
            self.TR_accuracy.append(float(self.read_yaml_file(_yaml_file)))
            print("%f     %s" %(self.TR_accuracy[-1], _test_str))

        print("-------------------------------------------------------------")
        print(" TRAINING Error ")
        print("-------------------------------------------------------------")
        for _k, _txt_file in enumerate(self.txt_files):
            _test_str = os.path.basename(_txt_file)
            self.TR_error.append(float(self.read_txt_file(_txt_file)))
            print("%f     %s" %(self.TR_error[-1], _test_str))

#=============================================================
# Test Quantization Parameter
#=============================================================
class   modify_quantization_param:
    def __init__(self, _args):
        self._args      = _args
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

#=============================================================
# Test General ALgorithm
#=============================================================
class local_operation_class:
    def __init__(self, _args):
        #self.op_param   = ["-m", "QtAdam", "-d", "FashionMNIST", "-e", "200", "-n", "ResNet", "-g", "1", "-l", "0.01"]
        self.args       = _args
        self.s_arg_data = "argdata.dat"
        self.l_data     = self.args.l_data
        self.l_qparam   = self.args.l_qparam

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
        _new_dir = os.path.join(result_op.data_dir, self.c_mqp._args.key_data)
        os.makedirs(_new_dir, exist_ok=True)
        # Rename Files
        _key_data = str(self.l_qparam[self.c_mqp._args.key_data])
        _value_data = str(self.l_data[_data_id])
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

# =============================================================
# Test Processing
# =============================================================
if __name__ == "__main__":
    gs = GPU_service()
    # Test 2
    gs.GPU_setting_func()
    # Test 1
    gs.GPU_info_summary()

