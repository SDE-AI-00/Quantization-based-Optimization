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


# =============================================================
# Test Processing
# =============================================================
if __name__ == "__main__":
    gs = GPU_service()
    # Test 2
    gs.GPU_setting_func()
    # Test 1
    gs.GPU_info_summary()

