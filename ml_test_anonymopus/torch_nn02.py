#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# NeurIPS Research Test
# Working Directory :
###########################################################################
_description = '''\
====================================================
torch_nn02.py : Based on torch module
                    Written by 
====================================================
Example : python torch_nn02.py 
'''

#=============================================================
# Definitions
#=============================================================
import os
import time
import torch
import torch.nn as nn
from Read_Data import MNIST_set
from Read_Data import CIFAR10_set
from Read_Data import CIFAR100_set
from Read_Data import FashionMNIST_set
from Read_Data import ImageNet_set
from Read_Data import STL10_set

from torch.utils.tensorboard import SummaryWriter
import gc
from tqdm import tqdm
#-------------------------------------------------------------
# Description of CNN, LeNet, ResNet
# Input  : 3 channel 32x32x3
#-------------------------------------------------------------
from torch_SmallNet import CNN
from torch_SmallNet import LeNet
from torch_resnet import ResNet as resnet_base
from torch_resnet import ResidualBlock
import torch_resnet as resnet_service
from torch_vgg import VGG
from torch_efficientnet import EfficientNet_model

from service_process_board import process_data_storage
from service_process_board import send_notify
from service_process_board import config_yaml
import my_debug as DBG

def ResNet(inputCH, outCH, num_layers=5):
    Lnum_layers = resnet_service.check_numlayers(num_layers)
    block = ResidualBlock
    model = resnet_base(num_layers=Lnum_layers, block=block, num_classes=outCH, inputCH=inputCH)
    return model

#=============================================================
# Function for Test Processing
#=============================================================
g_msg = []
# --------------------------------------------------------
# Service Function
# --------------------------------------------------------
def _sprint(msg, b_print=True):
    g_msg.append(msg)
    if b_print:
        print(msg)
    else:
        pass

def _write_operation(opPATH):
    with open(opPATH, 'wt') as f:
        for _msg in g_msg:
            f.write(_msg + "\n")
    g_msg.clear()
    print("Operation Result File : %s " %opPATH)

def Check_modelName(args):
    l_algorithm = ['SGD', 'Adam', 'AdamW', 'QSGD', 'QtAdamW', 'QtAdam', 'stdSGD', 'ASGD', 'NAdam', 'RAdam']
    b_correct = False
    for _name in l_algorithm:
        if args.model_name == _name:
            b_correct = True
            break
        else: pass

    if b_correct:
        print("Correct Model Name [{}]".format(args.model_name))
    else:
        DBG.dbg("Unexpected Model Name!! [{}]".format(args.model_name))
        DBG.dbg("Please Check the Model Name !!!")
        exit()

def Get_Network_Name(args):
    if args.data_set == 'MNIST' or args.data_set == 'FashionMNIST':
        _net_name = 'CNN'
    elif args.data_set == 'ImageNet':
        _net_name = args.vgg_arch
    else:
        _net_name = args.net_name

    return _net_name
# --------------------------------------------------------
# Parsing the Argument : parser.parse_args(['--sum', '7', '-1', '42'])
# --------------------------------------------------------
import argparse
import textwrap

def _ArgumentParse(_intro_msg, L_Param, bUseParam=False):
    parser = argparse.ArgumentParser(
        prog='torch_nn02.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(_intro_msg))

    parser.add_argument('-g', '--device', help="Using [(0)]CPU or [1]GPU",
                        type=int, default=0)
    parser.add_argument('-gn','--gpu_index', nargs='+', help="GPU index, Default value is =[0] (List), ex) -gn 0 1 ..",
                        type=int, default=[0])
    parser.add_argument('-l', '--learning_rate', help="learning_rate (default=0.001)",
                        type=float, default=0.001)
    parser.add_argument('-e', '--training_epochs', help="training_epochs (default=15)",
                        type=int, default=15)
    parser.add_argument('-b', '--batch_size', help="batch_size (default=100)",
                        type=int, default=100)
    parser.add_argument('-f', '--model_file_name', help="model file name (default='torch_nn02_CNN.pt')",
                        type=str, default="torch_nn02_CNN.pt")
    parser.add_argument('-m', '--model_name', help="model name 'SGD', 'Adam', 'AdamW', 'ASGD', 'NAdam', 'RAdam' (default='Adam')",
                        type=str, default="Adam")
    parser.add_argument('-n', '--net_name', help="Network name 'CNN', 'LeNet', 'ResNet', 'VGG' 'EfficientNet' (default='LeNet')",
                        type=str, default="LeNet")
    parser.add_argument('-d', '--data_set', help="data set 'MNIST', 'CIFAR10' (default='MNIST')",
                        type=str, default="MNIST")
    parser.add_argument('-a', '--autoproc', help="Automatic Process without any plotting [(0)] plotting [1] Automatic-no plotting",
                        type=int, default=0)
    parser.add_argument('-pi', '--proc_index_name', help="Process Index Name. It is generated automatically (default='')",
                        type=str, default='')
    parser.add_argument('-rl', '--num_resnet_layers', help="The number of layers in a block to ResNet (default=5)",
                        type=int, default=5)
    parser.add_argument('-qp', '--QParam', help="Quantization Parameter, which is read from the config_quantization.yaml file (default=0)",
                        type=int, default=0)
    parser.add_argument('-rd', '--result_directory', help="Directory for Result (Default: ./result)",
                        type=str, default="result")
    parser.add_argument('-sn', '--scheduler_name', help="Learning rate scheduler (Default : Constant Learning)",
                        type=str, default="constant")
    parser.add_argument('-ev', '--evaluation', help="Only Inference or Evaluation with a learned model [(0)] Training and Inference [1] Inference Only",
                        type=int, default=0)
    ## Only for Top Level Usage
    parser.add_argument('-bt', '--batchproc', help="Batch Processing with 'argdata.dat' file or Not [(0)] Single processing [1] Multiple Processing",
                        type=int, default=0)
    parser.add_argument('-ag', '--arg_data_file', help="Argument data File for Batch Processing (default: 'argdata.dat')",
                        type=str, default='argdata.dat')
    parser.add_argument('-ntf','--noti_target_file', help="Target file for Notification (default: 'work_win01.bat')",
                        type=str, default="work_win01.bat")
    parser.add_argument('-lp', '--LrnParam', help="Learning Parameter, which is read from the config_learning.yaml file (default=0)",
                        type=int, default=0)
    parser.add_argument('-an', '--analysis_level', help="Analysis Level [(-1)] No analysis [0] One Epoch Analysis",
                        type=int, default=-1)
    parser.add_argument('-sw', '--summary_writer', help="Summary Writer [0] No Summary Writer [1] Summary-Writer of Tensorboard is active",
                        type=int, default=1)

    ## For VGG
    parser.add_argument('-vgg_ar', '--vgg_arch', help="Architecture of VGG : 'VGG11', 'VGG13', 'VGG16', 'VGG19' (default='VGG19')",
                        type=str, default='VGG19')
    ## For EfficientNet
    parser.add_argument('-eb', '--efficientnet_id',
                        help="[0] Index of Efficient Net. This framework provide from b0 to b7: Default is b0",
                        type=int, default=0)
    ## Quite Model Loading
    parser.add_argument('-qm', '--quite_model_loading',
                        help="[0] print model [1:Default] quite model loading",
                        type=int, default=1)
    ## Cleaning GPU : for debug
    parser.add_argument('-clgpu', '--cleaning_gpu',
                        help="[0:Default] no cleaning [1] cleaning the GPU and exit()",
                        type=int, default=0)

    # Use Parameters (True) or default Argument (False)
    if bUseParam:
        args = parser.parse_args(L_Param)
    else:
        args = parser.parse_args()

    args.batch_size = 128 if args.data_set == 'CIFAR10' else args.batch_size
    args.autoproc   = True if args.autoproc == 1 else False
    args.batchproc  = True if args.batchproc == 1 else False
    args.cleaning_gpu=True if args.cleaning_gpu == 1 else False
    #args.net_name   = Get_Network_Name(args=args)
    args.proc_index_name = args.net_name + '_' \
                         + args.model_name + '_' \
                         + args.scheduler_name + '_' \
                         + 'lr' + str(args.learning_rate) + '_' \
                         + 'ep' + str(args.training_epochs)
    args.evaluation = True if args.evaluation == 1 else False
    # limit EfficientNet ID
    args.efficientnet_id = 0 if args.efficientnet_id < 0 else ( 7 if args.efficientnet_id > 7 else args.efficientnet_id)

    _current_path   = os.getcwd()
    _result_path    = os.path.join(_current_path, args.result_directory)
    args.result_directory = _result_path

    # Read Config File
    cy              = config_yaml()
    args.QParam     = cy.read_yaml_file(_yaml_file_name=cy.get_config_file_name(_category='quantization'))
    args.LrnParam   = cy.read_yaml_file(_yaml_file_name=cy.get_config_file_name(_category='learning'))

    # Set SummaryWriter for Tensorboard
    args.summary_writer = SummaryWriter() if args.summary_writer ==1 else args.summary_writer

    if os.path.exists(args.result_directory):
        print("There exists the path for result files %s " %(args.result_directory))
    else:
        os.mkdir(args.result_directory)

    Check_modelName(args)
    _sprint(_intro_msg)
    return args

# --------------------------------------------------------
# Processing Function
# --------------------------------------------------------
from torch_learning import learning_module

class op_class:
    def __init__(self, L_Param, b_multiproc=False, bUseParam=False):
        # Set Test Processing
        self._args      = _ArgumentParse(_description, L_Param, bUseParam) if b_multiproc else L_Param
        # self._args    = _ArgumentParse(_description, L_Param, bUseParam)
        self._device    = 'cuda' if self._args.device == 1 and torch.cuda.is_available() else 'cpu'
        self.GPU_DEVICE = torch.device(self._device)

        # Debug code : 2024-04-16
        self.b_parallel = self.Set_GPU_Processing()

        # print(torch.cuda.memory_summary(self.GPU_DEVICE))
        # Set Data Processing : Some data parameters depending on LoadingData ...
        self.Dset       = self.Set_Data_Processing(self._args, self._device)
        self.LoadingData= self.Dset.data_loader(bTrain=True, bsuffle=False)
        self.model      = self.Set_Model_Processing(self._args,self._device, self.Dset)

        # Set Operation
        self._args.model_file_name   = 'torch_nn02_' + self._args.proc_index_name + '.pt'
        self._error_trend_file       = 'error_' + self._args.proc_index_name + '.pickle'

        # Set Learning Parameter
        self._total_batch    = len(self.LoadingData)
        self.criterion       = torch.nn.CrossEntropyLoss()
        self.optimizer       = learning_module(model=self.model, args=self._args, total_batch=self._total_batch)
        self.c_opt           = self.optimizer.optimizer
        self._Learning_time  = 0
        self._Evaluation_time= 0
        self.op_result       = []
        self.b_Qalgorithm    = self._args.model_name == 'QSGD' or \
                               self._args.model_name == 'QtAdamW' or \
                               self._args.model_name == 'QtAdam'
        # Data recording
        self._data_recorder  = process_data_storage()
        # Final Processing in Initialization
        self.print_and_record_learning_parameters()
        # Evaluation
        self.load_model()
        # Analysis Level
        self.b_analysis     = True if self._args.analysis_level > -1 else False
        self.writer         = self._args.summary_writer

        # For debug : 2024-04-16
        #DBG.dbg("Program Terminated!!  This Debugging process")
        #exit()
    # --------------------------------------------------------
    # Service functions for __init__
    # --------------------------------------------------------
    def Set_Model_Processing(self, args, device, Dset):
        if args.data_set == 'MNIST' or args.data_set == 'FashionMNIST':
            base_model = CNN(inputCH=Dset.inputCH, outCH=Dset.outputCH)
        else:
            if args.net_name == 'ResNet':
                base_model = ResNet(inputCH=Dset.inputCH, outCH=Dset.outputCH, num_layers=args.num_resnet_layers)
            elif args.net_name =='VGG' or args.net_name == 'VGG11' or args.net_name == 'VGG13' or args.net_name == 'VGG16' or args.net_name == 'VGG19':
                base_model = VGG(architecture=VGG.get_VGG_architecture(args.vgg_arch),
                            in_channels=Dset.inputCH, num_classes=Dset.outputCH,
                            in_height=Dset.height, in_width=Dset.width)
            elif args.net_name == 'EfficientNet':
                base_model = EfficientNet_model(architecture=args.efficientnet_id,
                                                in_channels=Dset.inputCH,
                                                in_height=Dset.height,
                                                in_width=Dset.width,
                                                num_classes=Dset.outputCH).get_efficientnet_model()
            elif args.net_name == 'LeNet':
                base_model = LeNet(inputCH=Dset.inputCH, outCH=Dset.outputCH)
            else:
                print(" Unknown Network Model : %s \n Please Check the Network Name \n Program is Terminated" %args.net_name)
                exit()

        if self._device == 'cuda' and self.b_parallel:
            _model = base_model.cuda()
            model  = nn.DataParallel(_model).to(device)
            print("Parallel Processing with Multiple GPU")
        else:
            model  = base_model.to(device)
            print("Serial Processing with Single GPU %s %s" %(self._args.device, self.b_parallel))

        return model

    def Set_Data_Processing(self, args, device):
        if args.data_set == 'MNIST':
            Dset = MNIST_set(batch_size=args.batch_size, bdownload=True)
        elif args.data_set == 'FashionMNIST':
            Dset = FashionMNIST_set(batch_size=args.batch_size, bdownload=True)
        elif args.data_set == 'CIFAR10':
            Dset = CIFAR10_set(batch_size=args.batch_size, bdownload=True)
        elif args.data_set == 'CIFAR100':
            Dset = CIFAR100_set(batch_size=args.batch_size, bdownload=True)
        elif args.data_set == 'ImageNet':
            Dset = ImageNet_set(batch_size=args.batch_size, bdownload=False)
        elif args.data_set == 'STL10':
            if args.net_name == 'ResNet':
                Dset = STL10_set(batch_size=args.batch_size, input_size=32, bdownload=True)
            else:
                Dset = STL10_set(batch_size=args.batch_size, bdownload=True)
        else:
            Dset = None
            DBG.dbg("Data set is not assigned !! It is Error!!!")
            exit()

        return Dset

    def Set_GPU_Processing(self, b_parallel_GPU = False):
        # Set GPU Parallel Processing
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        _device_count  = torch.cuda.device_count()
        if _device_count > 1 and len(self._args.gpu_index) > 1:
            b_parallel_GPU = True
            _msg = ' '.join(list(map(str, self._args.gpu_index)))
            os.environ["CUDA_VISIBLE_DEVICES"] = _msg
        else :
            b_parallel_GPU = False

        if len(self._args.gpu_index) == 1 and self._args.gpu_index[0] > 0:
            _msg = self._device
            self._device = _msg + ":" + str(self._args.gpu_index[0])
        else: pass

        if self._device=='cpu' or _device_count == 0: pass
        else:
            # Clean the cache memory in GPU
            gc.collect()
            #torch.cuda.empty_cache()
            with torch.cuda.device(self._device):
                torch.cuda.empty_cache()

        print('Count of using GPUs:', _device_count)
        if self._device=='cpu' or _device_count == 0: pass
        else:
            print('Current cuda device:', torch.cuda.current_device())
        print('Process cuda device:', self._device)
        print('Parallel Processing: %s' %('True' if b_parallel_GPU else 'False'))

        if self._args.cleaning_gpu :
            print("Only Activation for Cleaning GPU. Program Terminated !!")
            exit()
        else:
            pass
        return b_parallel_GPU
    # --------------------------------------------------------
    # Service functions for Print and Record
    # --------------------------------------------------------
    def print_and_record_learning_parameters(self, b_print=True):
        _sprint("Information of Operation :")
        _sprint("   Data Set              : %s" %(self._args.data_set), b_print=b_print)
        _sprint('   Total number of Batch : {}'.format(self._total_batch), b_print=b_print)
        _sprint("   Batch SIze            : %d" %(self.Dset.batchsize), b_print=b_print)
        _sprint("   Dimension of Data     : {}".format(self.Dset.datashape), b_print=b_print)
        _sprint("   Hardware Platform     : %s" %(self._device), b_print=b_print)
        _sprint("   Model File Name       : %s" %(self._args.model_file_name), b_print=b_print)
        _sprint("   Error Trend File Name : %s" %(self._error_trend_file), b_print=b_print)
        _sprint("   Learning algorithm    : %s" %self._args.model_name, b_print=b_print)
        _sprint("   Learning rate         : {}".format(self._args.learning_rate), b_print=b_print)
        _sprint("   Learning Schedule     : %s" % self._args.scheduler_name, b_print=b_print)

        if self.b_Qalgorithm:
            _quantization_param = self._args.QParam['Quantization']
            _sprint("Quantization Parameters  : ", b_print=b_print)
            _sprint("   Initial QP            : {}".format(self.c_opt.Q_proc.c_qtz.get_QP()), b_print=b_print)
            _sprint("   base                  : {}".format(_quantization_param['base']), b_print=b_print)
            _sprint("   eta                   : {}".format(_quantization_param['eta']), b_print=b_print)
            _sprint("   kappa                 : {}".format(_quantization_param['kappa']), b_print=b_print)
            _sprint("   warmp_up_period       : {}".format(_quantization_param['warmp_up_period']), b_print=b_print)

        if self._args.net_name == 'ResNet':
            if self.b_parallel :
                _sprint("   ResNet model works on Multiple GPUs")
            else:
                _sprint("   ResNet num. of Layers : %d" % (self.model.total_layers), b_print=b_print)

        _parse_gpuinfo = self._device.split(':')
        if _parse_gpuinfo[0] == 'cuda' and self._args.analysis_level > 1:
            _sprint("Information of GPU Status :")
            _sprint(torch.cuda.memory_summary(self.GPU_DEVICE))

        _sprint("-----------------------------------------------------------------", b_print=b_print)
        _sprint("\n")

    # For Check Error in self.criterion
    def evaluate_cost(self, _prediction, Y):
        try:
            return self.criterion(_prediction, Y)
        except Exception as ex:
            DBG.dbg("Error of Evaluation in  Learning Process ")
            DBG.dbg("Error :", ex )
            exit()
    # --------------------------------------------------------
    # Learning : X input, Y Label or Target using LoadingData based on Torch's data_loader
    # --------------------------------------------------------
    def _learning(self):
        _start_time = time.time()
        for epoch in range(self._args.training_epochs):
            _avg_cost, _k = 0, 0
            for X, Y in tqdm(self.LoadingData, desc="Process : ", leave=False, mininterval=1.0):
                # Data loading on CPU or GPU
                X, Y = X.to(self._device), Y.to(self._device)

                self.optimizer.zero_grad()
                _prediction = self.model.forward(X)
                _cost = self.evaluate_cost(_prediction, Y)
                _cost.backward()

                # Debug --------------------------------------------------------------
                self.optimizer.Set_cost_info(_cost=_cost, _avg_cost=_avg_cost, b_active=self.b_Qalgorithm)
                # Debug --------------------------------------------------------------

                self.optimizer.learning(epoch)

                # Update Index to batch and record the error to tensorboard
                _avg_cost += _cost / self._total_batch

            # Update Learning rate to each epoch
            self.optimizer.scheduler.step()

            # Record the Average Cost
            self.writing_learning_result_per_epoch(epoch, _avg_cost)

        self._Learning_time = time.time() - _start_time
        # Data in Buffer to Tensorboard
        self.writer.flush()

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------
    def _test_processing(self):
        print("This process is evaluation. First, for the training data, Second, for the test data")
        l_train_result, l_test_result = [], []
        with torch.no_grad():
            _total, _correct, _accuracy = self.Dset.Test_Function(self.model, self._device, ClassChk=False, bTrain=True)
            _sprint("-----------------------------------------------------------------")
            _sprint("Train Data")
            _sprint("Total samples : %d   Right Score : %d " % (_total, _correct))
            _sprint("Accuracy      : %f" % _accuracy)
            l_train_result.append(_total)
            l_train_result.append(_correct)
            l_train_result.append(_accuracy)

            _start_time = time.time()
            _total, _correct, _accuracy = self.Dset.Test_Function(self.model, self._device, ClassChk=False, bTrain=False)
            self._Evaluation_time = time.time() - _start_time
            _sprint("-----------------------------------------------------------------")
            _sprint("Test Data")
            _sprint("Total samples : %d   Right Score : %d " % (_total, _correct))
            _sprint("Accuracy      : %f" % _accuracy)
            l_test_result.append(_total)
            l_test_result.append(_correct)
            l_test_result.append(_accuracy)

            _sprint("-----------------------------------------------------------------")
            _sprint("Total Learning Time   : %.3f sec" % (self._Learning_time))
            if self._args.training_epochs > 0:
                _sprint("Average Learning Time : %.3f sec" % (self._Learning_time / self._args.training_epochs))
            else:
                _sprint("Average Learning Time : No Learning, Epoch is Zero")
            _sprint("Evaluation Time       : %.3f sec" % (self._Evaluation_time))
            self.op_result.append(l_train_result)
            self.op_result.append(l_test_result)
            self.op_result.append(self._Evaluation_time if self._args.evaluation else self._Learning_time)

    # --------------------------------------------------------
    # Final Stage
    # --------------------------------------------------------
    def writing_learning_result_per_epoch(self, epoch, _avg_cost):
        _learning_rate = self.optimizer.get_optimizer_parameter(_param='lr')
        self._data_recorder.write_data_on_board(_avg_cost, _learning_rate)

        _msg_data   = ("[Epoch : %4d] cost = %f   " %(epoch, _avg_cost))
        if self.b_Qalgorithm:
            _msg_data += ("QP Index: %2.3f  sup_QP: %2.3f " % (self.c_opt.Q_proc.Get_QPIndex(), self.c_opt.Q_proc.Get_supQP()))
        else:
            _msg_data += ("LR = %f" % (_learning_rate))

        _sprint(_msg_data)
        # record the data to the buffer
        self.writer.add_scalar("Loss/train", _avg_cost, epoch)
        #writer.add_scalar("Learning Rate", _learning_rate, epoch)

    def _final(self):
        self._current_path = os.getcwd()
        os.chdir(self._args.result_directory)

        # Save Model(pt file) and learning data recording (pickle)
        torch.save(self.model.state_dict(), self._args.model_file_name)
        self._data_recorder.save_process_data(_file_name=self._error_trend_file)
        _opfilename = 'operation_' + self._args.proc_index_name + '.txt'
        _ymfilename = 'operation_' + self._args.proc_index_name + '.yaml'
        _write_operation(_opfilename)

        # Make YAML File
        now = time
        [l_train_result, l_test_result, Learning_time] = self.op_result
        _sprint("data_name: ", b_print=False)
        _sprint("   - Average Cost(Epoch)", b_print=False)
        _sprint("   - learning rate", b_print=False)
        _sprint("Train Data Result:", b_print=False)
        _sprint("   Total samples : %d" %l_train_result[0], b_print=False)
        _sprint("   Right Score   : %d" %l_train_result[1], b_print=False)
        _sprint("   Accuracy      : %f" %l_train_result[2], b_print=False)
        _sprint("Test Data Result :", b_print=False)
        _sprint("   Total samples : %d" %l_test_result[0], b_print=False)
        _sprint("   Right Score   : %d" %l_test_result[1], b_print=False)
        _sprint("   Accuracy      : %f" %l_test_result[2], b_print=False)
        _sprint("Total Learning Time: %.3f" %Learning_time, b_print=False)
        _sprint("Operation Time: {}".format(now.strftime('%Y-%m-%d %H:%M:%S')))

        self.print_and_record_learning_parameters(b_print=False)
        _write_operation(_ymfilename)

        os.chdir(self._current_path)
        return self._args

    def load_model(self):
        if self._args.evaluation :
            try:
                self.model.load_state_dict(self._args.model_file_name)
            except:
                DBG.dbg("There is not any proper model file in this directory. \n Please, copy an appropriate model file here.")
                print("Process abnormally finish")
                exit()
        else :
            print("This Processing is learning. Not evaluation. Learning is just started!!")
            print("-----------------------------------------------------------------")
        return

    def evaluation_final(self, b_print=True):
        #self._current_path = os.getcwd()
        #os.chdir(self._args.result_directory)
        g_msg.clear()

        # Save Model(pt file) and learning data recording (pickle)
        _ymfilename = 'evaluation_' + self._args.proc_index_name + '.yaml'
        # Make YAML File
        [l_train_result, l_test_result, Evaluation_time] = self.op_result
        _sprint("data_name: ", b_print=b_print)
        _sprint("   - Average Cost(Epoch)", b_print=b_print)
        _sprint("   - learning rate", b_print=b_print)
        _sprint("Train Data Result:", b_print=b_print)
        _sprint("   Total samples : %d" %l_train_result[0], b_print=b_print)
        _sprint("   Right Score   : %d" %l_train_result[1], b_print=b_print)
        _sprint("   Accuracy      : %f" %l_train_result[2], b_print=b_print)
        _sprint("Test Data Result :", b_print=b_print)
        _sprint("   Total samples : %d" %l_test_result[0], b_print=b_print)
        _sprint("   Right Score   : %d" %l_test_result[1], b_print=b_print)
        _sprint("   Accuracy      : %f" %l_test_result[2], b_print=b_print)
        _sprint("Total Evaluation Time: %.3f" %Evaluation_time, b_print=b_print)

        #self.print_and_record_learning_parameters(b_print=b_print)
        _write_operation(_ymfilename)

        #os.chdir(self._current_path)
        return self._args
# =============================================================
# Top Level Service Function
# =============================================================
def generate_notify(_target_file="work_win01.bat"):
    _head_msg   = "Test Process is finished. Please Check \n"
    c_noti      = send_notify()
    _noti_msg   = _head_msg + c_noti.extract_batfile(_target_file)
    c_noti.send_noti_mail(_msg=_noti_msg)

def clean_result_directory(result_dirName ="result" ):
    curr_dir = os.getcwd()
    result_directory = os.path.join(curr_dir, result_dirName)
    os.chdir(result_directory)
    _file_list = [_file for _file in os.listdir(result_directory)]
    for _k, _file in enumerate(_file_list):
        if os.path.isfile(_file):
            os.remove(_file)
        else: pass
    os.chdir(curr_dir)
    print("Clean the result directory : ", result_directory )

def multiple_training(s_arg_data="argdata.dat", b_multiproc=True, b_UseParam=True):
    clean_result_directory()
    with open(s_arg_data, 'rt') as f:
        for _k, _line in enumerate(f.readlines()):
            _idx = _line.find("::")
            if _idx < 0 :
                _operation_param = _line.split()
                if len(_operation_param) == 0:
                    print("There is not any data at Line : {0:3}".format(_k))
                    print("multiple_training Forced finished !!!")
                    params = -1
                    #DBG.dbg("Check the file %s for this error" %(s_arg_data), _active=True)
                    #exit()
                else:
                    params  = training(_operation_param, b_multiproc=b_multiproc, bUseParam=b_UseParam)
            else: pass

    return params

# =============================================================
# Test Processing
# =============================================================
def training(L_Param, b_multiproc=False, bUseParam=False):
    c_op = op_class(L_Param=L_Param, b_multiproc=b_multiproc, bUseParam=bUseParam)
    c_op._learning()
    c_op._test_processing()
    params = c_op._final()
    return params

def evaluation(L_Param, b_multiproc=False, bUseParam=False):
    c_op = op_class(L_Param=L_Param, b_multiproc=b_multiproc, bUseParam=bUseParam)
    c_op._test_processing()
    params = c_op.evaluation_final()
    return params

if __name__ == "__main__":
    _args               = _ArgumentParse(_intro_msg=_description, L_Param=[])
    b_FundamentalUse    = not _args.batchproc # True ... Single Learning
    _arg_data_file      = _args.arg_data_file
    _noti_target_file   = _args.noti_target_file if b_FundamentalUse else _arg_data_file

    if b_FundamentalUse:
        params = training(_args, bUseParam=True)
    else:
        params = multiple_training(s_arg_data=_arg_data_file)

    print("-----------------------------------------------------------------")
    try:
        generate_notify(_target_file=_noti_target_file)
    except Exception as e:
        DBG.dbg("Noti Error : ", e)
    _args.summary_writer.close()
    print("=============================================================")
    print("Process Finished!!")
    print("=============================================================")