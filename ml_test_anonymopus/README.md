READ ME
===

[toc] 

## Package Specificaion
### Fundamental Package 

| File Name         | Specification |
|-------------------|---------------|
| nips_quant.py     | Quantization Processing |
| Read_Data.py      | Data Read (MNIST, CIFAR10, CIFAR100, ImageNet:146GB) |
| torch_learning.py | Optimizer and Scheduler |
| torch_resnet.py   | ResNet-50               |
| torch_SmallNet.py | LeNet                   |
| torch_nn02.py     | Main Operation          |

### Service Package 
- Common Service Package

| File Name         | Specification |
|-------------------|---------------|
| my_debug.py       | For Debugging |

- Specific Service Packgae

| File Name         | Specification |
|-------------------|---------------|
| service_*****.py  | ***** Service |

- Argument File for multiple batch learning process

| File Name         | Specification |
|-------------------|---------------|
| argdata.dat  | multiple processing |

- YAML for setting service_process.py
No specific modifications are needed for config_service.yaml and config_learning.yaml files. Any hyperparameter-related changes should be made in the config_quantization.yaml file. For more details, please refer to the Supplementary Material.

| File Name         | Specification |
|-------------------|---------------|
| config_service.yaml  | setting service_process.py |
| config_learning.yaml | torch.learning.py / torch_nn02.py|
| config_quantization.yaml  | nips_quant.py.py |

## Installation

We strongly recommend Miniconda environment to operate the provided test codes.
When you setup Miniconda packages (or previously  installed), we recommend the base virtual environment of your Miniconda platform as follows:

~~~
(base)yourID@home$
~~~

### For Linux 
Change the mode of the file "installation.sh" as follows:
~~~
chmod +x installation.sh
~~~
Run the installation.sh such that
~~~
source ./installation.sh
~~~
Following this, you write the name of the conda virtual environment after the following command :
~~~
Input the name of a conda virtual environment for test codes:
~~~
If you input the name as "test_env" such that 
~~~
test_env
~~~
, then you can verify the name as follows:
~~~
The name of rhe conda virtual environment: test_env
~~~
Automatically, the installation scripts setup the required python packages

### For Windows 
Instead of the installation shell files in Linux, we use the **installation.bat** for a windows system. The usage of the bat file is almost equal to that of a Linux system.  We clarify each command in the batch file

First, we update the Miniconda framework using the following commands:

~~~
conda update -n base conda
conda update --all
python -m pip install --upgrade pip
~~~

Following this, we input the follwoing code to install the test codes. You set the name of an Miniconda environment replace %conda_env_name% with what you want. 

~~~
conda env create --name %conda_env_name% python=3.11.4 
~~~

If you set the name of environment as "test_env", the installation code is as follows (It takes some times):

~~~
conda env create --name test_env python=3.11.4 
~~~
yAfter installing the required packages, activate the conda environment named with what you depicts.

~~~
conda activate %conda_env_name%
~~~

Finally, type the following commands to install pytorch and other required libraries 

~~~
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install tensorboard matplotlib pyyaml scipy
pip3 install wget
~~~

### Verification of installation 

You will see that the conda environment is changed with the "test_env". 
~~~
(test_env)yourID@home$
~~~

You can verify the conda environment is setup appropriately with the following python test code:
~~~
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
>>>2.2.2+cu118 True     # It indicates that the cuda chain for pytorch is setup proper.
~~~

To verify the test program works well, type the following command:
~~~
python torch_nn02.py -m Adam -d CIFAR10 -e 10 -n ResNet -g 1
~~~
After the downloading of CIFAR10 dataset, you can see the operation of the test codes.

#### Trouble Shooting 
- torch.cuda
If the result of torch.cuda.is_available() is false, the test code operates on the cpu only mode. It takes a lot of time to operates the test codes.  It means that your pakages in the environment are not compatible with the pytorch-cuda package, 

When you meet this case,  attempt to run the following command, and verify appropriate installation using the above command.

~~~
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
~~~
or
~~~
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
~~~

- torchvision
When importing of torchvision fails in your python environment, first you should clean the cahche for pip as follows:
~~~
pip cache info
pip cache purge
pip cache info
~~~
You can verify the result of the cleaning pip cache with **pip cache info**.
Next, torchvision requires the Pillow version 10.3.0, so that we should upgrade Pillow such that

~~~
pip install --upgrade Pillow
~~~
It enables the importing torchvision. 

- torchvision and torchvision::nms error

~~~
RuntimeError: operator torchvision::nms does not exist
~~~

Solution is here.  [RuntimeError: operator torchvision::nms does not exist - vision - PyTorch Forums](https://discuss.pytorch.org/t/runtimeerror-operator-torchvision-nms-does-not-exist/192829)

The correuption between torch and torchvision brings this bug, so that we reinstall torchvision soley with pip3 as follows:

~~~
pip3 install torchvision --index-url https://download.pytorch.org/whl/cu118
~~~

For verification, you can type pip show torchvision as follows:

~~~
(py3.11.4) >pip show torchvision
Name: torchvision
Version: 0.17.2+cu118
Summary: image and video datasets and models for torch deep learning
Home-page: https://github.com/pytorch/vision
Author: PyTorch Core Team
Author-email: soumith@pytorch.org
License: BSD
Location: C:\ProgramData\miniconda3\envs\py3.11.4\Lib\site-packages
Requires: numpy, pillow, torch
Required-by:
~~~




- pip upgrade 
Sometimes, upgrading of pip3 fails. In this case, we should remove the setup tools for python and reinstall that as follows: 
~~~
python -m pip uninstall pip setuptools
python -m ensurepip
python -m pip install --upgrade pip
~~~

- Version conflicts of pytorch

  Recently, the pytorch is upgraded to the version 2.3, so sometimes installing of pytorch will be fail.  In this case, you just reinstall the pytorch such that 
~~~
(For CPU) pip3 install torch torchvision torchaudio 
(For GPU) pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
~~~  



## Usage of Test code : Torch_nn02.py

### Set python Environment
In Ubuntu, you can use the following steps. 
First, check the Python version, and make sure to select version 3.10. 
Note that the library packages may vary depending on the Python version. 
It is recommended to use a virtual environment to avoid conflicts between packages.

~~~
sudo update-alternatives --config python3
~~~

### Basic IO

#### Input Files 
parameter -d 

| Data Set          | Parameter|  Notes |
|-------------------|----------|--------|
| MNIST Data Set    | MNIST    |        |
| CIFAR10 Data Set  | CIFAR10  |        |
| CIFAR100 Data Set | CIFAR100 |        |
| ImageNet          | ImageNet |        |
| STL10 Data Set    | STL10    |        |

#### Network Models

Parameter : -n

| Network MOdel     | Parameter|  Notes |
|-------------------|----------|--------|
| Simple 2 Layers   | CNN      |        |
| LeNet             | LeNet    |        |
| ResNet            | ResNet   |        |
| VGG19             | VGG      |'VGG11', 'VGG13', 'VGG16', 'VGG19' (default='VGG19')|
| EfficientNet      | EfficientNet    | from b0 to b7 (Default b7)|


#### output Files 

| Spec | Format | Example|
|---|---|---|
| Neural Network File | torchnn02+ Model name + Algorithm name  + Epoch.pt | torch_nn02ResNetAdam.pt |
| Operation File  | operation + Model name + Algorithm name + Epoch.txt | operation_ResNetAdam100.txt |
| Error Trend File| error_ + Model name + Algorithm name + Epoch.pickle | error_ResNetAdam100.pickle |

### For Cifar-10 Data Set 

#### LeNet
~~~
python torch_nn02.py -m Adam -d CIFAR10 -e 100 -n LeNet -g 1
~~~

#### ResNet 
~~~
python torch_nn02.py -m Adam -d CIFAR10 -e 100 -n ResNet -g 1 
~~~

#### Note
- When you don't use the CUDA but CPU mode, you don't set 'g' option or '-g 0'.
- For ResNet, you should set '-g 1' option, because the size of the network is so large. 

### Parallel Processing For Mutiple GPUs
When your computer equipped with mutiple GPUs, we can use it with Pytorch's **torch.nn module. **

1. See Torchnn_02.py ,Set_Model_Processing,  Line 269~273 
~~~python
base_model = ResNet() or VGG() ... 

if self._args.device == 'cuda' and len(self._args.gpu_index) > 1:
	_model = base_model.cuda()
	model  = nn.DataParallel(_model).to(device)
else:
	model  = base_model.to(device)
~~~

2. Parameter 
~~~python 
parser.add_argument('-gn','--gpu_index', nargs='+', help="GPU index, Default value is =[0] (List), ex) -gn 0 1 ..",                   type=int, default=[0])
~~~
Example
~~~
-g 1 -gn 0, 1
~~~

## Quantization Algorithm

- QSGLD
~~~
python torch_nn02.py -m QSGD -d CIFAR10 -e 100 -n ResNet -g 1
~~~

- QtAdam
~~~
python torch_nn02.py -m QtAdam -d CIFAR10 -e 100 -n ResNet -g 1
~~~

## Torch_testNN.py Usage
- The Data Set,  Network Spec (*.pt file), Network name should be specified 
- The test program plot an error trend when an error file exists

### Sinlge Processing 
~~~
python torch_testNN.py -d CIFAR10 -n ResNet -ng 1 -e error_ResNetAdam15.pickle -m torch_nn02ResNetAdam.pt 
~~~

### Batch Processing
- The test program takes the processing argument from the "argdata.dat" on the working directory
~~~
python torch_nn02.py -bt 1
~~~


### Scheduler Option '-sn'
- $t$ denotes the number of the epoch.

| Scheduler Name | Parameter  | Specification |
|----------------|------------|---------------|
| ConstantLR     | Default    |               |
| LambdaLR       | LambdaLR   | $0.95^{t}$    |
| ExponentialLR  | exp        |               |
| CyclicLR       | cyclicLR   |
| CosineAnnealingWarmRestarts | CAWR or CosineAnnealingWarmRestarts | $\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}}) \left( 1 + \cos \left( \frac{T_{\text{cur}}}{T_{\text{max}}} \pi \right) \right)$   |
| CustomCosineAnnealingWarmUpRestarts | CCAWR or CustomCosineAnnealingWarmRestarts |   |


## Appendix

#### Torch_nn02.py Help Message

~~~
(py3.10.0) sderoen@sderoen-System-Product-Name:~/Works/python_work_2023/nips2023_work$ python torch_nn02.py -h
usage: torch_nn02.py [-h] [-g DEVICE] [-l LEARNING_RATE] [-e TRAINING_EPOCHS] [-b BATCH_SIZE] [-f MODEL_FILE_NAME] [-m MODEL_NAME] [-n NET_NAME]
                     [-d DATA_SET] [-a AUTOPROC] [-pi PROC_INDEX_NAME] [-rl NUM_RESNET_LAYERS] [-qp QPARAM] [-rd RESULT_DIRECTORY]
                     [-sn SCHEDULER_NAME] [-ev EVALUATION] [-bt BATCHPROC] [-ag ARG_DATA_FILE] [-ntf NOTI_TARGET_FILE] [-lp LRNPARAM]

options:
  -h, --help            show this help message and exit
  -g DEVICE, --device DEVICE
                        Using [(0)]CPU or [1]GPU
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning_rate (default=0.001)
  -e TRAINING_EPOCHS, --training_epochs TRAINING_EPOCHS
                        training_epochs (default=15)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch_size (default=100)
  -f MODEL_FILE_NAME, --model_file_name MODEL_FILE_NAME
                        model file name (default='torch_nn02_CNN.pt')
  -m MODEL_NAME, --model_name MODEL_NAME
                        model name 'SGD', 'Adam', 'AdamW', 'ASGD', 'NAdam', 'RAdam' (default='Adam')
  -n NET_NAME, --net_name NET_NAME
                        Network name 'CNN', 'LeNet', 'ResNet' (default='LeNet')
  -d DATA_SET, --data_set DATA_SET
                        data set 'MNIST', 'CIFAR10' (default='MNIST')
  -a AUTOPROC, --autoproc AUTOPROC
                        Automatic Process without any plotting [(0)] plotting [1] Automatic-no plotting
  -pi PROC_INDEX_NAME, --proc_index_name PROC_INDEX_NAME
                        Process Index Name. It is generated automatically (default='')
  -rl NUM_RESNET_LAYERS, --num_resnet_layers NUM_RESNET_LAYERS
                        The number of layers in a block to ResNet (default=5)
  -qp QPARAM, --QParam QPARAM
                        Quantization Parameter, which is read from the config_quantization.yaml file (default=0)
  -rd RESULT_DIRECTORY, --result_directory RESULT_DIRECTORY
                        Directory for Result (Default: ./result)
  -sn SCHEDULER_NAME, --scheduler_name SCHEDULER_NAME
                        Learning rate scheduler (Default : Constant Learning)
  -ev EVALUATION, --evaluation EVALUATION
                        Only Inference or Evaluation with a learned model [(0)] Training and Inference [1] Inference Only
  -bt BATCHPROC, --batchproc BATCHPROC
                        Batch Processing with 'argdata.dat' file or Not [(0)] Single processing [1] Multiple Processing
  -ag ARG_DATA_FILE, --arg_data_file ARG_DATA_FILE
                        Argument data File for Batch Processing (default: 'argdata.dat')
  -ntf NOTI_TARGET_FILE, --noti_target_file NOTI_TARGET_FILE
                        Target file for Notification (default: 'work_win01.bat')
  -lp LRNPARAM, --LrnParam LRNPARAM
                        Learning Parameter, which is read from the config_learning.yaml file (default=0)
~~~

### Examples

- When the learning and the testing data in a specific directory, we set the arguments for the test program such that  

~~~
(python01)>python torch_testNN.py -d CIFAR10 -m ./Result_data/LeNet/torch_nn02LeNetAdam100.pt -n LeNet -e ./Result_data/LeNet/error_LeNetAdam100.pickle -p 100 -ng 0
~~~


## Service_process.py 

- Service process provides the function for gathering the learning and the testing of each algorithm.
- After learning, it plots and writes summary files from the result files generated by the test program.

### Representative Option 

| Option Name         | Contents                                                     |
|---------------------|--------------------------------------------------------------|
| -pr --processing    | [(0)] single file processing [1] multiple files processing   |
| -an --analysis      | [(0)] Algorithm Analysis     [1] Learning rate Analysis [2] Eta Analysis |


### help 
~~~
(py3.10.0) D:\Work_2023\nips2023_work>python service_process_board.py -h
usage: test pytorch_inference [-h] [-rd RESULT_DIR] [-t TRAINING] [-pr PROCESSING] [-gp GRAPHIC] [-ea EQUAL_ALGORITHM]
                              [-el EQUAL_LEARNING_RATE] [-cf CONFIG]

====================================================
service_process_board.py :
                    Written by Jinwuk @ 2023-04-25
====================================================
Example : service_process_board.py

options:
  -h, --help            show this help message and exit
  -rd RESULT_DIR, --result_dir RESULT_DIR
                        Depict result directory (Default: result)
  -t TRAINING, --training TRAINING
                        [0] test [(1)] training
  -pr PROCESSING, --processing PROCESSING
                        [(0)] single file processing [1] multiple files processing
  -gp GRAPHIC, --graphic GRAPHIC
                        [0] no plot graph [(1)] active plot graph
  -ea EQUAL_ALGORITHM, --equal_algorithm EQUAL_ALGORITHM
                        [(0)] no equal_algorithm [1] equal_algorithm
  -el EQUAL_LEARNING_RATE, --equal_learning_rate EQUAL_LEARNING_RATE
                        [0] no equal_learning_rate [1] equal_learning_rate
  -cf CONFIG, --config CONFIG
                        Config file for Service Process (default) config_service.yaml
~~~



## Appendix 

### Operation with SSH Under Conda Environment For Windows 

If you test the provided Python program with SSH in Windows, you should find where the execution file for conda on your computer. For instance, we suppose that the conda execution file is in the following directory:

~~~
C:\ProgramData\miniconda3
~~~

Additionally, the batch file for the conda environment is located as the following:

~~~
C:\ProgramData\miniconda3\Scripts\activate.bat
~~~

Then, you can activate the conda environment using the following command on an SSH console.

~~~
C:\ProgramData\miniconda3\Scripts\activate.bat C:\ProgramData\miniconda3

(base) admin@DESKTOP-PN7OHLT C:\Users\admin\Downloads\Jinwuk_Private_folder\Work\2024\nips2023_work>
~~~

Afterward, you should activate an appropriate conda environment. For example, if the name of the conda virtual environment is "py3.11.6", you should type the following commands:

~~~
conda activate py3.11.6
~~~

The above command enables the conda virtual environment for testing.

### Operation with SSH Under Conda Environment For Linux

Activation of the conda environment in Linux is straightforward compared to the case of Windows.

First, find your conda virtual environment.

~~~
conda info --envs
~~~

