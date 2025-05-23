Here is the README file translated into English while maintaining the original Markdown format and ensuring technical documentation accuracy.

---

# README  

[toc]  

This Git repository is used for my new Python project.  
https://github.com/SDE-AI-00/Quantization-based-Optimization

## Test Program  
## **nlp_test02.py**  

The main representative Python file is **nlp_test02.py**.
Currently, **nlp_test02.py** is a verification program but is not in use.  

## **nlp_main01.py**  

The **main procedure**, previously implemented as a module, has been converted into a **class object**.  
Thus, **nlp_test2.py** is now used as a backup, and the new **main program is nlp_main01.py**.  

- **As of August 2021**, testing should be conducted as follows:  
    ```
    python nlp_main01.py -a 1 -s 1 -i -1.232 -1.212 -t 4000
    ```

## **Editor**  

For general **Python file editing**, the recommended editor is **Atom**.  
For **Python development environments**, use **PyCharm or VSCode** if debugging is required. Otherwise, **Atom + Conda** is recommended.  

## **Command for Each Algorithm**  

### **Algorithm Table**  

Simple Gradient and Armijo apply only a step size rule, meaning `-a 0 -s 1` is identical to `-a 1 -s 1`.  

| Index | Algorithm |
|---|---|
| 0 | Gradient Descent |
| 1 | Conjugate Gradient Polak |
| 2 | Conjugate Gradient Fletcher |
| 3 | Quasi Newton |
| 4 | AdaGrad  |
| 5 | AdaDelta |
| 6 | RMSProp  |
| 7 | ADAM     |
| 8 | Momentum |
| 9 | Nesterov  |

### **Step Size Rule**  

| Index | Step Size Method |
|---|---|
| 0 | Constant Learning Rate |
| 1 | Armijo Rule |
| 2 | Line Search (Golden Search) |
| 3 | Time Decaying |
| 4 | Armijo One-Step (Idle and Go) |
| 5 | Armijo One-Step (Fast Idle and Go)* |
| 6 | Line Search (Idle and Go)* |

### **Basic Algorithm Rules**  

### **Gradient with Constant Value**  
```python
python nlp_main01.py -a 0 -s 0 -i -1.232 -1.212 -t 4000
```
### **Gradient with Armijo**  
```python
python nlp_main01.py -a 0 -s 1 -i -1.232 -1.212 -t 4000
```
### **Conjugate Polak-Ribiere**  
```python
python nlp_main01.py -a 1 -s 1 -i -1.232 -1.212 -t 4000
```
### **Conjugate Fletcher-Reeves**  
```python
python nlp_main01.py -a 2 -s 1 -i -1.232 -1.212 -t 4000
```
### **Quasi-Newton**  
```python
python nlp_main01.py -a 3 -s 1 -i -1.232 -1.212 -t 4000
```
### **Quantized Newton (q=0, Annealing Style)**  
```python
python nlp_main01.py -a 3 -s 1 -i -1.232 -1.212 -t 4000 -q 0 -qm 2
```
### **Quantized Conjugate (FR s=3, PR s=2)**  
```python
python nlp_main01.py -a 4 -s 3 -i -1.232 -1.212 -t 4000 -q 3 -qm 2
```

## **Git Commands**  

### **Commit to Master**  
```
git add *
git commit -m "Description of this confirmed version"
```

### **Push to Master**  
```
git.exe push --progress "origin" master
```

## **Test Program**  

### **Result Parser**  
- **result_parser.py**: Class Library  
- **test_parser01.py**: Batch file execution to analyze test results  

## **Python Program Help**  

```
usage: nlp_test02    [-h] [-a ALGORITHM] [-w WEIGHT] [-df DEBUGFREQ] [-s STEPSIZE] [-q QUANTIZE]
                     [-qm QUANTIZE_METHOD] [-i INITIAL_POINT [INITIAL_POINT ...]] [-t ITERATIONS]
                     [-f FUNCTIONID] [-p PRINTOUTGIF] [-l LEARNINGRATE] [-n NOSHOWFIGURE] [-g MOMENTGAMMA]
                     [-fo FASTPARAMETER] [-d DIMENSION] [-qt QUITE_MODE] [-msg MESSAGE] [-rf RANDOM_FIX]
                     [-em EMERGENCY]
------------------------------------------------
Nonlinear Optimization Testbench
------------------------------------------------
Example : nlp_main01.py -a 4 -w 0 -s 2
Meaning : Using Quasi-Newton BFGS with Step size evaluated by Line Search

optional arguments:
  -h, --help            show this help message and exit
  -a ALGORITHM, --algorithm ALGORITHM
                        Search Algorithm [0] Gradient Descent [1] Conjugate Gradient (Polak-Ribiere)...
```

