Visualisation of Simulated Annealing algorithm to solve the Travelling Salesman Problem in Python
=======
[toc]

## Introduction 

Using [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) metaheuristic to solve the [travelling salesman problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem), and animating the results.

A simple implementation which provides decent results.

Requires [python3](https://docs.python.org/3/), [matplotlib](https://matplotlib.org/) and [numpy](http://www.numpy.org/) to work

--------

An example of the resulting route on a TSP with 70 nodes on 200x200 grid.

<p align="center"><img src="https://media.giphy.com/media/3ohjUONfy5IqbaX1kY/giphy.gif" width="400"></p>

-------

An example of learning progress expressed as the distance of the path over the time

<p align="center"><img src="https://i.imgur.com/lk6v1V3.png" width="400"></p>

## Operation

### Presettings 

- Environment : base / python01 /py3.11.4  and others

- folder
  ~~~
  ./python-tsp-simulated-annealing-master
  ~~~
    - or Newest (After 2024.09.14)
  ~~~
    ./tsp_test
  ~~~
  
- Directory Tree    
~~~
C:.
├─init		 % Initial Files 
├─result	 % Result Files
└─__pycache__
~~~


- Simulated Annealing 
~~~
python tsp_proc.py -l 1 -an 0 -al 0
~~~

- Qunatized Optimization 
~~~
python tsp_proc.py -l 1 -an 0 -al 1 
~~~

- Qunatum Annealing 
~~~
python tsp_proc.py -l 1 -an 0 -al 2 
~~~

- Parameter 
| Parameters | Meaning |
|------------|---------|
| -l  1      | Load saved the locations of cities (init_data.pkl) |
| -al 1      | Algorithms (1 : Quantized Optimization) |
| -an 0      | No Animation |

- Parabolic Potential Well
   - Simulated Annealing
   ~~~
     python tsp_proc.py -l 1 -an 0 -al 0 -tp 0
   ~~~
   - Quantized Optimization
   ~~~
     python tsp_proc.py -l 1 -an 0 -al 1 -tp 0
   ~~~
   - Quantum Annealing
   ~~~
     python tsp_proc.py -l 1 -an 0 -al 1 -tp 0
   ~~~



#### help

~~~
(py3.11.4) D:\Work_2024\tsp_test>python plot_function.py -h
usage: plot_function.py [-h] [-f FUNCTION] [-d DOMAIN [DOMAIN ...]] [-p USE_PARAM]

====================================================
plot_function.py : function plot 
====================================================
Example : python plot_function.py -f 1 -d -6 6

options:
  -h, --help            show this help message and exit
  -f FUNCTION, --function FUNCTION
                        Function [0] test_multimodal [1] parabolic_wash_board [2] Arbitraty Function
  -d DOMAIN [DOMAIN ...], --domain DOMAIN [DOMAIN ...]
                        Domain
  -p USE_PARAM, --use_param USE_PARAM
                        Use Parameter
~~~

### Plotting function as 3-Dimension based on 2-Dimension Input (cec_2022_testfunction.py)

## Help options

### tsp_proc.py (or tsp.py)
~~~
(py3.11.4) C:\Users\sdero\Downloads\Jinwuk_private_folder\Jinwuks_work\work_2024\tsp_test>python tsp_proc.py -h
usage: tsp.py [-h] [-t TEMP] [-a ALPHA] [-wd SIZE_WIDTH] [-ht SIZE_HEIGHT] [-p POPULATION_SIZE] [-i STOPPING_ITER]
              [-st STOPPING_TEMP] [-fn DATAFILENAME] [-l LOADDATAFROMFILE] [-al ALGORITHM] [-an ACTIVE_ANIMATION]
              [-dm DEBUG_MESSAGE] [-tp TSP_PWP] [-lf TST_FUNC_ID] [-bd BAND] [-cm COMPARISON] [-gp GENETIC_POPULATION]

====================================================
tsp.py : TSP and Simulated Annealing Test
====================================================
Example : python tsp_proc.py -l 1

options:
  -h, --help            show this help message and exit
  -t TEMP, --temp TEMP  Initial Temperature
  -a ALPHA, --alpha ALPHA
                        alpha
  -wd SIZE_WIDTH, --size_width SIZE_WIDTH
                        size_width
  -ht SIZE_HEIGHT, --size_height SIZE_HEIGHT
                        size_height
  -p POPULATION_SIZE, --population_size POPULATION_SIZE
                        population_size
  -i STOPPING_ITER, --stopping_iter STOPPING_ITER
                        Stopping Iteration
  -st STOPPING_TEMP, --stopping_temp STOPPING_TEMP
                        stopping_temp
  -fn DATAFILENAME, --datafilename DATAFILENAME
                        File Name for data
  -l LOADDATAFROMFILE, --loaddatafromfile LOADDATAFROMFILE
                        Load Initial Data from PKLfile [default] 0 (int) [1] Load
  -al ALGORITHM, --algorithm ALGORITHM
                        Algorithm [0:Default] Simulated_Annealing [1] Quantized Optimization [2] Quantum Optimization
                        [3] GA Optimization
  -an ACTIVE_ANIMATION, --active_animation ACTIVE_ANIMATION
                        Active Animation
  -dm DEBUG_MESSAGE, --debug_message DEBUG_MESSAGE
                        [0] No Debug Message [1:Default] Active Debug Messgae
  -tp TSP_PWP, --tsp_pwp TSP_PWP
                        [0] parabolic_washboard_potential [1:Default] Travelling Salesman Problem
  -lf TST_FUNC_ID, --tst_func_id TST_FUNC_ID
                        [0:Default] parabolic_washboard_potential [1] Xin-She Yang N4 [2] Salomon [3] Drop-Wave [4]
                        Schaffel N2
  -bd BAND, --band BAND
                        Band frequency for parabolic_washboard_potential [10.0: Default]
  -cm COMPARISON, --comparison COMPARISON
                        Comapre with the initial and final path [0:default] simultaneous, [1] sequential
  -gp GENETIC_POPULATION, --genetic_population GENETIC_POPULATION
                        Number of chromossome for the genetic algorithm
~~~

## References

Parabolic washboard potential energy
~~~
@article{Stella_2005,
  title = {Optimization by quantum annealing: Lessons from simple cases},
  author = {Stella, Lorenzo and Santoro, Giuseppe E. and Tosatti, Erio},
  journal = {Phys. Rev. B},
  volume = {72},
  issue = {1},
  pages = {014303},
  numpages = {15},
  year = {2005},
  month = {Jul},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevB.72.014303},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.72.014303}
}
~~~

