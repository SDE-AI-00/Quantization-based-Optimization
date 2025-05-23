###########################################################
# p_test.py
# Algorithm for removing multiple data that satisfy the conditions in the list.
###########################################################
import numpy as np

threshold_cost  = 0.01 
small_hLine = "-----------------------------------------------------"

_test_data  = []

_test_data.append([288,    0.00030940,    0.03963954])
_test_data.append([20,    1.55942005,    2.44354544])
_test_data.append([161,   1.44630696,    0.42504425])
_test_data.append([32,    0.00601198,    0.06520842])
_test_data.append([1705,    0.00008364,    0.02038701])
_test_data.append([1996,    3.01386979,    1.49830984])
_test_data.append([80,    0.00002550,    0.01132196])
_test_data.append([864,    0.00034487,    0.04124986])
_test_data.append([99,    0.00008076,    0.02018282])
_test_data.append([29,    0.00046716,    0.04878575])

# Representation of Original data
for _j, _data in enumerate(_test_data):
    print("[%2d]  %6d   %12.8f   %12.8f" %(_j, _data[0], _data[1], _data[2]))
print(small_hLine)

# Check data meets the condition 
_fail_data = []
for _j, _data in enumerate(_test_data):
    _costdiff_data = _data[1]
    if _costdiff_data > threshold_cost:
        _fail_data.append([_j, _data])

for _data in _fail_data:
    print("[%2d]  %6d   %12.8f   %12.8f" %(_data[0], _data[1][0], _data[1][1], _data[1][2]))
print(small_hLine)

# Remove data meets the condition 
# make remove set
_remove_set = []
for _data in _fail_data:
    _remove_set.append(_data[1]) 

# remove appropriate components
_result = [_k for _k in _test_data if _k not in _remove_set]

# Representation of corrected data
for _j, _data in enumerate(_result):
    print("[%2d]  %6d   %12.8f   %12.8f" %(_j, _data[0], _data[1], _data[2]))
print(small_hLine)






