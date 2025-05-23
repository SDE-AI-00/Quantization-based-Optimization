echo off
echo ###################################################################
echo   This is the batch file to test Quantized NLP 
echo   Usage  qnlp.bat 
echo   Example   : qnlp 
echo ###################################################################

del TXT_Result.txt
echo on

python nlp_test02.py -a 0 -s 1 -i -1.212 -1.32 -t 4000 -qt 1 -f 0 -msg "Rosenblatt 2D"
python nlp_test02.py -a 1 -s 1 -i -1.212 -1.32 -t 4000 -qt 1 -f 0 -msg "Rosenblatt 2D"
python nlp_test02.py -a 2 -s 1 -i -1.212 -1.32 -t 4000 -qt 1 -f 0 -msg "Rosenblatt 2D"
python nlp_test02.py -a 3 -s 1 -i -1.212 -1.32 -t 4000 -qt 1 -f 0 -msg "Rosenblatt 2D"

python nlp_test02.py -a 0 -s 1 -i -1.212 -1.32 -t 4000 -qt 1 -q 3 -qm 2 -f 0 -msg "Rosenblatt 2D Qunatization"
python nlp_test02.py -a 1 -s 1 -i -1.212 -1.32 -t 4000 -qt 1 -q 3 -qm 2 -f 0 -msg "Rosenblatt 2D Qunatization"
python nlp_test02.py -a 2 -s 1 -i -1.212 -1.32 -t 4000 -qt 1 -q 3 -qm 2 -f 0 -msg "Rosenblatt 2D Qunatization"
python nlp_test02.py -a 3 -s 1 -i -1.212 -1.32 -t 4000 -qt 1 -q 3 -qm 2 -f 0 -msg "Rosenblatt 2D Qunatization"

python nlp_test02.py -a 0 -s 1 -d 5 -t 4000 -qt 1 -f 0 -msg "Rosenblatt 5D"
python nlp_test02.py -a 1 -s 1 -d 5 -t 4000 -qt 1 -f 0 -msg "Rosenblatt 5D"
python nlp_test02.py -a 2 -s 1 -d 5 -t 4000 -qt 1 -f 0 -msg "Rosenblatt 5D"
python nlp_test02.py -a 3 -s 1 -d 5 -t 4000 -qt 1 -f 0 -msg "Rosenblatt 5D"

python nlp_test02.py -a 0 -s 1 -d 5 -t 4000 -qt 1 -q 3 -qm 2 -f 0 -msg "Rosenblatt 5D Qunatization"
python nlp_test02.py -a 1 -s 1 -d 5 -t 4000 -qt 1 -q 3 -qm 2 -f 0 -msg "Rosenblatt 5D Qunatization"
python nlp_test02.py -a 2 -s 1 -d 5 -t 4000 -qt 1 -q 3 -qm 2 -f 0 -msg "Rosenblatt 5D Qunatization"
python nlp_test02.py -a 3 -s 1 -d 5 -t 4000 -qt 1 -q 3 -qm 2 -f 0 -msg "Rosenblatt 5D Qunatization"

python nlp_test02.py -a 0 -s 1 -d 100 -t 4000 -qt 1 -f 0 -msg "Rosenblatt 100D"
python nlp_test02.py -a 1 -s 1 -d 100 -t 4000 -qt 1 -f 0 -msg "Rosenblatt 100D"
python nlp_test02.py -a 2 -s 1 -d 100 -t 4000 -qt 1 -f 0 -msg "Rosenblatt 100D"
python nlp_test02.py -a 3 -s 1 -d 100 -t 4000 -qt 1 -f 0 -msg "Rosenblatt 100D"

python nlp_test02.py -a 0 -s 1 -d 100 -t 4000 -qt 1 -q 3 -qm 2 -f 0 -msg "Rosenblatt 100D Qunatization"
python nlp_test02.py -a 1 -s 1 -d 100 -t 4000 -qt 1 -q 3 -qm 2 -f 0 -msg "Rosenblatt 100D Qunatization"
python nlp_test02.py -a 2 -s 1 -d 100 -t 4000 -qt 1 -q 3 -qm 2 -f 0 -msg "Rosenblatt 100D Qunatization"
python nlp_test02.py -a 3 -s 1 -d 100 -t 4000 -qt 1 -q 3 -qm 2 -f 0 -msg "Rosenblatt 100D Qunatization"

move TXT_Result.txt Rosenblatt_Result.txt
