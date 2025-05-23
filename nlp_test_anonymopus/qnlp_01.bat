echo off
echo ###################################################################
echo   This is the batch file to test Quantized NLP 
echo   Usage  qnlp.bat 
echo   Example   : qnlp 
echo ###################################################################

del TXT_Result.txt
del Full_Result_NLP.txt
del Report_NLP.txt
del nlp_results
echo on

python test_nlp01.py 


