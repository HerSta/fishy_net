import sys 
import os
sys.path.append(os.path.abspath("t9ek80"))
from t9ek80.StructEchoTraceWBT import StructEchoTraceWBT
import pickle


my_file = open("t9ek80/fish_singletargets_testlist.bin", "rb")


objs = []
while True:
    try:
        objs.append(pickle.load(my_file))
    except EOFError:
        break


print("hi")
