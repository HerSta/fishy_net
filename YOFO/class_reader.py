import sys 
import os
sys.path.append(os.path.abspath("t9ek80"))
from t9ek80.StructEchoTraceWBT import StructEchoTraceWBT
import pickle
import numpy as np
import visualizer

def get_sfish():
    my_file = open("t9ek80/fish_singletargets_testlist.bin", "rb")
    sfish = []
    while True:
        try:
            sfish.append(pickle.load(my_file))
        except EOFError:
            break
    return sfish




def main():
    sfish = get_sfish()
    visualizer.plot_sfish_freq(sfish)
    print("hi")
