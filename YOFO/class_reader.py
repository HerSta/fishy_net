import sys 
import os
sys.path.append(os.path.abspath("t9ek80"))
from t9ek80.StructEchoTraceWBT import StructEchoTraceWBT
import fish_finder
import pickle
import numpy as np
import visualizer
import pandas as pd

def get_sfish(filename):
    my_file = open(filename, "rb")
    sfish = []
    while True:
        try:
            sfish.append(pickle.load(my_file))
        except EOFError:
            break
    return sfish


def csv_remove_dups():
    file_name = "t9ek80/fish_singletargets_100db.csv"
    file_name_output = "t9ek80/fish_singletargets_100db_nodups.csv"

    df = pd.read_csv(file_name, sep=",")

    # Notes:
    # - the `subset=None` means that every column is used 
    #    to determine if two rows are different; to change that specify
    #    the columns as an array
    # - the `inplace=True` means that the data structure is changed and
    #   the duplicate rows are gone  
    df.drop_duplicates(subset=None, inplace=True)

    # Write the results to a different file
    df.to_csv(file_name_output)


def prepare_data():
    cfish = fish_finder.cfish_df()
    cfish.columns = ["Fish"]
    threshold_value = 1
    replace_value = 1
    cfish["Fish"] = cfish["Fish"].where(cfish["Fish"] <= threshold_value, replace_value)
    #cfish.columns = ["Fish"]

    sfish = get_sfish()
    data = {"Frequency": [(x.compensatedFrequencyResponse) for x in sfish]}
    sfish_df = pd.DataFrame(data, columns=["Frequency"], index=[str(x.time) for x in sfish])

    sfish_df = sfish_df.sort_index()

    start = "2019-03-03 07:00:00"
    end   = "2019-03-03 17:00:00"

    _,_,common = fish_finder.get_common(sfish_df, cfish,start, end)

    common.to_csv("fish_learning_data.csv", header=False)

    return "hi"



def main():
    csv_remove_dups()
    #prepare_data()



    #sfish = get_sfish()
    #for i in range(10):
    #    visualizer.plot_sfish_freq(sfish[i], show=False, save=True)
    #print("hi")


if __name__=="__main__":
    main()