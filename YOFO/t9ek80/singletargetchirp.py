# ----------------------------------------------------------------------------
#    Method       EK80 generic v1.0
#    Description  Subscribes to EK80 data depending on the config.xml file and report data.
#                 Comunicate with the EK80/EK60/EK15
#    By:          Kongsberg Maritime AS, Terje Nilsen 2020
#-----------------------------------------------------------------------------
import sys
import csv
import t9ek80
import pandas as pd
from datetime import datetime, timedelta
import os

import StructEchoTraceWBT
import pickle

startime = ""
filename = ""

class ek80(t9ek80.t9ek80):
    
    def __init__(self, argv):
        super(ek80, self).__init__(argv)

    def getDebug(self):
        return False

    def report(self, Payload, Decode, timenow, mtype, desimate):
        # THESE ARE FOR EXITING THE PROGRAM AFTER WE HAVE FETCHED AND STORED ALL OUR VALUES
        global startime

        # dt is datetime
        try:
            dt_timenow = datetime.strptime(timenow, "%Y-%m-%dT%H:%M:%SZ")
        except:
            dt_timenow = "null"

        if mtype == "SingleTarget" or mtype == "SingleTargetChirp":
            for element in Payload:
                print("New message arrived! Writing to file")
                if csv:
                    with open(filename, "a") as f:
                        f.write(str(dt_timenow) + ',' +  str(element[0]) + "," + str(element[3]) + "," + str(element[4]) + "," + str(element[5]))
                        f.write("\n")
                else:
                    echotrace = StructEchoTraceWBT.StructEchoTraceWBT(dt_timenow)
                    echotrace.populate(element)
                    with open(filename, "ab") as fp: #b is for binary mode
                        pickle.dump(echotrace, fp)
        if dt_timenow == startime:
            print("DONE")
            os._exit(1)
        if startime == "":
            startime = dt_timenow

        
        """

        if mtype == "SingleTarget" or mtype == "SingleTargetChirp":
            print("Writing a new entry to the csv file...")
            df.to_csv("fish_singletargets.csv", index=False)
            for element in Payload:
                df = df.append({"Time" : dt_timenow, "Depth": element[0], "Forward" : element[3], "Side" : element[4], "Sa" : element[5]}, ignore_index=True)"""


def main():
    global filename
    csv = True

    print("Waiting for messages.")
    if csv:
        filename = "fish_singletargets_70db_v2.csv"
        with open(filename, "w") as file:
            file.write("Time,Depth,Along,Athwart,Sa")
            file.write("\n")
    else:
        filename = "fish_singletargets_60db_testlist.bin"
    run = ek80(sys.argv)
    run.main()


main()