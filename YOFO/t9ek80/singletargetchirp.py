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


class ek80(t9ek80.t9ek80):
    
    def __init__(self, argv):
        super(ek80, self).__init__(argv)

    def report(self, Payload, Decode, timenow, mtype, desimate):
    
        print("Working...")
        # THESE ARE FOR EXITING THE PROGRAM AFTER WE HAVE FETCHED AND STORED ALL OUR VALUES
        global df
        global startime

        # dt is datetime
        dt_timenow = datetime.strptime(timenow, "%Y-%m-%dT%H:%M:%SZ") + timedelta(hours=1)

        if dt_timenow == startime:
            print("DONE")
            df.to_csv("fish_singletargets.csv", index=False)
            os._exit(1)
        if startime == "":
            startime = dt_timenow

        

        if mtype == "SingleTarget":
            for element in Payload:
                df = df.append({"Time" : dt_timenow, "Depth": element[0], "Forward" : element[3], "Side" : element[4], "Sa" : element[5]}, ignore_index=True)



startime = ""
df = pd.DataFrame(columns=["Time", "Depth", "Forward", "Side", "Sa"])
run = ek80(sys.argv)
run.main()
print("Done!")












                #print("Time:    {:s}   Depth:   {:f}   Forward: {:f}   Side:    {:f}   Sa:      {:f}"\
                    #.format(timenow,element[0],element[3],element[4],element[5]))