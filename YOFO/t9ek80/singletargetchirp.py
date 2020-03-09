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

    def getDebug(self):
        return False

    def report(self, Payload, Decode, timenow, mtype, desimate):
        # THESE ARE FOR EXITING THE PROGRAM AFTER WE HAVE FETCHED AND STORED ALL OUR VALUES
        global df
        global startime

        # dt is datetime
        dt_timenow = datetime.strptime(timenow, "%Y-%m-%dT%H:%M:%SZ") + timedelta(hours=1)


        if mtype == "SingleTarget" or mtype == "SingleTargetChirp":
            with open("fish_singletargets.csv", "a") as file:
                for element in Payload:
                    # Filter off elements that are within 2 meters of sonar or surface
                    if element[0] < 2 or element[0] > 10:
                        continue

                    print("New message arrived! Writing to csv")
                    file.write(str(dt_timenow) + ',' +  str(element[0])+ ',' + str(element[3]) + ',' + str(element[4]) + ',' + str(element[5]))
                    file.write("\n")
                    



        
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



startime = ""
print("Waiting for messages.")
#df = pd.DataFrame(columns=["Time", "Depth", "Forward", "Side", "Sa"])
with open("fish_singletargets.csv", "w") as file:
    file.write("Time, Depth, Forward, Side, Sa")
    file.write("\n")
run = ek80(sys.argv)
run.main()












                #print("Time:    {:s}   Depth:   {:f}   Forward: {:f}   Side:    {:f}   Sa:      {:f}"\
                    #.format(timenow,element[0],element[3],element[4],element[5]))