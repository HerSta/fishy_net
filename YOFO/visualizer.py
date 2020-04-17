import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator # to get integer labels
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import t9ek80.StructEchoTraceWBT
from datetime import timedelta

def plot_sfish(ts, threshold, show=True, save=False):
    time = ts["Time"]
    # count number of fish
    t_prev = time[0]
    detections = [1]
    counter = 0
    for t in time[1:]:
        if t == t_prev:
            detections[counter] += 1
        else:
            detections.append(1)
            counter += 1
        t_prev = t

    #del detections[-1]

    time = time.drop_duplicates()

    ax = plt.figure().gca()
    bar_width = 0.9 / len(detections)
    plt.bar(time, detections, bar_width)
    plt.ylabel('Number of Fish')
    plt.xlabel('Time')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gcf().autofmt_xdate() #make the xlabel slightly prettier
    #ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    if save:
        plt.savefig("figures/Number_sfish" + str(threshold) + "db_nodiscard.png")
        print("Figure saved to the figures folder!")
    if show:
        plt.show()


def plot_singletargets():
    times = []
    depths = []

    with open ("t9ek80/fish_singletargets_60db.csv", "r") as file:
        reader = csv.reader(file)
        sortedlist = sorted(reader, key=operator.itemgetter(0), reverse=False)
        count = 0
        for row in sortedlist:
            if row[0] == "Time":
                continue
            utc_time = row[0][11:]
            #utc_time = datetime.strptime(utc_time,"%H:%M:%S")

            times.append(utc_time)
            depths.append(float(row[1]))

    time_start = datetime.strptime(times[0], "%H:%M:%S")
    time_end = datetime.strptime(times[-1], "%H:%M:%S")

    total_seconds = (time_end - time_start).total_seconds()

    all_times = []
    all_depths = []
    now = time_start
    while now != time_end:
        all_times.append(now.strftime("%H:%M:%S"))
        all_depths.append(0)
        now += timedelta(seconds=1)


    # Remove all duplicate timesteps. These are made when detecting several fish simultaneously
    count = 1
    current_time = times[0]
    for time in times[1:]:   
        if time == current_time:
            del times[count]
            del depths[count]
            
            current_time = times[count-1]
        else:
            current_time = time
            count += 1


    all_count = 0
    count = 0
    for all_time in all_times:
        if all_time == times[count]:
            all_depths[all_count] = depths[count]
            count += 1
            all_count += 1
        else:
            all_count += 1

    plt.plot(all_times, all_depths)
    plt.xticks([time_start.strftime("%H:%M:%S"), all_times[round(len(all_times) / 2)],time_end.strftime("%H:%M:%S")])
    plt.ylabel("Meters [m]")
    plt.xlabel("Time [s]")
    plt.show()



def plot_sfish_freq(sfish):
    """
    Plot the chirp frequency for a single fish
    """

    #QUESTION: how does the frequency match the frequency response?

    frequencies = np.linspace(185, 255, len(sfish.uncompensatedFrequencyResponse))
    plt.plot(frequencies, sfish.uncompensatedFrequencyResponse)
    plt.ylabel("Strength (dB)")
    plt.xlabel("Frequency")
    plt.show()
