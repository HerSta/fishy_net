import cv2
import glob
import math
import os
import operator
import csv
import pandas as pd


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator # to get integer labels
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta

img_width = 1920
img_height = 1080

I_x = 1920
I_y = 1080

save_plots = False
show_plots = False

def create_img_label_path_dict(path):
    img_names = [img for img in glob.glob(path + "*.jpg")]
    img_labels = [lab for lab in glob.glob(path + "*.txt")]
    imgs_labels = dict(zip(img_names, img_labels))
    return imgs_labels

def calculate_sonar_region():
    son_depth = 12 #meters

    cam_hfov = math.radians(92) 
    son_fov = math.radians(7)
    t_x = 0.2 # distance between sonar and camera

    x_o = 2*(math.sin(cam_hfov / 2) *(son_depth)) / (math.sin((math.pi / 2) - (cam_hfov / 2)))
    print(x_o)
    #y_0 = (I_y * x_o) / I_x

    R_s =  (math.sin(son_fov / 2) *(son_depth)) / (math.sin((math.pi / 2) - (son_fov / 2)))

    img_ratio = img_height / img_width

    cam_rect_a = img_ratio * img_width

    # Scaling factor between world and image
    S = I_x / x_o

    # Image dimensions of sonar radius and offset
    R_i_s = R_s * S
    t_i_x = S * t_x

    print("Sonar radius in image: " + str(R_i_s))
    print("Offset from center of image: " + str(t_i_x))

    return R_i_s, t_i_x





def transparent_circle(img,center,radius,color,thickness):
    center = tuple(map(int,center))
    rgb = [255*c for c in color[:3]] # convert to 0-255 scale for OpenCV
    alpha = 0.2 
    radius = int(radius)
    if thickness > 0:
        pad = radius + 2 + thickness
    else:
        pad = radius + 3
    roi = slice(center[1]-pad,center[1]+pad),slice(center[0]-pad,center[0]+pad)

    try:
        overlay = img[roi].copy()
        cv2.circle(img,center,radius,rgb, thickness=thickness, lineType=cv2.LINE_AA)
        opacity = alpha
        cv2.addWeighted(src1=img[roi], alpha=opacity, src2=overlay, beta=1. - opacity, gamma=0, dst=img[roi])
    except:
        return

    return img

   




def mark_sonar_region(img_path, R_i_s, t_i_x):
    img_center_x = img_width // 2
    img_center_y = img_height // 2
    img = transparent_circle(cv2.imread(img_path), (img_center_x + round(t_i_x), img_center_y), R_i_s, (255,0,0), -1)

    #cv2.imshow("x", img)
    #cv2.waitKey()
    img_name = img_path.rsplit("/", 1)
    cv2.imwrite("imgs/imgs_with_sonar_region/" + img_name[1], img)



def crop_img(img):
    return

def display_labels(img, labels):
    """
    img: one image
    labels: labels connected to img

    """
    # file format: The initial 0 correspond to the 0th class; fish. The next two numbers are the $(x,y)$ 
    # position of the center of the box while the last two numbers $(w,h)$ are the width and the height of the box relative to the image width and height.
    with open(labels) as fp:
        lines = fp.readlines()

        for line in lines:
            if line == "":
                break

            c, x, y, w, h = [float(x) for x in line.split()]

            center_x = int(x * img_width)
            center_y = int(y * img_height)

            box_width = int(w * img_width)
            box_height = int(h * img_height)

            top_left_x = center_x - box_width // 2
            top_left_y = center_y + box_height // 2

            bot_right_x = center_x + box_width // 2
            bot_right_y = center_y - box_height // 2

            cv2.rectangle(img, (top_left_x, top_left_y), (bot_right_x, bot_right_y), (0,0,255),2)
    cv2.imshow("x", img)
    cv2.waitKey()

def sonar_region_center(offset):
    img_center_x = img_width // 2
    img_center_y = img_height // 2

    sonar_center_x = img_center_x + offset
    sonar_center_y = img_center_y

    center = Point(sonar_center_x,sonar_center_y)
    return center


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y



def find_fish_in_sonar_region(labels, radius, sonar_center):
    """
    This function checks how many of the labeled fish falls within the sonar center.
    The timestamp is extracted from the filename of the labels. 


    Return: number_of_fish, timestamp
    """
    number_of_fish = 0

    with open(labels) as fp:
            lines = fp.readlines()

            for line in lines:
                if line == "":
                    break
                    
                c, x, y, w, h = [float(x) for x in line.split()]

                center_x = int(x * img_width)
                center_y = int(y * img_height)

                box_center = (center_x, center_y)

                # check if the center of the bounding box is within the sonar circle using x^2 + y^2 < r^2
                if (math.pow(center_x - sonar_center.x,2) + (math.pow(center_y - sonar_center.y, 2))) < math.pow(radius, 2):
                    number_of_fish += 1
    # if there actually is a fish we also want to get the timestamp
    labels_name = labels.rsplit("/", 1)
    timestamp = labels_name[1][:-4] #remove .txt 
    #timestamp = datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S")
    timestamp = correct_timestamp(timestamp)


    if number_of_fish > 0:
        return number_of_fish, timestamp
    else:
        return 0, timestamp


def correct_timestamp(timestamp):
    """
    The filename and images contain different timestamps. I assume that the
     time written in the image is correct,
     while the filenames are wrong. The error
    is either 26 or 27 seconds seemingly randomly.
    It is also shown that it continuously growes larger over time. Like in
    august it might be several minutes.
    """
    error_seconds = 0
    utc_time = datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S")
    utc_time = utc_time + timedelta(seconds=error_seconds)

    return utc_time


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


def cfish_ts():
    """
    Find number of fish within the sonar region and their timestamps
    Creates a plot that shows the number of fish within the sonar region for a time period.

    cfish are fish found in camera images

    Returns: 
    ts - a timeseries containing all the detections made. Values are number of fish detected.
    """

    imgs_labels_paths, R_i_s, _, sonar_center = init_data()

    timeseries = []
    for _, label_path in imgs_labels_paths.items():
        #img = cv2.imread(img_path)
        #display_labels(img, label_path)
        number_of_fish, timestamp = find_fish_in_sonar_region(label_path, R_i_s, sonar_center)

        timeseries.append((timestamp, number_of_fish))
    
    zlst = list(zip(*timeseries))

    ts = pd.Series(zlst[1], index=zlst[0], name="cfish")

    ax = plt.figure().gca()
    bar_width = 5.0 / len(ts.index)
    plt.bar(ts.index, ts.values, bar_width)

    plt.title('Fish found in the sonar region of the images by optical analysis')
    plt.ylabel('Number of fish')
    plt.xlabel('Time')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.gcf().autofmt_xdate() #make the xlabel slightly prettier
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))


    if save_plots:
        plt.savefig("figures/Number_cfish.png")
        print("Figure saved to the figures folder!")
    if show_plots:
        plt.show()
        
    return ts
        

def sfish_ts():
    ts = pd.read_csv("t9ek80/fish_singletargets_70db.csv", parse_dates=[0])
    

    start_time = "2019-03-03 07:00:00"
    end_time = "2019-03-03 17:00:00"

    ts = ts[ts["Time"] < end_time]

    
    
    time = ts["Time"]
    #s = pd.Series(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"))
    #time = time.append(s)
    #time = time.sort_values(axis=0)


    depth = ts["Depth"]
    #k = pd.Series("0")
    #depth = depth.append(k)
    
    
    is_fishes = [1] * len(depth)
    
    ax = plt.figure().gca()
    bar_width = 1.0 / len(ts.index)
    plt.bar(time, is_fishes, bar_width)

    #plt.plot(time, depth, "x")

    plt.ylabel('Fish present')
    plt.xlabel('Time')
    #ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.gcf().autofmt_xdate() #make the xlabel slightly prettier
    #ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))


    if save_plots:
        plt.savefig("figures/Number_sfish.png")
        print("Figure saved to the figures folder!")
    if show_plots:
        plt.show()
        
 
    
    return ts

def init_data():
    """
    Gets and returns the paths to all the images and their labels in a dictionary.

    Return: imgs_labels_paths
    """
    
    #fish_root = os.getcwd()

    #relative path to images
    img_path = "data/imgs/20190303/"
    imgs_labels_paths = create_img_label_path_dict(img_path)
    
    # Calculate radius and offset of sonar region
    R_i_s, t_i_x = calculate_sonar_region()
    sonar_center = sonar_region_center(t_i_x)

    return imgs_labels_paths, R_i_s, t_i_x, sonar_center


def compare_cfish_sfish():
    cfish = cfish_ts()
    sfish = sfish_ts()


    #Before merging cfish and sfish, cfish must be a dataframe and not a series
    cfish = pd.DataFrame(data = cfish.values, index = cfish.index)
    cfish.columns = ["Num_fish in img"]

    # remove cfish rows that do not have fish in them
    cfish = cfish[cfish["Num_fish in img"] != 0]

    #maybe check plus-minus one second diff

    total_cfish = len(cfish)
    total_sfish = len(sfish)
    print("=========================")
    print("Comparing cfish and sfish!")
    print("Total timestamps with detected cfish: " + str(total_cfish) + " sfish: " + str(total_sfish))

    #maybe create a function to make sfish and cfish have the same start/stop times

  
    sfish.set_index("Time", inplace=True)

    #how="inner" means to take the intersection of keys
    #on=None means to use the index as keys
    tfish = pd.merge(cfish, sfish, how="inner", on=None, left_index=True, right_index=True)

    tfish = tfish[tfish["Num_fish in img"] != 0]

    print("The total number of timestamps where a fish was found both with sonar and camera: " + str(len(tfish)))
    print("Since a million fish during the afternoon... we only look from 09:40 to 10:00")

    cstart_time = "2019-03-03 09:40:00"
    cend_time = "2019-03-03 10:00:00"


    # Within 20 min the camera found for example 10 fish. We now want to find the time windows in the 
    # sonar data where all the same 10 fish are found. We expect the sonar to find MORE than 10, since it
    # records data continuously. However we require that all 10 cfish be identified as sfish.

    # Now morning_cfish contains 20 minutes of detections
    morning_cfish = cfish[cfish.index < cend_time]
    morning_cfish = morning_cfish[morning_cfish.index > cstart_time]

    # Now we need a routine that finds the maximum detections in common between sonar and camera
    
    dets_time = shift(cfish, sfish, 60*60*4, 60*60*1) #shift sfish 1 hour backwards and 4 forwards
    plt.plot(dets_time)
    plt.xlabel("Seconds shifted")
    plt.ylabel("Number of detections in common")
    plt.show()

    print("Morning cfish: " + str(len(morning_cfish)) + " Morning sfish: " + str(len(morning_sfish))+ " Morning tfish: " + str(len(morning_tfish)))
    print("hi")

def shift(ts1, ts2, forward, backward):
    """
    Takes two timeseries and compares them while shifting ts1 forwards and backwards. ts2 stays fixed.
    """
    max_cdets = len(ts1) # this is 6
    max_dets = 0

    max_time_start = ""
    max_time_end = ""


    dets_time = []
    for i in range(-backward, forward):
        sstart_time = "2019-03-03 09:40:00"
        send_time = "2019-03-03 10:00:00"


        sstart_time_utc = datetime.strptime(sstart_time, "%Y-%m-%d %H:%M:%S")
        sstart_time_utc = sstart_time_utc + timedelta(seconds=i)

        send_time_utc = datetime.strptime(send_time, "%Y-%m-%d %H:%M:%S")
        send_time_utc = send_time_utc + timedelta(seconds=i)

        # sfish contains all sonar detections from an entire day
        # here we just take a 20 min interval of detections
        ts2_subset = ts2[ts2.index < send_time_utc]
        ts2_subset = morning_sfish[morning_sfish.index > sstart_time_utc]

#        morning_tfish = common[common.index < send_time_utc]
#        morning_tfish = morning_tfish[morning_tfish.index > sstart_time_utc]

        common = pd.merge(ts1, ts2_subset, how="inner", on=None, left_index=True, right_index=True)

        num_dets = len(common)
        if num_dets > max_dets:
            max_dets = num_dets
            max_time_start = sstart_time_utc
            max_time_end = send_time_utc

        if num_dets == max_cdets:
            print("Best timeinterval found!")
            continue

        # log the number of detections at this particular time
        dets_time.append((num_dets, i))

    return dets_time



def main():
    global save_plots 
    global show_plots 

    save_plots = False
    show_plots = False

    compare_cfish_sfish()


    #imgs_labels_paths, R_i_s, t_i_x, sonar_center = init_data()
    #cfish_ts(imgs_labels_paths, R_i_s, sonar_center)
    #sfish_ts()
    
    # Store all images with their sonar regions
    #for img_path in imgs:
    #    mark_sonar_region(img_path, R_i_s, t_i_x)

   
   
   
   
   
   
   
   
   
   


    #display_labels(cv2.imread(list(imgs_labels_paths.keys())[12]), list(imgs_labels_paths.values())[12])
    
    



    print("Done!")


main()