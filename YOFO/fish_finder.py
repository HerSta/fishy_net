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

import visualizer

img_width = 1920
img_height = 1080

I_x = 1920
I_y = 1080

son_depth = 14
cam_hfov = math.radians(92)
son_fov = math.radians(7) #actually 7
t_x = 0.2

save_plots = False
show_plots = False

def create_img_label_path_dict(path):
    img_names = [img for img in glob.glob(path + "*.jpg")]
    img_labels = [lab for lab in glob.glob(path + "*.txt")]
    imgs_labels = dict(zip(img_names, img_labels))
    return imgs_labels

def calculate_sonar_region():

    x_o = 2*(math.sin(cam_hfov / 2) *(son_depth)) / (math.sin((math.pi / 2) - (cam_hfov / 2)))
    print(x_o)
    #y_0 = (I_y * x_o) / I_x

    R_s =  (math.sin(son_fov / 2) *(son_depth)) / (math.sin((math.pi / 2) - (son_fov / 2)))

    img_ratio = img_height / img_width

    #cam_rect_a = img_ratio * img_width

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


def img_path_to_datetime(img_path):
    name = img_path.rsplit("/", 1)

    timestamp = datetime.strptime(name, "%Y-%m-%d-%H-%M-%S")
    print("hi")

def timestamp_to_img(timestamp):
    timestring = str(timestamp).replace(" ", "-").replace(":","-")
    path = "data/imgs/20190303/" + timestring + ".jpg"
    return cv2.imread(path)
    


def mark_sonar_region(img_path, R_i_s, t_i_x):
    img_center_x = img_width // 2
    img_center_y = img_height // 2
    img = transparent_circle(cv2.imread(img_path), (img_center_x + round(t_i_x), img_center_y), R_i_s, (255,0,0), -1)

    #cv2.imshow("x", img)
    #cv2.waitKey()
    #img_name = img_path.rsplit("/", 1)
    #cv2.imwrite("imgs/imgs_with_sonar_region/" + img_name[1], img)
    return img

def mark_sonar_region(img, R_i_s, t_i_x):
    img_center_x = img_width // 2
    img_center_y = img_height // 2
    img = transparent_circle(img, (img_center_x + round(t_i_x), img_center_y), R_i_s, (255,0,0), -1)

    #cv2.imshow("x", img)
    #cv2.waitKey()
    #img_name = img_path.rsplit("/", 1)
    #cv2.imwrite("imgs/imgs_with_sonar_region/" + img_name[1], img)
    return img

def plot_img_with_sonar_region(timestamp):
    imgs_labels_paths, R_i_s, t_i_x, sonar_center = init_data()
    img = timestamp_to_img(timestamp)
    img = mark_sonar_region(img, R_i_s, t_i_x)
    cv2.imshow("name",img)
    cv2.waitKey()
    print("hi")



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


def cfish_ts():
    """
    Find number of fish within the sonar region and their timestamps
    Creates a plot that shows the number of fish within the sonar region for a time period if specified globally.

    Returns: 
    ts - a timeseries containing all the detections made. Values are number of fish detected.
    """

    imgs_labels_paths, R_i_s, _, sonar_center = init_data()

    timeseries = []
    for _, label_path in imgs_labels_paths.items():
        number_of_fish, timestamp = find_fish_in_sonar_region(label_path, R_i_s, sonar_center)
        timeseries.append((timestamp, number_of_fish))
    
    zlst = list(zip(*timeseries))
    ts = pd.Series(zlst[1], index=zlst[0], name="cfish")


    if save_plots or show_plots:
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

def sfish_ts(threshold_db, start, end, discard):
    """
    Generates a timeseries of fish detected by the sonar between start and end.

    Returns: timeseries
    """
    ts = pd.read_csv("t9ek80/fish_singletargets_"+ str(threshold_db) + "db" + str(discard) + ".csv", parse_dates=[0])
    ts = ts.sort_values("Time")
    ts = ts[ts["Time"] < end]
    ts = ts[ts["Time"] > start] 
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

def compare_cfish_sfish(threshold_db, s_start, s_end, discard, compare_start, compare_end):
    cfish = cfish_ts()
    sfish = sfish_ts(threshold_db, s_start, s_end, discard)

    #Before merging cfish and sfish, cfish must be a dataframe and not a series
    cfish = pd.DataFrame(data = cfish.values, index = cfish.index)
    cfish.columns = ["Num_fish in img"]

    # remove cfish rows that do not have fish in them
    cfish = cfish[cfish["Num_fish in img"] != 0]

    total_cfish = len(cfish)
    total_sfish = len(sfish)
    print("Total timestamps with detected cfish: " + str(total_cfish) + " sfish: " + str(total_sfish))
  
    sfish.set_index("Time", inplace=True)

    sfish = sfish.loc[~sfish.index.duplicated(keep="first")]

    dets_time = shift(cfish, sfish,compare_start, compare_end, 1, 1, 30,30,) #shift sfish 1 hour backwards and 4 forwards

    get_common(cfish, sfish, compare_start, compare_end)
    return dets_time
    
def compare_no_shift(threshold_db, s_start, s_end, discard, compare_start, compare_end):
    cfish = cfish_ts()
    sfish = sfish_ts(threshold_db, s_start, s_end, discard)

    #Before merging cfish and sfish, cfish must be a dataframe and not a series
    cfish = pd.DataFrame(data = cfish.values, index = cfish.index)
    cfish.columns = ["Num_fish in img"]

    # remove cfish rows that do not have fish in them
    cfish = cfish[cfish["Num_fish in img"] != 0]

    total_cfish = len(cfish)
    total_sfish = len(sfish)
    sfish.set_index("Time", inplace=True)

    sfish = sfish.loc[~sfish.index.duplicated(keep="first")]


    cfish_m, sfish_m, common = get_common(cfish, sfish, compare_start, compare_end)

    print("Betwween " + compare_start + " and " + compare_end + " we have \n " + str(len(cfish_m)) + " cfish \n" + str(len(sfish_m)) + " sfish")
    for row in common.iterrows():
        #plot row[0]
        print("Timestamp for common fish: " + str(row[0]))
        plot_img_with_sonar_region(row[0])
        print("showing")



def get_cfish_detections(time_start, time_end):
    cfish = cfish_ts()
    cfish = cfish[cfish.index < time_end]
    cfish = cfish[cfish.index > time_start]
    cfish = cfish[cfish.values != 0]
    print("Between " + time_start + " and " + time_end + ", " + str(len(cfish)) + " fish was found in the optical data.")
    return len(cfish)

def get_common(ts1, ts2, time_start, time_end):
    ts1 = ts1[ts1.index < time_end]
    ts1 = ts1[ts1.index > time_start]

    ts2 = ts2[ts2.index < time_end]
    ts2 = ts2[ts2.index > time_start]

    return ts1, ts2, pd.merge(ts1, ts2, how="inner", on=None, left_index=True, right_index=True)

def shift(ts1, ts2, compare_start, compare_end, hours_forward, hours_backward, seconds_forward, seconds_backward):
    """
    Takes two timeseries and compares them while shifting ts1 forwards and backwards. ts2 stays fixed.
    """

    dets_time = []
    print("Started shifting...")
    # We do not need to check every second. just +/- 60 seconds for all hours
    for h in range(-hours_backward, hours_forward + 1):
        common = shift_seconds(ts1, ts2, compare_start, compare_end, seconds_backward, seconds_forward, at_hour=h)    
        dets_time.append((h,common))
    return dets_time

def shift_seconds(ts1, ts2, compare_start, compare_end, seconds_backward, seconds_forward, at_hour):
    """
    Returns a list of tuples at the form (seconds shifted, number of detections)
    """
    det_list = []
    for s in range(-seconds_backward, seconds_forward + 1):

        sstart_time_utc = datetime.strptime(compare_start, "%Y-%m-%d %H:%M:%S")
        sstart_time_utc = sstart_time_utc + timedelta(seconds=s) + timedelta(hours=at_hour)


        send_time_utc = datetime.strptime(compare_end, "%Y-%m-%d %H:%M:%S")
        send_time_utc = send_time_utc + timedelta(seconds=s) + timedelta(hours=at_hour)

        ts2_subset = ts2[ts2.index < send_time_utc]
        ts2_subset = ts2_subset[ts2_subset.index > sstart_time_utc]

        # since we find the common detections based on timestamps we re-add the subtracted time
        ts2_subset.index -= (timedelta(seconds=s) + timedelta(hours=at_hour))
        common = pd.merge(ts1, ts2_subset, how="inner", on=None, left_index=True, right_index=True)
        num_dets = len(common)

        # log the number of detections at this particular time
        det_list.append((s, num_dets))
    return det_list

def main():
    global save_plots 
    global show_plots 

    save_plots = False
    show_plots = False

    threshold_db = 65
    discard = "" #_nodiscard or nothing

    s_start = "2019-03-03 07:00:00"
    s_end = "2019-03-03 17:00:00"
    #sfish = sfish_ts(threshold_db, s_start, s_end)
    interest_start = "2019-03-03 09:00:00"
    interest_end = "2019-03-03 11:00:00"
    common = compare_no_shift(threshold_db, s_start, s_end, discard, interest_start, interest_end)

    #common = compare_cfish_sfish(threshold_db, s_start, s_end, discard, interest_start, interest_end)
    #cfish_n = get_cfish_detections(interest_start, interest_end)
    
    #visualizer.plot_shifted_commons(common,cfish_n, threshold_db, show=True, save=True)

    #visualizer.plot_sfish(sfish, threshold_db, show=True, save=True)

    #sfish_ts()

    #imgs_labels_paths, R_i_s, t_i_x, sonar_center = init_data()
    #cfish_ts(imgs_labels_paths, R_i_s, sonar_center)
    #sfish_ts()
    
    # Store all images with their sonar regions
    #for img_path in imgs:
    #    mark_sonar_region(img_path, R_i_s, t_i_x)
   
   
    #display_labels(cv2.imread(list(imgs_labels_paths.keys())[12]), list(imgs_labels_paths.values())[12])
    

    print("Done!")


main()