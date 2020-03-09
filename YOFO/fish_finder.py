import cv2
import glob
import math
import os
import operator
import csv

import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta

img_width = 1920
img_height = 1080

I_x = 1920
I_y = 1080

def create_img_label_path_dict(path):
    img_names = [img for img in glob.glob(path + "*.jpg")]
    img_labels = [lab for lab in glob.glob(path + "*.txt")]
    imgs_labels = dict(zip(img_names, img_labels))
    return imgs_labels

def calculate_sonar_region():
    son_depth = 12 #meters

    cam_hfov = math.radians(92) 
    son_fov = math.radians(7)
    t_x = 0.1 # distance between sonar and camera

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



def find_fish_in_sonar_region(labels,radius, sonar_center):
    number_of_fish = 0
    timestamp = ""

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
                    timestamp = correct_timestamp(timestamp)


    if number_of_fish > 0:
        return number_of_fish, timestamp
    else:
        return 


def correct_timestamp(timestamp):
    """
    The filename and images contain different timestamps. I assume that the time written in the image is correct, while the filenames are wrong. There error
    is either 26 or 27 seconds seemingly randomly.
    """
    error_seconds = 26
    utc_time = datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S")
    utc_time = utc_time + timedelta(seconds=error_seconds)

    return utc_time


def plot_singletargets():
    times = []
    depths = []

    


    with open ("data/fish_singletargets2.csv", "r") as file:
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



def main():
    fish_root = os.getcwd()

    #relative path to images
    img_path = "data/cam_with_labels/20190303/"
    imgs_labels_paths = create_img_label_path_dict(img_path)

    imgs = list(imgs_labels_paths.keys())
    


    # Calculate radius and offset of sonar region
    R_i_s, t_i_x = calculate_sonar_region()
    sonar_center = sonar_region_center(t_i_x)

    # Store all images with their sonar regions
    #for img_path in imgs:
    #    mark_sonar_region(img_path, R_i_s, t_i_x)

    # Find number of fish within the sonar region and their timestamps
    number_of_images_with_fish_in_sonar_region = 0
    for img_path, label_path in imgs_labels_paths.items():
        #img = cv2.imread(img_path)
        #display_labels(img, label_path)
        result = find_fish_in_sonar_region(label_path, R_i_s, sonar_center)
        if result != None:
            number_of_images_with_fish_in_sonar_region += 1
            print(result)
    print(number_of_images_with_fish_in_sonar_region)


    #display_labels(cv2.imread(list(imgs_labels_paths.keys())[12]), list(imgs_labels_paths.values())[12])
    
    



    print("Done!")


plot_singletargets()