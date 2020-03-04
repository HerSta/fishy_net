import cv2
import glob
import math
import os


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

    cam_fov = math.radians(100) 
    son_fov = math.radians(7)
    t_x = 0.1 # distance between sonar and camera

    


    R_o = (math.sin(cam_fov / 2) *(son_depth)) / (math.sin((math.pi / 2) - (cam_fov / 2)))
    print(R_o)



    R_s =  (math.sin(son_fov / 2) *(son_depth)) / (math.sin((math.pi / 2) - (son_fov / 2)))



    img_ratio = img_height / img_width

    cam_rect_a = img_ratio * img_width

    # width of camera in world coordinates
    y_o = 2 * R_o / math.sqrt(math.pow(I_x / I_y, 2) + 1)

    # Scaling factor between world and image
    S = I_y / y_o

    # Image dimensions of sonar radius and offset
    R_i_s = (R_s / R_o) * math.sqrt(math.pow(img_height,2) + math.pow(img_width,2))
    t_i_x = S * t_x

    print("Sonar radius in image: " + str(R_i_s))
    print("Offset from center of image: " + str(t_i_x))

    return R_i_s, t_i_x





def transparent_circle(img,center,radius,color,thickness):
    center = tuple(map(int,center))
    rgb = [255*c for c in color[:3]] # convert to 0-255 scale for OpenCV
    alpha = 0.1 
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




def main():
    fish_root = os.getcwd()

    #relative path to images
    img_path = "imgs/cam_with_labels/20190303/"
    imgs_labels_paths = create_img_label_path_dict(img_path)

    imgs = list(imgs_labels_paths.keys())

    #display_labels(cv2.imread(list(imgs_labels_paths.keys())[12]), list(imgs_labels_paths.values())[12])
    
    R_i_s, t_i_x = calculate_sonar_region()
    
    for img_path in imgs:
        mark_sonar_region(img_path, R_i_s, t_i_x)



    print("Done!")


main()