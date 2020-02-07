from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "fish2_last.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    
    return parser.parse_args()
    
args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = False



num_classes = 1
classes = load_classes("data/fish.names")



#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32


modules = model.module_list




#Set the model in evaluation mode
model.eval()

read_dir = time.time()
#Detection phase
try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print ("No file or directory with the name {}".format(images))
    exit()
    
if not os.path.exists(args.det):
    os.makedirs(args.det)

load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]

im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)


leftover = 0
if (len(im_dim_list) % batch_size):
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover            
    im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                        len(im_batches))]))  for i in range(num_batches)]  

write = 0



import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

def PCA2(data, k=2):
 # preprocess the data
    X = data
    X_mean = torch.mean(X,0)
    X = X - X_mean.expand_as(X)

    # svd
    U,S,V = torch.svd(torch.t(X))
    return torch.mm(X,U[:,:k])

def comp_2d(image_2d): # FUNCTION FOR RECONSTRUCTING 2D MATRIX USING PCA
	cov_mat = image_2d - np.mean(image_2d , axis = 1)
	eig_val, eig_vec = np.linalg.eigh(np.cov(cov_mat)) # USING "eigh", SO THAT PROPRTIES OF HERMITIAN MATRIX CAN BE USED
	p = np.size(eig_vec, axis =1)
	idx = np.argsort(eig_val)
	idx = idx[::-1]
	eig_vec = eig_vec[:,idx]
	eig_val = eig_val[idx]
	numpc = 100 # THIS IS NUMBER OF PRINCIPAL COMPONENTS, YOU CAN CHANGE IT AND SEE RESULTS
	if numpc <p or numpc >0:
		eig_vec = eig_vec[:, range(numpc)]
	score = np.dot(eig_vec.T, cov_mat)
	recon = np.dot(eig_vec, score) + np.mean(image_2d, axis = 1).T # SOME NORMALIZATION CAN BE USED TO MAKE IMAGE QUALITY BETTER
	recon_img_mat = np.uint8(np.absolute(recon)) # TO CONTROL COMPLEX EIGENVALUES
	return recon_img_mat


start_det_loop = time.time()
for i, batch in enumerate(im_batches):
    start = time.time()

    for l in range(85,95):
        if l == 83 or l == 84 or l==94 or l == 95 or l == 107:
            continue



        with torch.no_grad():
            prediction = model(Variable(batch), False, l)


             
            images = prediction.squeeze().data.numpy()[:,:,:]
            
            # READ DIMENSIONS OF IMAGE
            #print("Original images shape: " + str(images.shape))
            depth,width, height = images.shape[0], images.shape[1], images.shape[2]

            # SWAP AXIS FROM 32,416,416 TO 416,416,32
            images = np.swapaxes(images, 0, 1)
            images = np.swapaxes(images, 1, 2)



            # Reshaping images into array (416,416,32) becomes (416*416,32)
            images = images.reshape((width*height, depth), order="C")
            print("Reshaped images shape: " + str(images.shape))

            #img_to_plot = np.reshape(images, (width, height, depth))
            #plt.imshow(img_to_plot[:,:,0])
            #plt.show()


            # as we scale down the layers become deeper than the width/height of the images
            # producing more eigenvalues than measures
           

            pca = PCA()

            score  = pca.fit_transform(images)

            #print("score shape: " + str(score.shape))
            #if depth > width*height:
            #    depth = max(width, height)
            #    phi = np.reshape(score, (height, width, depth*depth), order="C")
            #else:
            #    phi = np.reshape(score, (height, width, depth), order="C")
            #plt.imshow(phi[:,:,0])
            #plt.show()
            #print("phi shape: " + str(phi.shape))
            
            
            
            #for p in range(2):
            #    plt.imshow(np.real(phi[:,:,p]))
            #    plt.axis("off")
            #    #plt.show()
            #    plt.savefig("C:\\Users\\peterhs\\Dropbox\\Apper\\Overleaf\\Project_thesis\\figures\\internal_yolo\\traveling_through\\layer_backup_pca_img_" + str(l+1) + "_" + str(p), pad_inches=0, bbox_inches="tight")
            #print("hi")

        
        
        
            var_r = pca.explained_variance_ratio_
            #img = score.reshape((416,416), order="C")

            #fig = plt.figure()

            plt.bar([1, 2, 3, 4, 5], [var_r[0],var_r[1],var_r[2],var_r[3],var_r[4]])
            ax = plt.axes()
            ax.set_ylabel("Variance Ratio")
            ax.set_xlabel("Principal Component")

        plt.savefig("C:\\Users\\peterhs\\tmp\\layer_pca_" + str(l+1) + "_" + str(0), pad_inches=0, bbox_inches="tight")
        plt.clf()
            #plt.show()
            # for pred in prediction:
            #     for idx,image in enumerate(pred):
            #         if idx == 0:
            #             continue
            #         if idx == 2:
            #             break
            #                           

            #         #normalizer = interp1d([image.min(),image.max()], [0,1])
            #         try:
            #             plt.imshow(image, cmap="viridis")
            #         except:
            #             print("error at layer: " + str(l+1))


            # plt.axis('off') #removes the numbers around the image



def remove_topbot(img):
    img_h = img.shape[1]

    # these numbers come from the original image which is not quadratic
    percent_to_remove = 91 / 416

    lower = int(percent_to_remove * img_h)
    upper = int((1 - percent_to_remove) * img_h)

    #for default = 91, 325

    img = img[lower:upper,:]
    return img


#   prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)
#
#    end = time.time()
#
#    if type(prediction) == int:
#
#        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
#            im_id = i*batch_size + im_num
#            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
#            print("{0:20s} {1:s}".format("Objects Detected:", ""))
#            print("----------------------------------------------------------")
#        continue
#
#    prediction[:,0] += i*batch_size    #transform the attribute from index in batch to index in imlist 
#
#    if not write:                      #If we have't initialised output
#        output = prediction  
#        write = 1
#    else:
#        output = torch.cat((output,prediction))
#
#    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
#        im_id = i*batch_size + im_num
#        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
#        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
#        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
#        print("----------------------------------------------------------")
    


def first_convolution():
    #plotting our actual image
    my_img = prep_image_plottable(loaded_ims[0], inp_dim)
    #plt.imshow(my_img)

    #fetch the weights from the first layer
    weights = modules[0].conv_0.weight


    color_channels = 3
    from scipy import ndimage, signal
    for i in range(32):
        filter = weights.data.numpy()[i] #3x3x3 matrix

        #filter = np.array([[[0,0,0],  [0,0,0], [0,0,0]], [[0,0,0], [0,1,0],  [0,0,0]], [[0,0,0],  [0,0,0], [0,0,0]]])
        pixel_grid = np.zeros((416,416))
        for j in range(color_channels):
            pixel_grid += ndimage.convolve(my_img[:,:,j], filter[:,:,j], mode="constant", cval=0.0)
            #print(j)

        #filtered_img = ndimage.convolve(my_img, filter)
        pixel_grid_cropped = pixel_grid[91:325,:]
        plt.imshow(pixel_grid_cropped, cmap="gray")
        plt.axis('off')
        #plt.savefig("C:\\Users\\peterhs\\Dropbox\\Apper\\Overleaf\\Project_thesis\\figures\\internal_yolo\\layer_last_" + str(i), bbox_inches="tight")
        #plt.show()
        

    #apply the weights to the image


    #dont forget about bias and activation function!!!!

    #plot image
    #plt.show()

    print("hi")


first_convolution()
   
try:
    output
except NameError:
    print ("No detections were made")
    exit()

im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)


output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2



output[:,1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
    
output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("pallete", "rb"))

draw = time.time()


def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


list(map(lambda x: write(x, loaded_ims), output))

det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("\\")[-1]))

list(map(cv2.imwrite, det_names, loaded_ims))


end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")

