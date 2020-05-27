# fishy_net


This repo consists of 3 main programs:



* Darknet - for running detections and classifications on images
* Labelimg - for labeling and verifying that images has been labeled correctly
* Data_generator - for splitting data and enabling darknet to process data


## When pseudo-labeling with darknet:

navigate to /home/thesis-herman/fishy_net/darknet at our remote server.
```
./darknet detector test fish2.data fish2.cfg ../hermans_configs/fish2_last.weights -thresh 0.25 -dont_show -save_labels < pseudo.txt
```
The file pseudo.txt must include the paths to all the images that you want to label. The actual labels will be located at the same location as the images.