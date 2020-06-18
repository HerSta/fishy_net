# fishy_net

## For the masters thesis YOFO is the place to be

## When pseudo-labeling with darknet:

navigate to /home/thesis-herman/fishy_net/darknet at our remote server.
```
./darknet detector test fish2.data fish2.cfg ../hermans_configs/fish2_last.weights -thresh 0.25 -dont_show -save_labels < pseudo.txt
```
The file pseudo.txt must include the paths to all the images that you want to label. The actual labels will be located at the same location as the images.