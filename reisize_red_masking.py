import cv2
import numpy as np
import glob
import re

# getting the images
IMG_SHAPE = (576, 160)
# put correct location of images
data_folder = 'Folder in which images are there'
count = 1
output_folder = 'Folder in which output or final image will be their'
destin = glob.glob(data_folder + "*.png")
destin.sort(key=lambda f: int(re.sub('\D', '', f)))
#print(destin)
for file2image in destin:
    img1 = cv2.imread(file2image)
    # setting the ranges:
    img1 = cv2.resize(img1, IMG_SHAPE)
    #  red done
    lower_red = np.array([0,0,199])      # [0,0,195] # [0,0,218]
    upper_red = np.array([160,245,255])   # [160,240,255]
    mask1 = cv2.inRange(img1, lower_red, upper_red)
    red = cv2.bitwise_and(img1, img1, mask=mask1)
    
    red[np.where((red==[0,0,255]).all(axis=2))]=[255,0,255] 
    red[np.where((red!=[255,0,225]).all(axis=2))]=[0,0,255] 
    red[np.where((red==[0,0,0]).all(axis=2))]=[0,0,255] 

    name = output_folder + '/umm_road_'+ str(count) + '.png'
    cv2.imwrite(name, red)

    count += 1

    # cv2.imshow("mask1", mask1)
    # cv2.imshow('red', red)
    # cv2.imshow('img1',img1)
    # cv2.imshow('img2',img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

count = 1
