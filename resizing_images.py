from itertools import filterfalse
import cv2
import numpy as np
import os
import glob


IMG_SHAPE = (576, 160)

data_folder = './'
count = 1
#data = glob(os.path.join(data_folder, '*.png'))
#print(data)
for file2image in glob.glob("*.png"):
    #for file2image in glob(os.path.join(data_folder, '*.png')):
    print("Hello")
    image = cv2.imread(file2image)
    resized_img = cv2.resize(image, IMG_SHAPE)
    # cv2.imshow('frame', resized_img)
    name = str(count) + 'resized' + '.png'
    cv2.imwrite(name, resized_img)
    count += 1
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()










