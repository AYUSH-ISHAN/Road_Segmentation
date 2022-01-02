

###  Also add the part of resizing the image and other such preprocessing part ... Or can even leave
### as it is .  (576, 160) - height, width while opposite order is followed in opencv.

import argparse 
import os
import cv2
import numpy as np

parser = argparse.ArgumentParser(description="Pass in the params for data loading")
parser.add_argument('--location', type=str, default=os.getcwd()+'//data_road', help='put in the file location of the image dataset')
parser.add_argument('--size', type=int, default=512,help='input size of the HD image')
parser.add_argument('--weights_location', type=str,default='./weights' ,help='put in location of image')

args = parser.parse_args()

dataset_location = args.location
input_img_size =  args.size 
weights_location = args.weights_location

train_img_loc = dataset_location + '//training//image_2//'
train_sem_loc = dataset_location + '//training//gt_image_2//'

# print(train_img_loc)
# print(train_sem_loc)

image_list = sorted([image for image in os.listdir(train_img_loc)])
sem_img_list = sorted([semanted for semanted in os.listdir(train_sem_loc)])

# print(image_list)
# print(sem_img_list)

final_image_list = []
final_semanted_image_list = []

for img, sem in zip(image_list, sem_img_list):
    # print('Hello')
    image_read=cv2.imread(train_img_loc+str(img))
    image_read = cv2.resize(image_read ,(160, 576))
    # print(img)
    # print(image_read)
    semanted_read=cv2.imread(train_sem_loc+str(sem))
    semanted_read = cv2.resize(semanted_read ,(160, 576))
    final_image_list.append(image_read)
    final_semanted_image_list.append(semanted_read)

final_image_list = np.array(final_image_list)
final_semanted_image_list = np.array(final_semanted_image_list)
# print(image_list)
np.save('image_array.npy', final_image_list, allow_pickle=True)
np.save('semanted_array.npy', final_semanted_image_list, allow_pickle=True)
    
data = np.load('image_array.npy', allow_pickle=True)
print(data)

