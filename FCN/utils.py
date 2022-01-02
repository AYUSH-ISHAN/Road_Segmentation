'''
Run the code after downloading the VGG-16 weights and placing it in ./data folder
'''
import random
import os
from glob import glob
import re  # a python module to search the files 
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import shutil
import time
import scipy
from PIL import Image

def Batch_Generator_Function(data_folder, img_shape):
    #print(data_folder)
    def batches_generator(Batch_Size, shuffle=True):
        
        '''Taking the folders out from the location of images'''
        #print(data_folder)
        path_2_img = glob(os.path.join(data_folder, 'image_2', '*.png'))
        path_2_sem_img = {
            re.sub(r'_(lane|road_)', '_', os.path.basename(destin)) : destin 
            for destin in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        #print(path_2_img)
        #print(path_2_sem_img)
        bg_color = np.array([255,0,0])
        #cv2.imshow('frame', bg_color)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        if shuffle:
            random.shuffle(path_2_img)

        for batch in range(0, len(path_2_img), Batch_Size):
            image_file = []
            semanted_file = []

            for image in path_2_img[batch : batch + Batch_Size]:
                #print(path_2_sem_img['uu_000078.png'])
                # semanted_image = path_2_sem_img[os.path.basename(image)]
                # image = Image.open(image).resize(img_shape)
                #semanted_img = Image.open(semanted_image).resize(img_shape)
                image = cv2.imread(image, cv2.IMREAD_COLOR)
                image = cv2.resize(image, img_shape)
                semanted_img = cv2.imread(semanted_image, cv2.IMREAD_COLOR)
                semanted_img = cv2.resize(semanted_img, img_shape)
                #print(semanted_img)
                #print(semanted_img.shape)
                semanted_img = np.array(semanted_img)
                semanted_bg = np.all(semanted_img == bg_color, axis=2)
                semanted_bg = semanted_bg.reshape(*semanted_bg.shape, 1)
                semanted_image = np.concatenate((semanted_bg, np.invert(semanted_bg)), axis=2)
                #cv2.imshow('frmae_1', semanted_bg)
                #cv2.imshow('frame', semanted_image)
                #cv2.waitKey(0)
                print(semanted_image.shape)
                #for i in semanted_image:
                #	print(i)
                #image = np.array(image)
                #cv2.imshow('frame', semanted_img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                image_file.append(image)
                semanted_file.append(semanted_image)

            #yield np.array(image_file), np.array(semanted_file)
            yield image_file, semanted_file
    return batches_generator

def test_generator(sess, logits, keep_prob, image_pl, data_folder, image_shape):
   
    '''Be careful with this part and 
    also take a look at it'''
    
    for file2image in glob(os.path.join(data_folder, 'image_2', '*.png')):
        #img = cv2.resize(cv2.imread(file2image), image_shape)
        #img = Image.open(file2image).resize(image_shape)
        #scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        img = cv2.imread(file2image, cv2.IMREAD_COLOR)
        img = cv2.resize(img, image_shape)
        image_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [img]})
        image_softmax = image_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (image_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        masked_street = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        
        # masked_street = Image.fromarray((masked_street).astype(np.uint8))
        # semanted_street = Image.fromarray((img).astype(np.uint8))

        semanted_street = cv2.add(semanted_street, masked_street)
        semanted_street.paste(masked_street, box = None, mask=masked_street)
        
        #print(masked_street)
        #masked_street = cv2.cvtColor(masked_street, cv2.COLOR_RGBA2RGBA)   # check this line out
        #semanted_street = cv2.bitwise_and(semanted_street, semanted_street, mask=masked_street)
        
        yield os.path.basename(file2image), semanted_street#, np.array(semanted_street)

def saving_test_images(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to disk
    print('Training Finished!')
    print('Saving test images to: {}, please wait...'.format(output_dir))
    image_outputs = test_generator(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        
        
        cv2.imwrite(os.path.join(output_dir, name), image)
        
        # img = Image.fromarray(image)
        # img.save(os.path.join(output_dir, name))

    print('All augmented images are saved!')

def output_generator(sess, logits, keep_prob, image_pl, data_folder, image_shape):

        for file2image in glob(os.path.join(data_folder, '*.png')):
            #img = cv2.resize(cv2.imread(file2image), image_shape)
            # img = Image.open(file2image).resize(image_shape)
            # img = np.array(img)

            img = cv2.imread(file2image, cv2.IMREAD_COLOR)
            semanted_street = cv2.resize(img, image_shape)

            tic = time.time()
            image_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_pl: [img]})  # keep_prob = 1.0 initially.
            #print("Hello",image_softmax[0])
            #demo_img = image_softmax
            image_softmax = image_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            #print("Hi",image_softmax[0][:,0])
            #print(len(image_softmax[0][:,0]))
            #print(image_softmax[0][:,1])
            #print(len(image_softmax[0][:,1]))
            #new_img_soft = []
            #for img in image_softmax[0][:,0]:
            #	new_img_soft.append(img)
            #for img in image_softmax[0][:,1]:
            #	new_img_soft.append(img)
            #image_softmax = image_softmax[0].reshape(image_shape[0], image_shape[1])
            #demo_img = image_softmax
            
            segmentation = (image_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
            
            #segmentation = (np.array(new_img_soft)> 0.5).reshape(image_shape[0], image_shape[1], 2)
            
            #demo_img = segmentation
            masked_street = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            
            # masked_street = Image.fromarray((masked_street).astype(np.uint8))
            # semanted_street = Image.fromarray((img).astype(np.uint8))
    
            #semanted_street = np.append(semanted_street, axis=1)
            #print(masked_street)
            print("Command change !")
            #print(semanted_street)
            #demo_img = semanted_street
            #semanted_street.paste(masked_street, box=None, mask=masked_street)
            semanted_street.paste(masked_street, box = None, mask=masked_street)
            #masked_street = cv2.cvtColor(masked_street, cv2.COLOR_RGBA2RGB)   # check this line out
            #  add another axis here.. Otr check the format of used in older code.
            #semanted_street = cv2.bitwise_and(semanted_street, semanted_street, mask=masked_street)
            toc = time.time()
            flash = 1.0 / (toc-tic)

            yield os.path.basename(file2image), np.array(semanted_street), flash, np.array(masked_street)#, np.array(demo_img)

            yield os.path.basename(file2image), semanted_street, flash, np.array(masked_street)#, np.array(demo_img)

def predictometer(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, print_speed=False):
    
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print('Predicting images...')

    image_outputs = output_generator(
        sess, logits, keep_prob, input_image, data_dir, image_shape)

    counter = 0
    for name, image, speed_, masked_street in image_outputs:
        #print(image)
        
        image = np.array(image)
        cv2.imwrite(os.path.join(output_dir, name), image)



        # img = Image.fromarray(image)
        # img.save(os.path.join(output_dir, name))


        #cv2.imwrite(os.path.join(output_dir, name, str('Hello')), masked_street)
        cv2.imshow('frame', masked_street)
        #print(demo_img)
        #print(demo_img.shape)
        #cv2.imshow('frame_1', demo_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if print_speed is True:
            counter+=1
            print("Processing file: {0:05d},\tSpeed: {1:.2f} fps".format(counter, speed_))

    print('All augmented images are saved to: {}.'.format(output_dir))




