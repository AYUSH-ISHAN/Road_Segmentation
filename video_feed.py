'''
Import this video_feed.py to the run.py in the training=False part. 
To detect in the video part.

Predicting via video feed part.

Make this code to choose between video feed and real time feed
'''
from glob import glob
import cv2
import scipy
import numpy as np
import shutil
import tensorflow as tf
import csv
import time
import os
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras import regularizers, optimizers

MODEL_PATH = './model/model.ckpt'
CLASSES = 2

def loading_pre_VGG(session, path_2_VGG):

    tf.compat.v1.saved_model.loader.load(session, ['vgg16'], path_2_VGG)
    image_input = tf.compat.v1.get_default_graph().get_tensor_by_name('image_input:0')
    keep_prob = tf.compat.v1.get_default_graph().get_tensor_by_name('keep_prob:0')
    output_layer_3 = tf.compat.v1.get_default_graph().get_tensor_by_name('layer3_out:0')
    output_layer_4 = tf.compat.v1.get_default_graph().get_tensor_by_name('layer4_out:0')
    output_layer_7 = tf.compat.v1.get_default_graph().get_tensor_by_name('layer7_out:0')

    return image_input, keep_prob, output_layer_3, output_layer_4, output_layer_7

def architecture(output_layer_3, output_layer_4, output_layer_7, classes = CLASSES):
  
    '''Try to introduce the l2_regularization layer'''

    vgg_layer7_logits = Conv2D(
        classes, kernel_size=1,
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer= regularizers.l2(1e-4), name='vgg_layer7_logits')(output_layer_7)
    vgg_layer4_logits = Conv2D(
        classes, kernel_size=1,
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer= regularizers.l2(1e-4), name='vgg_layer4_logits')(output_layer_4)
    vgg_layer3_logits = Conv2D(
        classes, kernel_size=1,
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer= regularizers.l2(1e-4), name='vgg_layer3_logits')(output_layer_3)
    fcn_decoder_layer1 = Conv2DTranspose(
        classes, kernel_size=4, strides=(2, 2),
        padding='same',
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer= regularizers.l2(1e-4), name='fcn_decoder_layer1')(vgg_layer7_logits)
    fcn_decoder_layer2 = tf.add(
        fcn_decoder_layer1, vgg_layer4_logits, name='fcn_decoder_layer2')

    fcn_decoder_layer3 = Conv2DTranspose(
        classes, kernel_size=4, strides=(2, 2),
        padding='same',
        kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
        kernel_regularizer= regularizers.l2(1e-4), name='fcn_decoder_layer3')(fcn_decoder_layer2)

    fcn_decoder_layer4 = tf.add(
        fcn_decoder_layer3, vgg_layer3_logits, name='fcn_decoder_layer4')
    fcn_decoder_output = Conv2DTranspose(
        classes, kernel_size=16, strides=(8, 8),
        padding='same',
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer= regularizers.l2(1e-4), name='fcn_decoder_layer4')(fcn_decoder_layer4)

    return fcn_decoder_output

def guessing_or_predicting(frame, print_speed=False):
    num_classes = 2
    image_shape = (160, 576)
    runs_dir = './runs'

    # Path to vgg model
    vgg_path = os.path.join('./data', 'vgg')

    with tf.compat.v1.Session() as sess:
        # Predict the logits
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = loading_pre_VGG(sess, vgg_path)
        nn_last_layer = architecture(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits = tf.reshape(nn_last_layer, (-1, num_classes))

        # Restore the saved model
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, MODEL_PATH)
        print("Restored the saved Model in file: %s" % MODEL_PATH)

        # Predict the samples
        predictometer(runs_dir, frame, sess, image_shape, logits, keep_prob, input_image, print_speed)

def output_generator(sess, logits, keep_prob, image_pl, frame, image_shape):

        #for file2image in glob(os.path.join(data_folder, '*.png')):
        
            image = scipy.misc.imresize(frame, image_shape)
            tic = time.time()
            image_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_pl: [image]})  # keep_prob = 1.0 initially.
            
            image_softmax = image_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            
            segmentation = (image_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
            
            masked_street = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            mask = scipy.misc.toimage(masked_street, mode="RGBA")
            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)
            
            print("Command change !")
           
            toc = time.time()
            flash = 1.0 / (toc-tic)

            yield np.array(street_im), flash
            
def predictometer(runs_dir, frame, sess, image_shape, logits, keep_prob, input_image, print_speed=False):
    
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print('Predicting images...')

    image_outputs = output_generator(
        sess, logits, keep_prob, input_image, frame, image_shape)

    counter = 0
    for image, speed_ in image_outputs:
        
        #scipy.misc.imsave(os.path.join(output_dir, name), image)
        out.write(image)
        if print_speed is True:
        	
               counter+=1
               print("Processing file: {0:05d},\tSpeed: {1:.2f} fps".format(counter, speed_))

        # sum_time += laptime

    # pngCounter = len(glob1(data_dir,'*.png'))

    print('All augmented images are saved to: {}.'.format(output_dir))

if __name__ == "__main__":

    video_feed = True

    '''
    Image are being saved in the predictometer function
    '''

    #video_capture = cv2.VideoCapture(0)
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (600, 600)) # ot keep it of the height and width of images.
    if video_feed:

        cap = cv2.VideoCapture('video.mp4')
        #cap = cv2.VideoCapture('Video Of Travel.mp4')

        if cap.isOpened() == False:
            print('Error opening video stream or file')

        while (cap.isOpened()):

            ret, frame = cap.read()
            if ret == True:
                guessing_or_predicting(frame, print_speed=True)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xff == ord('q'):
                    break
            else:
                break

cap.release()
out.release()
cv2.destroyAllWindows()
        







