'''
Rename it as testing.py 


Code is working but output is not satisfactory.

'''

from tensorflow.keras.models import load_model
from Unets_model import unet
import time
import tensorflow as tf
import datetime
from IPython.display import clear_output
import cv2
import matplotlib.pyplot as plt

model = unet()
model.load_weights("./model_h5/unet/unets_nan_epoch_20.h5")
INPUT_SIZE = (400, 400, 3)#(1242, 375, 3) # 
CLASSES = 2  # # num_classes = 2 for road or not road. 
IMG_HEIGHT, IMG_WIDTH = 400, 400#1242, 375  # 400, 400
AUTOTUNE = tf.data.experimental.AUTOTUNE
count_img = 0
batch_time = time.time()
def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    return img

def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img

def get_label(file_path):
    img = tf.io.read_file(file_path)
    label_img = tf.image.decode_jpeg(img, channels=3)
    label_img = tf.image.resize(label_img, [IMG_HEIGHT, IMG_WIDTH])
    return label_img

list_ds_test = tf.data.Dataset.list_files("./data_road/testing/image_2/*", shuffle=False)
#list_ds_test_label = tf.data.Dataset.list_files("./data_road/testing/calib/*", shuffle=False)

# Set `num_parallel__calls` so multiple images are loaded/processed in parallel.
test_X = list_ds_test.map(process_path, num_parallel_calls=AUTOTUNE)
#test_Y = list_ds_test_label.map(get_label, num_parallel_calls=AUTOTUNE)

test_dataset = tf.data.Dataset.zip((test_X))#, test_Y))

@tf.function
def test_step(images):
    print("Hello !")
    images = tf.convert_to_tensor(images)
    CLASSE = tf.convert_to_tensor(CLASSES)
    pred_img = model(images, CLASSE)

    return pred_img
    
    #pred_mask = tf.argmax(pred_img, axis=-1)
    #pred_mask = pred_mask[..., tf.newaxis]
    
    #test_mIoU(label, pred_mask)
    

for image_batch in test_dataset.batch(8):
    count_img += 8
   # label_batch = semanted_batch#convert_class(label_batch.numpy())
    
    output = test_step(image_batch)

    plt.imshow(output[0])
   
    plt.show()
