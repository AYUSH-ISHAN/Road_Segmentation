
############################################

###    MODEL IS TRAINING ...nOW, WORK ON PREDICTION PART.
'''
trained weights are there in model_h5 unets..
'''

'''
THINGS TO DO: 

1. Try to train the model on its original dimensions.
2. See the outputs 
'''



import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import datetime
import time
from Unets_model import unet
from IPython.display import clear_output
#from Unets_model import UNETS
from tensorflow.keras.optimizers import Adam, RMSprop

# check by older epoch and batch size.
EPOCHS = 20 # 2000
BATCH_SIZE = 8# 32
INPUT_SIZE = (400, 400, 3)#(1242, 375, 3) # 
CLASSES = 2  # # num_classes = 2 for road or not road. 
IMG_HEIGHT, IMG_WIDTH = 400, 400#1242, 375  # 400, 400
AUTOTUNE = tf.data.experimental.AUTOTUNE

#model = unet(INPUT_SIZE, 3) # size and other parameters are already in arguements (Work with this 
#classes = 3)
model = unet()
#model = UNETS(INPUT_SIZE, 2)
model.summary()
#model.compile(loss="categorical_crossentropy", optimizer=Adam(lr), metric=['accuracy'])

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

list_ds_train = tf.data.Dataset.list_files("./data_road/training/image_2/*", shuffle=False)
list_ds_train_label = tf.data.Dataset.list_files("./data_road/training/gt_image_2/*", shuffle=False)

# train_X = np.load('image_array.npy', allow_pickle=True)   # This is the input image
# train_Y = np.load('semanted_array.npy', allow_pickle=True)   # This is the semanted image

# X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=10)
train_X = list_ds_train.map(process_path, num_parallel_calls=AUTOTUNE)
train_Y = list_ds_train_label.map(get_label, num_parallel_calls=AUTOTUNE)
train_dataset = tf.data.Dataset.zip((train_X, train_Y))

list_ds_test = tf.data.Dataset.list_files("./data_road/testing/image_2/*", shuffle=False)
list_ds_test_label = tf.data.Dataset.list_files("./data_road/testing/calib/*", shuffle=False)

# Set `num_parallel__calls` so multiple images are loaded/processed in parallel.
test_X = list_ds_test.map(process_path, num_parallel_calls=AUTOTUNE)
test_Y = list_ds_test_label.map(get_label, num_parallel_calls=AUTOTUNE)

test_dataset = tf.data.Dataset.zip((test_X, test_Y))

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
    except RuntimeError as e:
        print(e)


optimizer = tf.keras.optimizers.Adam(1e-4)

loss_object = tf.keras.losses.CategoricalCrossentropy()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
train_mIoU =tf.keras.metrics.MeanIoU(num_classes = CLASSES, name = "train_mIoU") 
test_mIoU =tf.keras.metrics.MeanIoU(num_classes = CLASSES, name = "test_mIoU")
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/unets_log/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/unets_log/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

@tf.function
def train_step(images, label):
    print("Hello !")
    with tf.GradientTape() as tape:
        images = tf.convert_to_tensor(images)
        CLASSE = tf.convert_to_tensor(CLASSES)
        pred_img = model(images, CLASSE)        
        loss =  loss_object(label, pred_img)

    gradients_of_model = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(label, pred_img)
 
@tf.function
def test_step(images, label):
    print("Hello !")
    images = tf.convert_to_tensor(images)
    CLASSE = tf.convert_to_tensor(CLASSES)
    pred_img = model(images, CLASSE)
    loss =  loss_object(label, pred_img)
    
    #pred_mask = tf.argmax(pred_img, axis=-1)
    #pred_mask = pred_mask[..., tf.newaxis]
    
    #test_mIoU(label, pred_mask)
    test_loss(loss)
    test_accuracy(label, pred_img)


def train(train_dataset, test_dataset, epochs, batch_size):
    
    for epoch in range(epochs):
        
        if epoch >=10:
            optimizer = tf.keras.optimizers.Adam(1e-5)

        
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()    
        
        test_loss.reset_states()
        test_accuracy.reset_states()
                
        count_img = 0
        batch_time = time.time()
        
        for image_batch, semanted_batch in train_dataset.batch(batch_size):
            count_img += batch_size
            label_batch = semanted_batch #convert_class(label_batch.numpy())
            
            if tf.random.uniform(()) > 0.5:
                image_batch = tf.image.flip_left_right(image_batch)
                label_batch = tf.image.flip_left_right(label_batch)

            image_batch = tf.image.random_brightness(image_batch, 0.3)
            #label_batch = tf.image.random_brightness(label_batch, 0.3)
            train_step(image_batch, label_batch)
            
            if count_img % 1000 == 0:
                clear_output(wait=True)
                # show_predictions(image_batch[:3], label_batch[:3], model)      
                print('epoch {}, step {}, train_acc {}, loss {} , time {}'.format(epoch+1,
                                                                        count_img,
                                                                        train_accuracy.result()*100,
                                                                        train_loss.result(),
                                                                        time.time()- batch_time))
                train_loss.reset_states()
                train_accuracy.reset_states()    
                
                batch_time = time.time()
                
                
        count_img = 0
        batch_time = time.time()

        for image_batch, semanted_batch in test_dataset.batch(batch_size):
            count_img += batch_size
            label_batch = semanted_batch#convert_class(label_batch.numpy())
            
            test_step(image_batch, label_batch)
            
            
            if count_img % 1000 == 0:
                clear_output(wait=True)
                # show_predictions(image_batch[:3], label_batch[:3], model)
                print('epoch {}, step {}, test_acc {}, loss {} , time {}'.format(epoch+1,
                                                                        count_img,
                                                                        test_accuracy.result()*100,
                                                                        test_loss.result(),
                                                                        time.time()- batch_time))
                batch_time = time.time()
                

                
        clear_output(wait=True)

        for image_batch, semanted_batch in test_dataset.take(3).batch(3):
            label_batch = semanted_batch#convert_class(label_batch.numpy())
            # show_predictions(image_batch[:3], label_batch[:3], model)        

        print ('Time for epoch {}  is {} sec'.format(epoch + 1, round(time.time()-start),3))

        print ('train_acc {}, loss {} , test_acc {}, loss {}'.format(train_accuracy.result()*100,
                                                                                 train_loss.result(),
                                                                                 test_accuracy.result()*100,
                                                                                 test_loss.result()
                                                                                 ))
        
        path = "model_h5/unets/unets_" + str(test_loss.result().numpy())+"_epoch_"+str(epoch+1)+".h5" 
        model.save(path)
            
        with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch+1)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch+1)             
                #tf.summary.scalar('mIoU', train_mIoU.result(), step=epoch)    
                
        with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch+1)
                tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch+1)    
                #tf.summary.scalar('mIoU', test_mIoU.result(), step=epoch)           
train(train_dataset, test_dataset, 20, 8)



model.summary()




test_loss.reset_states()
test_accuracy.reset_states()
test_mIoU.reset_states()
#model = UNETS(INPUT_SIZE, CLASSES)
model.load_weights("model_h5/unet/unet_0.35704005_epoch_9.h5")
count_img = 0
batch_time = time.time()

for image_batch, semanted_batch in test_dataset.batch(8):
    count_img += 8
    label_batch = semanted_batch#convert_class(label_batch.numpy())
    
    test_step(image_batch, label_batch)
            
    if count_img % 1000 == 0:
        clear_output(wait=True)
        # show_predictions(image_batch[:3], label_batch[:3], model)
        print('epoch {}, step {}, test_acc {}, loss {} ,mIoU {}, time {}'.format(1,
                                                                count_img,
                                                                test_accuracy.result()*100,
                                                                test_loss.result(),
                                                                test_mIoU.result(),
                                                                time.time()- batch_time))
        batch_time = time.time()

print('epoch {}, step {}, test_acc {}, loss {} ,mIoU {}, time {}'.format(1,
                                                                count_img,
                                                                test_accuracy.result()*100,
                                                                test_loss.result(),
                                                                test_mIoU.result(),
                                                                time.time()- batch_time))                            


print(count_img)
