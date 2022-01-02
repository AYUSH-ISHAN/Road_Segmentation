from tensorflow.keras.models import load_model
from Unets_model_1 import unet
import time
import tensorflow as tf
import datetime
from IPython.display import clear_output

model = unet()
model.load_weights("model_h5/unet/unets_nan_epoch_20.h5")
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
