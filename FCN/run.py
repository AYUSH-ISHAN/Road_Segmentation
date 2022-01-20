
import tensorflow as tf
import csv
import time
import utils
import os
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras import regularizers, optimizers


'''
Change it to Tensorflow 2 mode (specially the architecture function part)
and then after this do some more editing.
'''


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

'''''
Covert it to TENSORFLOW VERSION :---->   2
'''


def architecture(output_layer_3, output_layer_4, output_layer_7, classes = CLASSES):

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


def optimizometmer(Neural_last_layer, correct_label, lr, classes = CLASSES):
    
    logits = tf.reshape(Neural_last_layer, (-1, classes))
    correct_label = tf.reshape(correct_label, (-1,classes))
    
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label))
    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate= lr)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

###   To edit after this>

def training(sess, epochs, batch_size, batches_generator, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    
    # Create log file
    log_filename = "./training_progress.csv"
    log_fields = ['learning_rate', 'exec_time (s)', 'training_loss']
    log_file = open(log_filename, 'w')
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()


    sess.run(tf.compat.v1.global_variables_initializer())

    l_rate = 0.0001

    print("Training...")
    print()
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        training_loss = 0
        training_samples = 0
        starttime = time.time()
        for image, label in batches_generator(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label,
                                          keep_prob: 0.8, learning_rate: l_rate})
            print("batch loss: = {:.3f}".format(loss))
            training_samples += 1
            training_loss += loss

        training_loss /= training_samples
        endtime = time.time()
        training_time = endtime-starttime

        print("Average loss for the current epoch: = {:.3f}\n".format(training_loss))
        log_writer.writerow({'learning_rate': l_rate, 'exec_time (s)': round(training_time, 2) , 'training_loss': round(training_loss,4)})
        log_file.flush()


def main_runner():
    num_classes = 2
    image_shape = (160, 576)
    DATA_LOCATION = './data'
    PREDICTION_LOCATION = './runs'
    '''
    Make sure to download the dataset and store in the './data'folder.. Otherwise it will 
    show error.
    '''
    with tf.compat.v1.Session() as sess:
        vgg_path = os.path.join(DATA_LOCATION, 'vgg')
        get_batches_fn = utils.Batch_Generator_Function(os.path.join(DATA_LOCATION, 'data_road/training'), image_shape)
        epochs = 30
        batch_size = 8
        correct_label = tf.compat.v1.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.compat.v1.placeholder(tf.float32, name='learning_rate')
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = loading_pre_VGG(sess, vgg_path)
        nn_last_layer = architecture(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimizometmer(nn_last_layer, correct_label, learning_rate, num_classes)
        training(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess, MODEL_PATH)
        print("Model is saved to file: %s" % save_path)
        utils.saving_test_images(PREDICTION_LOCATION , DATA_LOCATION, sess, image_shape, logits, keep_prob, input_image)



'''Edit after this'''

def guessing_or_predicting(test_data_path, print_speed=False):
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
        utils.predictometer(runs_dir, test_data_path, sess, image_shape, logits, keep_prob, input_image, print_speed)

if __name__ == '__main__':

    training_flag = True   # True: train the NN; False: predict with trained NN

    if training_flag:
        main_runner()
    else:
        # use the pre-trained model to predict more images
        TEST_DATA_PATH = './data/data_road/testing/image_2'
        guessing_or_predicting(TEST_DATA_PATH, print_speed=True)
