#!/usr/bin/env python3
import os.path
import tensorflow as tf
import video_utils
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

CLOUD_MODE = False


def loading_pre_vgg(sess, vgg_path):
    
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.compat.v1.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    image_input = tf.compat.v1.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = tf.compat.v1.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_output = tf.compat.v1.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_output = tf.compat.v1.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_output = tf.compat.v1.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_output, layer4_output, layer7_output


def architecture(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    
    conv1x1 = Conv2D(num_classes, 1, padding='same',
                               kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                               kernel_regularizer=regularizers.l2(1e-2))(vgg_layer7_out)

    upsample_1 = Conv2DTranspose(num_classes, 4, strides=(2, 2), padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                            kernel_regularizer=regularizers.l2(1e-2))(conv1x1)

    vgg_layer4_reshape = Conv2D(num_classes, 1, padding='same',
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                      kernel_regularizer=regularizers.l2(1e-2))(vgg_layer4_out)

    skip_layer_1 = tf.add(upsample_1, vgg_layer4_reshape)

    upsample_2 = Conv2DTranspose(num_classes, 4, strides=(2, 2), padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                            kernel_regularizer=regularizers.l2(1e-2))(skip_layer_1)

    vgg_layer3_reshape = Conv2D(num_classes, 1, padding='same',
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                          kernel_regularizer=regularizers.l2(1e-2))(vgg_layer3_out)

    skip_layer_2 = tf.add(upsample_2, vgg_layer3_reshape)

    upsample_final = Conv2DTranspose(num_classes, 16, strides=(8, 8), padding='same',
                                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                kernel_regularizer=regularizers.l2(1e-2))(skip_layer_2)

    return upsample_final


def optimizometer(nn_last_layer, correct_label, learning_rate, num_classes):
    
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


def trainer(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):

    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()

    if CLOUD_MODE:
        model_dir = '/output'
    else:
        model_dir = os.getcwd() + '/checkpoints'

    print('Start Training...\n')
    for i in range(epochs):
        print('Epoch {} ...'.format(i + 1))
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.0009})
            print('Loss = {:.3f}'.format(loss))
        print()
    saver.save(sess, model_dir + '/model')
    print('model saved!')

def runner():
    num_classes = 2
    image_shape = (160, 576)
    if CLOUD_MODE:
        data_dir = '/input'
    else:
        data_dir = './data'

    with tf.compat.v1.Session() as sess:

        vgg_path = os.path.join(data_dir, 'vgg')
        get_batches_fn = video_utils.Batch_Generator_Function(os.path.join(data_dir, 'data_road/training'), image_shape)

        epochs = 50
        batch_size = 5

        correct_label = tf.compat.v1.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.compat.v1.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = loading_pre_vgg(sess, vgg_path)

        nn_last_layer = architecture(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimizometer(nn_last_layer, correct_label, learning_rate, num_classes)

        trainer(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)


if __name__ == '__main__':
    runner()

