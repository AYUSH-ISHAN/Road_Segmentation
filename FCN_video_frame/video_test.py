import tensorflow as tf
import os
import numpy as np
import cv2

CLOUD_MODE= False
SAVING = False

def test_video(sess, image_shape, logits, keep_prob, input_image):
    if CLOUD_MODE:
        cap = cv2.VideoCapture('./video.mp4')
        video_writer = cv2.VideoWriter('./video.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                       (576, 160))
    else:
        cap = cv2.VideoCapture('./Video_Of_Travel.mp4')

    while (cap.isOpened()):
        ret, raw_img = cap.read()

        # raw_img = cv2.imread(image_file)
        raw_img = cv2.resize(raw_img, (576, 160))
        image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, input_image: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0]])).astype(np.uint8)

        result = cv2.addWeighted(raw_img, 1, mask, 0.5, 0)

        if SAVING:
            video_writer.write(result)
            print('write frame finished!')
        else:
            cv2.imshow('res', result)
            cv2.waitKey(1)

if CLOUD_MODE:
    model_dir = '/road_model'
else:
    model_dir = os.getcwd() + '/checkpoints'

def main():
    num_classes = 2
    image_shape = (160, 576)
    tf.compat.v1.disable_eager_execution()
    saver = tf.compat.v1.train.import_meta_graph(model_dir + '/model.meta')
    graph = tf.compat.v1.get_default_graph()

    logits = graph.get_operation_by_name('logits').outputs[0]
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    input_image = graph.get_tensor_by_name('image_input:0')


    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, model_dir + '/model')

        test_video(sess, image_shape, logits, keep_prob, input_image)

if __name__ == '__main__':
    main()
