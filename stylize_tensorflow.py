import time

import numpy as np
import tensorflow as tf

from utils import *


def main():
    parser = build_parser()
    options = parser.parse_args()
    content_image = imread(options.input)

    start_time = time.time()

    stylized_image = network(content_image)

    stop_time = time.time()
    print('Duration: {0:.3f}'.format(stop_time - start_time))

    imsave(options.output, stylized_image)


def network(input_image):

    ops = {}

    image = tf.placeholder(tf.float32, shape=None, name='image-placeholder')

    with tf.name_scope('preprocessing'):
        ops['preprocessing'] = tf.div(image, 255)
        ops['preprocessing'] = tf.expand_dims( ops['preprocessing'], 0)

    with tf.name_scope('pad_2'):
        ops['pad_2'] = pad(ops['preprocessing'], 4)

    with tf.name_scope('conv_3'):
        ops['conv_3'] = conv(ops['pad_2'], [1, 1, 1, 1], [9, 9, 3, 32])

    with tf.name_scope('norm_4'):
        ops['norm_4'] = norm(ops['conv_3'], [32])

    with tf.name_scope('relu_5'):
        ops['relu_5'] = tf.nn.relu( ops['norm_4'])

    with tf.name_scope('conv_6'):
        ops['conv_6'] = conv(ops['relu_5'], [1, 2, 2, 1], [3, 3, 32, 64])

    with tf.name_scope('norm_7'):
        ops['norm_7'] = norm(ops['conv_6'], [64])

    with tf.name_scope('relu_8'):
        ops['relu_8'] = tf.nn.relu(ops['norm_7'])

    with tf.name_scope('conv_9'):
        ops['conv_9'] = conv(ops['relu_8'], [1, 2, 2, 1], [3, 3, 64, 128])

    with tf.name_scope('norm_10'):
        ops['norm_10'] = norm(ops['conv_9'], [128])

    with tf.name_scope('relu_11'):
        ops['relu_11'] = tf.nn.relu(ops['norm_10'])

    ops['res_block_11'] = ops['relu_11']
    for i in range(12, 17):
        with tf.name_scope('res_block_' + str(i)):
            ops['res_block_' + str(i)] = res_block(ops['res_block_' + str(i-1)])

    with tf.name_scope('conv_transpose_17'):
        ops['conv_transpose_17'] = conv_transpose(ops['res_block_16'], [1, 2, 2, 1], [3, 3, 64, 128], ops['conv_6'])

    with tf.name_scope('norm_18'):
        ops['norm_18'] = norm(ops['conv_transpose_17'], [64])

    with tf.name_scope('relu_19'):
        ops['relu_19'] = tf.nn.relu(ops['norm_18'])

    with tf.name_scope('conv_transpose_20'):
        ops['conv_transpose_20'] = conv_transpose(ops['relu_19'], [1, 2, 2, 1], [3, 3, 32, 64], ops['conv_3'])

    with tf.name_scope('norm_21'):
        ops['norm_21'] = norm(ops['conv_transpose_20'], [32])

    with tf.name_scope('relu_22'):
        ops['relu_22'] = tf.nn.relu(ops['norm_21'])

    with tf.name_scope('pad_23'):
        ops['pad_23'] = pad(ops['relu_22'], 1);

    with tf.name_scope('conv_24'):
        ops['conv_24'] = conv(ops['pad_23'], [1, 1, 1, 1], [3, 3, 32, 3])

    with tf.name_scope('deprocessing'):
        ops['squeeze'] = tf.squeeze(ops['conv_24'])
        vgg_mean_0 = tf.constant(103.939)
        vgg_mean_1 = tf.constant(116.779)
        vgg_mean_2 = tf.constant(123.68)
        red, green, blue = tf.split(2, 3, ops['squeeze'])
        ops['bgr'] = tf.concat(2, [blue + vgg_mean_2, green + vgg_mean_1, red + vgg_mean_0])

    # TensorBoard output
    tf.summary.FileWriter("./tb/", tf.get_default_graph()).close()

    # Run session
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, 'model/texture_net.chkp')
    output = sess.run(ops['bgr'], feed_dict={image: input_image})
    sess.close()

    return output


def res_block(input):
    res_block_ops = {}

    with tf.name_scope('pad_1'):
        res_block_ops['pad_1'] = pad(input, 1);

    with tf.name_scope('conv_2'):
        res_block_ops['conv_2'] = conv(res_block_ops['pad_1'], [1, 1, 1, 1], [3, 3, 128, 128])

    with tf.name_scope('norm_3'):
        res_block_ops['norm_3'] = norm(res_block_ops['conv_2'], [128])

    with tf.name_scope('relu_4'):
        res_block_ops['relu_4'] = tf.nn.relu(res_block_ops['norm_3'])

    with tf.name_scope('pad_5'):
        res_block_ops['pad_5'] = pad(res_block_ops['relu_4'], 1)

    with tf.name_scope('conv_6'):
        res_block_ops['conv_6'] = conv(res_block_ops['pad_5'], [1, 1, 1, 1], [3, 3, 128, 128])

    with tf.name_scope('norm_7'):
        res_block_ops['norm_7'] = norm(res_block_ops['conv_6'], [128])

    with tf.name_scope('add_8'):
        res_block_ops['sum'] = tf.add_n([res_block_ops['norm_7'], input])

    return res_block_ops['sum']

def conv(input, strides, shape_filter):
    filter = tf.Variable(tf.truncated_normal(shape_filter, stddev=0.1), name='filter')
    return tf.nn.conv2d(input, filter, strides, padding='VALID', use_cudnn_on_gpu=None)

def norm(input, shape_parameter):
    scale = tf.Variable(tf.truncated_normal(shape_parameter, stddev=0.1), name='scale')
    offset = tf.Variable(tf.truncated_normal(shape_parameter, stddev=0.1), name='offset')
    epsilon = 1e-5
    mean, var = tf.nn.moments(input, [1, 2], keep_dims=True)
    return tf.nn.batch_normalization(input, mean, var, offset, scale, epsilon)

def pad(input, number_repetition=1):
    single_paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]
    for i in range(number_repetition):
        input = tf.pad(input, single_paddings, "SYMMETRIC");
    return input

def conv_transpose(input, strides, shape_filter, corresponding_tensor):
    filter = tf.Variable(tf.truncated_normal(shape_filter, stddev=0.1), name='filter')
    shape = tf.shape(corresponding_tensor)
    outputshape = tf.pack([shape[0], shape[1], shape[2], shape[3]])
    return tf.nn.conv2d_transpose(input, filter, outputshape, strides, padding='VALID')



if __name__ == '__main__':
    main()
