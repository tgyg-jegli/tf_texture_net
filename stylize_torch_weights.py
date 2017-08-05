import time

import numpy as np
import tensorflow as tf

import lutorpy as lua
from utils import *

# Import of Lua/Torch Modules
require("torch")
require("nn")
require("lua_modules/TVLoss")
require("lua_modules/InstanceNormalization")


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

    # Load Torch model
    lua_model = torch.load('./model/model.t7')

    ops = {}

    image = tf.placeholder(tf.float32, shape=None, name='image-placeholder')

    with tf.name_scope('preprocessing'):
        ops['preprocessing'] = tf.div(image, 255)
        ops['preprocessing'] = tf.expand_dims(ops['preprocessing'], 0)

    with tf.name_scope('pad_2'):
        paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]
        ops['pad_2'] = tf.pad(ops['preprocessing'], paddings, "SYMMETRIC")
        ops['pad_2'] = tf.pad(ops['pad_2'], paddings, "SYMMETRIC")
        ops['pad_2'] = tf.pad(ops['pad_2'], paddings, "SYMMETRIC")
        ops['pad_2'] = tf.pad(ops['pad_2'], paddings, "SYMMETRIC")

    with tf.name_scope('conv_3'):
        weights = np.transpose(lua_model.modules[2].weight.asNumpyArray(), (2, 3, 1, 0))
        filter = tf.Variable(weights, name='filter')
        strides = [1, 1, 1, 1]
        ops['conv_3'] = tf.nn.conv2d(ops['pad_2'], filter, strides, padding='VALID', use_cudnn_on_gpu=None, data_format=None, name=None)

    with tf.name_scope('norm_4'):
        scale = tf.Variable(lua_model.modules[3].weight.asNumpyArray(), name='scale')
        offset = tf.Variable(lua_model.modules[3].bias.asNumpyArray(), name='offset')
        epsilon = 1e-5
        mean, var = tf.nn.moments(ops['conv_3'], [1, 2], keep_dims=True)
        ops['norm_4'] = tf.nn.batch_normalization(ops['conv_3'], mean, var, offset, scale, epsilon)

    with tf.name_scope('relu_5'):
        ops['relu_5'] = tf.nn.relu(ops['norm_4'])

    with tf.name_scope('conv_6'):
        weights = np.transpose(lua_model.modules[5].weight.asNumpyArray(), (2, 3, 1, 0))
        filter = tf.Variable(weights, name='filter')
        strides = [1, 2, 2, 1]
        ops['conv_6'] = tf.nn.conv2d(ops['relu_5'], filter, strides, padding='VALID', use_cudnn_on_gpu=None, data_format=None, name=None)

    with tf.name_scope('norm_7'):
        scale = tf.Variable(lua_model.modules[6].weight.asNumpyArray(), name='scale')
        offset = tf.Variable(lua_model.modules[6].bias.asNumpyArray(), name='offset')
        epsilon = 1e-5
        mean, var = tf.nn.moments(ops['conv_6'], [1, 2], keep_dims=True)
        ops['norm_7'] = tf.nn.batch_normalization(ops['conv_6'], mean, var, offset, scale, epsilon)

    with tf.name_scope('relu_8'):
        ops['relu_8'] = tf.nn.relu(ops['norm_7'])

    with tf.name_scope('conv_9'):
        weights = np.transpose(lua_model.modules[8].weight.asNumpyArray(), (2, 3, 1, 0))
        filter = tf.Variable(weights, name='filter')
        strides = [1, 2, 2, 1]
        ops['conv_9'] = tf.nn.conv2d(ops['relu_8'], filter, strides, padding='VALID', use_cudnn_on_gpu=None, data_format=None, name=None)

    with tf.name_scope('norm_10'):
        scale = tf.Variable(lua_model.modules[9].weight.asNumpyArray(), name='scale')
        offset = tf.Variable(lua_model.modules[9].bias.asNumpyArray(), name='offset')
        epsilon = 1e-5
        mean, var = tf.nn.moments(ops['conv_9'], [1, 2], keep_dims=True)
        ops['norm_10'] = tf.nn.batch_normalization(ops['conv_9'], mean, var, offset, scale, epsilon)

    with tf.name_scope('relu_11'):
        ops['relu_11'] = tf.nn.relu(ops['norm_10'])

    ops['res_block_11'] = ops['relu_11']
    for i in range(12, 17):
        with tf.name_scope('res_block_' + str(i)):
            ops['res_block_' + str(i)] = res_block(ops['res_block_' + str(i-1)], (i-1), lua_model)

    with tf.name_scope('conv_transpose_17'):
        weights = np.transpose(lua_model.modules[16].weight.asNumpyArray(), (2, 3, 1, 0))
        filter = tf.Variable(weights, name='filter')
        strides = [1, 2, 2, 1]
        shape = tf.shape(ops['conv_6'])
        outputshape = tf.stack([shape[0], shape[1], shape[2], shape[3]])
        ops['conv_17'] = tf.nn.conv2d_transpose(ops['res_block_16'], filter, outputshape, strides, padding='VALID', name=None)

    with tf.name_scope('norm_18'):
        scale = tf.Variable(lua_model.modules[17].weight.asNumpyArray(), name='scale')
        offset = tf.Variable(lua_model.modules[17].bias.asNumpyArray(), name='offset')
        epsilon = 1e-5
        mean, var = tf.nn.moments(ops['conv_17'], [1, 2], keep_dims=True)
        ops['norm_18'] = tf.nn.batch_normalization(ops['conv_17'], mean, var, offset, scale, epsilon)

    with tf.name_scope('relu_19'):
        ops['relu_19'] = tf.nn.relu(ops['norm_18'])

    with tf.name_scope('conv_transpose_20'):
        weights = np.transpose(lua_model.modules[19].weight.asNumpyArray(), (2, 3, 1, 0))
        filter = tf.Variable(weights, name='filter')
        strides = [1, 2, 2, 1]
        shape = tf.shape(ops['conv_3'])
        outputshape = tf.stack([shape[0], shape[1], shape[2], shape[3]])
        ops['conv_20'] = tf.nn.conv2d_transpose(ops['relu_19'], filter, outputshape, strides, padding='VALID', name=None)

    with tf.name_scope('norm_21'):
        scale = tf.Variable(lua_model.modules[20].weight.asNumpyArray(), name='scale')
        offset = tf.Variable(lua_model.modules[20].bias.asNumpyArray(), name='offset')
        epsilon = 1e-5
        mean, var = tf.nn.moments(ops['conv_20'], [1, 2], keep_dims=True)
        ops['norm_21'] = tf.nn.batch_normalization(ops['conv_20'], mean, var, offset, scale, epsilon)

    with tf.name_scope('relu_22'):
        ops['relu_22'] = tf.nn.relu(ops['norm_21'])

    with tf.name_scope('pad_23'):
        paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]
        ops['pad_23'] = tf.pad(ops['relu_22'], paddings, "SYMMETRIC")

    with tf.name_scope('conv_24'):
        weights = np.transpose(lua_model.modules[23].weight.asNumpyArray(), (2, 3, 1, 0))
        filter = tf.Variable(weights, name='filter')
        strides = [1, 1, 1, 1]
        ops['conv_24'] = tf.nn.conv2d(ops['pad_23'], filter, strides, padding='VALID', use_cudnn_on_gpu=None, data_format=None, name=None)

    with tf.name_scope('deprocessing'):
        ops['squeeze'] = tf.squeeze(ops['conv_24'])
        vgg_mean_0 = tf.constant(103.939)
        vgg_mean_1 = tf.constant(116.779)
        vgg_mean_2 = tf.constant(123.68)
        red, green, blue = tf.split(ops['squeeze'], num_or_size_splits=3, axis=2)
        ops['bgr'] = tf.concat([blue + vgg_mean_2, green + vgg_mean_1, red + vgg_mean_0], 2)

    # TensorBoard output
    tf.summary.FileWriter("./tb/", tf.get_default_graph()).close()

    # Init session
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Save session
    saver = tf.train.Saver()
    saver.save(sess, 'model/texture_net.chkp')

    # Run session
    output = sess.run(ops['bgr'], feed_dict={image: input_image})
    sess.close()

    return output

def res_block(input, index, lua_model):

    res_block_ops = {}

    with tf.name_scope('pad_1'):
        paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]
        res_block_ops['pad_1'] = tf.pad(input, paddings, "SYMMETRIC");

    with tf.name_scope('conv_2'):
        weights = np.transpose(lua_model.modules[index].modules[0].modules[1].modules[1].weight.asNumpyArray(), (2, 3, 1, 0))
        filter = tf.Variable(weights, name='filter')
        strides = [1, 1, 1, 1]
        res_block_ops['conv_2'] = tf.nn.conv2d(res_block_ops['pad_1'], filter, strides, padding='VALID', use_cudnn_on_gpu=None, data_format=None, name=None)

    with tf.name_scope('norm_3'):
        scale = tf.Variable(lua_model.modules[index].modules[0].modules[1].modules[2].weight.asNumpyArray(), name='scale')
        offset = tf.Variable(lua_model.modules[index].modules[0].modules[1].modules[2].bias.asNumpyArray(), name='offset')
        epsilon = 1e-5
        mean, var = tf.nn.moments(res_block_ops['conv_2'], [1, 2], keep_dims=True)
        res_block_ops['norm_3'] = tf.nn.batch_normalization(res_block_ops['conv_2'], mean, var, offset, scale, epsilon)


    with tf.name_scope('relu_4'):
        res_block_ops['relu_4'] = tf.nn.relu(res_block_ops['norm_3'])

    with tf.name_scope('pad_5'):
        paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]
        res_block_ops['pad_5'] = tf.pad(res_block_ops['relu_4'], paddings, "SYMMETRIC");

    with tf.name_scope('conv_6'):
        weights = np.transpose(lua_model.modules[index].modules[0].modules[1].modules[5].weight.asNumpyArray(), (2, 3, 1, 0))
        filter = tf.Variable(weights, name='filter')
        res_block_ops['conv_6'] = tf.nn.conv2d(res_block_ops['pad_5'], filter, strides, padding='VALID', use_cudnn_on_gpu=None, data_format=None, name=None)

    with tf.name_scope('norm_7'):
        scale = tf.Variable(lua_model.modules[index].modules[0].modules[1].modules[6].weight.asNumpyArray(), name='scale')
        offset = tf.Variable(lua_model.modules[index].modules[0].modules[1].modules[6].bias.asNumpyArray(), name='offset')
        epsilon = 1e-5
        mean, var = tf.nn.moments(res_block_ops['conv_6'], [1, 2], keep_dims=True)
        res_block_ops['norm_7'] = tf.nn.batch_normalization(res_block_ops['conv_6'], mean, var, offset, scale, epsilon)


    with tf.name_scope('add_8'):
        res_block_ops['sum'] = tf.add_n([res_block_ops['norm_7'], input])

    return res_block_ops['sum']


if __name__ == '__main__':
    main()
