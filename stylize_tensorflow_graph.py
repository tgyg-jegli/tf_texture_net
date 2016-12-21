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
    print('Duration: {0:.3f} s'.format(stop_time - start_time))

    imsave(options.output, stylized_image)


def network(input_image):

    # Init graph
    sess = tf.Session()
    saver = tf.train.import_meta_graph('model/texture_net.chkp.meta')
    saver.restore(sess, 'model/texture_net.chkp')

    ## Get placeholder and last op
    graph = tf.get_default_graph()
    image = graph.get_tensor_by_name('image-placeholder:0')
    fetch = graph.get_tensor_by_name('deprocessing/concat:0')

    # TensorBoard
    tf.summary.FileWriter("./tb/", sess.graph).close()

    # Run session
    output = sess.run(fetch, feed_dict={image: input_image})
    sess.close()

    return output


if __name__ == '__main__':
    main()
