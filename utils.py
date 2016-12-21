from argparse import ArgumentParser

import scipy.misc
import numpy as np


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--input', dest='input', help='input image', metavar='INPUT', required=True)
    parser.add_argument('--output', dest='output', help='output image', metavar='OUTPUT', required=True)
    return parser


def imread(path):
    return scipy.misc.imread(path).astype(np.float)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)
