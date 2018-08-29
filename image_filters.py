import imageio
import numpy as np
from math import *


def negative(im_name, im):
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            neg_byte = 255 - im[i][j]
            im[i][j] = neg_byte
    imageio.imwrite('post_processed_images/negative_' + im_name, im)


def logarithmic(im_name, im):
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            log_byte = (255/log(256)) * log(1 + im[i][j])
            im[i][j] = log_byte
    imageio.imwrite('post_processed_images/log_' + im_name, im)


def gamma_correction(im_name, im, exponent):
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            gamma_byte = (255/(256 ** exponent)) * ((1 + im[i][j]) ** exponent)
            im[i][j] = gamma_byte
    imageio.imwrite('post_processed_images/gamma_' + im_name, im)


def bit_layers(im_name, im):
    layer_0_im = np.ndarray((im.shape[0], im.shape[1]), dtype=np.uint8)
    layer_1_im = np.ndarray((im.shape[0], im.shape[1]), dtype=np.uint8)
    layer_2_im = np.ndarray((im.shape[0], im.shape[1]), dtype=np.uint8)
    layer_3_im = np.ndarray((im.shape[0], im.shape[1]), dtype=np.uint8)
    layer_4_im = np.ndarray((im.shape[0], im.shape[1]), dtype=np.uint8)
    layer_5_im = np.ndarray((im.shape[0], im.shape[1]), dtype=np.uint8)
    layer_6_im = np.ndarray((im.shape[0], im.shape[1]), dtype=np.uint8)
    layer_7_im = np.ndarray((im.shape[0], im.shape[1]), dtype=np.uint8)

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            bit_string = np.binary_repr(im[i][j], 8)
            layer_0_im[i][j] = np.uint8(int('0000000' + bit_string[7], 2))
            layer_1_im[i][j] = np.uint8(int('000000' + bit_string[6] + '0', 2))
            layer_2_im[i][j] = np.uint8(int('00000' + bit_string[5] + '00', 2))
            layer_3_im[i][j] = np.uint8(int('0000' + bit_string[4] + '000', 2))
            layer_4_im[i][j] = np.uint8(int('000' + bit_string[3] + '0000', 2))
            layer_5_im[i][j] = np.uint8(int('00' + bit_string[2] + '00000', 2))
            layer_6_im[i][j] = np.uint8(int('0' + bit_string[1] + '000000', 2))
            layer_7_im[i][j] = np.uint8(int(bit_string[0] + '0000000', 2))
    imageio.imwrite('post_processed_images/bitlayer0_' + im_name, layer_0_im)
    imageio.imwrite('post_processed_images/bitlayer1_' + im_name, layer_1_im)
    imageio.imwrite('post_processed_images/bitlayer2_' + im_name, layer_2_im)
    imageio.imwrite('post_processed_images/bitlayer3_' + im_name, layer_3_im)
    imageio.imwrite('post_processed_images/bitlayer4_' + im_name, layer_4_im)
    imageio.imwrite('post_processed_images/bitlayer5_' + im_name, layer_5_im)
    imageio.imwrite('post_processed_images/bitlayer6_' + im_name, layer_6_im)
    imageio.imwrite('post_processed_images/bitlayer7_' + im_name, layer_7_im)
