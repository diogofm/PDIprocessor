import imageio
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
