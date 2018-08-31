import imageio, sys, os
from image_filters import *


def run(EXPONENT, x_0_1, y_0_1, x_1_1, y_1_1, x_0_2, y_0_2, x_1_2, y_1_2, x_0_3, y_0_3, x_1_3, y_1_3):
    images_path_list = os.listdir('./images/')

    for image_file_name in images_path_list:
        im = imageio.imread('images/' + image_file_name)
        image_histogram(image_file_name, im)
        print("Histogram for " + image_file_name + " done.")
        negative(image_file_name, im)
        print("Negative for " + image_file_name + " done.")
        logarithmic(image_file_name, im)
        print("Logarithmic for " + image_file_name + " done.")
        gamma_correction(image_file_name, im, EXPONENT)
        print("Gamma Correction for " + image_file_name + " done.")
        histogram_equalization(image_file_name, im)
        print("Histogram Equalization for " + image_file_name + " done.")
        piecewise_linear(image_file_name, im,
                         x_0_1, y_0_1, x_1_1, y_1_1, x_0_2, y_0_2, x_1_2, y_1_2, x_0_3, y_0_3, x_1_3, y_1_3)
        print("Piecewise Linear for " + image_file_name + " done.")


if __name__ == '__main__':
    run(sys.argv[1],
        sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
        sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9],
        sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[13])
