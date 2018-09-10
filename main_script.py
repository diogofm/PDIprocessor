import imageio, sys, os
from image_filters import *


def run_day_one(EXPONENT, LOG, x_0_1, y_0_1, x_1_1, y_1_1, x_0_2, y_0_2, x_1_2, y_1_2, x_0_3, y_0_3, x_1_3, y_1_3):
    images_path_list = os.listdir('./images/')

    for image_file_name in images_path_list:
        im = imageio.imread('images/' + image_file_name)
        image_histogram(image_file_name, im)
        print("Histogram for " + image_file_name + " done.")
        negative(image_file_name, im.copy())
        print("Negative for " + image_file_name + " done.")
        logarithmic(image_file_name, im.copy(), LOG)
        print("Logarithmic for " + image_file_name + " done.")
        gamma_correction(image_file_name, im.copy(), EXPONENT)
        print("Gamma Correction for " + image_file_name + " done.")
        histogram_equalization(image_file_name, im.copy())
        print("Histogram Equalization for " + image_file_name + " done.")
        piecewise_linear(image_file_name, im.copy(),
                         x_0_1, y_0_1, x_1_1, y_1_1, x_0_2, y_0_2, x_1_2, y_1_2, x_0_3, y_0_3, x_1_3, y_1_3)
        print("Piecewise Linear for " + image_file_name + " done.")
        bit_layers(image_file_name, im)
        print("BitLayer for " + image_file_name + " done.")


def run_day_two(kernel_weighted, kernel_conv, boost_constant):
    images_path_list = os.listdir('./images/')

    for image_file_name in images_path_list:
        im = imageio.imread('images/' + image_file_name)
        image_histogram(image_file_name, im)
        print("Histogram for " + image_file_name + " done.")
        averaging(image_file_name, im.copy())
        print("Average filter for " + image_file_name + " done.")
        weighted_averaging(image_file_name, im.copy(), kernel_weighted)
        print("Weighted Average filter for " + image_file_name + " done.")
        print("Used Kernel:")
        print(kernel_weighted)
        median_filter(image_file_name, im.copy())
        print("Median filter for " + image_file_name + " done.")
        convolution(image_file_name, im.copy(), kernel_conv)
        print("Convolution filter for " + image_file_name + " done.")
        print("Used Kernel:")
        print(kernel_conv)
        laplacian(image_file_name, im.copy())
        print("Laplacian filter for " + image_file_name + " done.")
        sobel(image_file_name, im.copy())
        print("Sobel filter for " + image_file_name + " done.")
        highboost(image_file_name, im.copy(), boost_constant)
        print("Highboost filter for " + image_file_name + " done.")


if __name__ == '__main__':
    # run_day_one(0.4, (255/log(256)), 0, 0, 10, 10, 11, 11, 13, 100, 14, 101, 255, 255)

    weighted_kernel = np.array([[1, 5, 1], [5, 1, 5], [1, 5, 1]])
    conv_kernel = np.array([[0, 5, 0], [0, 5, 0], [0, 5, 0]])
    highboost_constant = 2

    run_day_two(weighted_kernel, conv_kernel, highboost_constant)

    # run(sys.argv[1],
    #     sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
    #     sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9],
    #     sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[13])
