import cv2
import numpy as np
import os

def main():
    print(cv2.__version__)

    #enter the name of input images
    input_image = input('Please enter the name of .bmp input image without extension. eg. crayons_mosaic ! \n').strip()
    original_image = input('Please enter the name of .jpg original image without extension. eg. crayons ! \n').strip()

    #read input images
    bmp_img = cv2.imread('image_set/' + input_image + '.bmp', 3)
    bmp_img = np.float32(bmp_img)
    orig_img = cv2.imread('image_set/' + original_image + '.jpg', 3)

    #create 3 channels(B,G,R) as per the Bayer's Pattern
    channel_b, channel_g, channel_r = create_channel(bmp_img)

    #perform linear interpolation, calculate noise in 3 channel and grayscale, concat the result images
    filter_b, filter_g, filter_r, reconstructed_image = linear_interpolation(channel_b, channel_g, channel_r)
    error_image = calc_root_squared_diff_3channel(orig_img, reconstructed_image)
    error_image_grayscale = calc_root_squared_diff_graysacle(orig_img, reconstructed_image)
    fin_concat_img = concat_images(orig_img, reconstructed_image)

    #perform Bill Freeman technique, calculate noise in 3 channel and grayscale, concat the result images
    bf_recontructed = bill_freeman(filter_b, filter_g, filter_r)
    bf_error_image = calc_root_squared_diff_3channel(orig_img, bf_recontructed)
    bf_error_image_grayscale = calc_root_squared_diff_graysacle(orig_img, bf_recontructed)
    bf_fin_concat_img = concat_images(orig_img, bf_recontructed)

    channel3_error_img = concat_images(error_image, bf_error_image)
    grayscale_error_img = concat_images(error_image_grayscale, bf_error_image_grayscale)

    #creating an 'Output' folder if it does not exist
    if not os.path.exists('Output'):
        os.mkdir('Output')

    #Writing the four Output files
    cv2.imwrite('Output/Original,demosaic.jpg', fin_concat_img)
    cv2.imwrite('Output/Original,bill_freeman_demosaic.jpg', bf_fin_concat_img)

    cv2.imwrite('Output/part1_diff,part2_diff-3channel.jpg', channel3_error_img) #error image in 3 channels
    cv2.imwrite('Output/part1_diff,part2_diff-grayscale.jpg', grayscale_error_img) #error image in grayscale

    #Displaying the four Output files
    cv2.imshow('Original,demosaic', fin_concat_img)
    cv2.imshow('Original,bill_freeman_demosaic', bf_fin_concat_img)

    cv2.imshow('part1_diff,part2_diff-3channel', channel3_error_img)
    cv2.imshow('part1_diff,part2_diff-grayscale', grayscale_error_img)


    key = cv2.waitKey(0)
    if key == 13:  # waiting for enter key to exit
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def bill_freeman(filter_b, filter_g, filter_r):
    filter_b = filter_b / 255
    filter_g = filter_g / 255
    filter_r = filter_r / 255

    b = filter_b - filter_r
    g = filter_g - filter_r

    bf_b = cv2.medianBlur(b, 3) + filter_r
    bf_g = cv2.medianBlur(g, 3) + filter_r

    bf_b = (np.clip(bf_b, 0, 1)) * 255
    bf_g = (np.clip(bf_g, 0, 1)) * 255
    filter_r = (np.clip(filter_r, 0, 1)) * 255

    bf_recontructed = cv2.merge((bf_b, bf_g, filter_r))
    bf_recontructed = np.uint8(bf_recontructed)

    return bf_recontructed


def concat_images(orig_img, reconstructed_image):
    return np.hstack((orig_img, reconstructed_image))


def calc_root_squared_diff_3channel(orig_img, recontructed):
    b, g, r = cv2.split(orig_img)
    b_new, g_new, r_new = cv2.split(recontructed)

    diff_b = np.sqrt((np.square(b / 255 - b_new / 255)))
    diff_g = np.sqrt((np.square(g / 255 - g_new / 255)))
    diff_r = np.sqrt((np.square(r / 255 - r_new / 255)))

    diff = cv2.merge(((diff_b * 255), (diff_g * 255), (diff_r * 255)))
    diff = np.uint8(diff)

    return diff

def calc_root_squared_diff_graysacle(orig_img, recontructed):
    b, g, r = cv2.split(orig_img)
    b_new, g_new, r_new = cv2.split(recontructed)

    diff_b = (np.square(b / 255 - b_new / 255))
    diff_g = (np.square(g / 255 - g_new / 255))
    diff_r = (np.square(r / 255 - r_new / 255))

    diff = np.sqrt(diff_b + diff_g + diff_r)
    diff = diff * 255

    diff = np.uint8(diff)

    return diff


def create_channel(bmp_img):
    b_bmp, g_bmp, r_bmp = cv2.split(bmp_img)
    width, height = bmp_img.shape[:2]

    b_channel = np.zeros((width, height), dtype=np.uint8)
    b_channel[::2] = 1
    b_channel[:, 1::2] = 0
    channel_b = cv2.bitwise_and(b_bmp, b_bmp, mask=b_channel)

    g_channel = np.ones((width, height), dtype=np.uint8)
    g_channel[::2] = 0
    g_channel[:, 0::2] = 0
    channel_g = cv2.bitwise_and(g_bmp, g_bmp, mask=g_channel)

    r_channel = np.ones((width, height), dtype=np.uint8) - b_channel - g_channel
    channel_r = cv2.bitwise_and(r_bmp, r_bmp, mask=r_channel)

    return channel_b, channel_g, channel_r


def linear_interpolation(channel_b, channel_g, channel_r):
    kernel_b = np.array([[.25, .5, .25], [.5, 1, .5], [.25, .5, .25]])
    kernel_g = np.array([[.25, .5, .25], [.5, 1, .5], [.25, .5, .25]])
    kernel_r = np.array([[0, .25, 0], [.25, 1, .25], [0, .25, 0]])

    filter_b = cv2.filter2D(channel_b, -1, kernel_b)
    filter_g = cv2.filter2D(channel_g, -1, kernel_g)
    filter_r = cv2.filter2D(channel_r, -1, kernel_r)

    reconstructed_image = cv2.merge((filter_b, filter_g, filter_r))
    reconstructed_image = np.uint8(reconstructed_image)

    return filter_b, filter_g, filter_r, reconstructed_image


if __name__ == "__main__":
    main()
