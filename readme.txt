*****README FILE*****

PREREQUISITES:-

Programming Language:- Python3

Libraries/Packages:-
	cv2 
	numpy
	os

Required files:-
1. Input .bmp mosaic image file 
2. Input original .jpg image file


RUN FILE:-
1. Put both the input files in the image_set folder.
2. Run A1.py
3. Enter the name of .bmp file without extension eg: for 'crayons_mosaic.bmp' enter 'crayons_mosaic'.
4. Enter the name of .jpg original image file without extension eg: for 'crayons.jpg' enter 'crayons'.
5. All the four output images will be displayed and written in a folder named 'Output'. 
6. Press ENTER KEY by selecting any displayed image to exit the program and close all the displayed images.


OUTPUT:-

Program outputs 4 images as follows:

1. 'Original,demosaic.jpg' - Comparison of original image and result of Linear Interpolation(Part 1).

2. 'Original,bill_freeman_demosaic.jpg' - Comparison of original image and result of Bill Freeman(Part 2).

3. 'part1_diff,part2_diff-3channel.jpg' - 3 channel comparison of image created by squared differences between the original and result of liner interpolation and the image created by squared differences between the original and result of Bill Freeman technique.

4. 'part1_diff,part2_diff-grayscale.jpg' - Grayscale comparison of image created by squared differences between the original and result of liner interpolation and the image created by squared differences between the original and result of Bill Freeman technique.


 

