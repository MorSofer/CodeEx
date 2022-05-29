#starting by importing necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse

#contructing the arguments parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-o", "--output", required=True, help="path to output directort to so augmention examples")
ap.add_argument("-p", "--prefix", type=str, default="image", help="output filename prefix")
args = vars(ap.parse_args())
#--image the path for imput image we want to apply agumnetation and visualize the result
#--output after we apply the agumantetion we want to save the result in order to inspect it.
#--prefix a string that will be prepended to the output image filename.

#loading our data and adding extra demnsion to the image, as we preparing the image for classification
#loading, converting to array and then reshape the image:
print("[INFO] loading exmaple image...")
image = load_img(args["image"])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

#Constucting the image generator for data augmention then
#initialize the total number of images generated so far
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
horizontal_flip=True, fill_mode="nearest")
total = 0

'''
* we focus on the parameters we most likely use for data agumantetion in our applications:
* rotation_range - control the degree of rnadom rotation of +/-n in our case +/-30
* heghit and width - used for horizontical and vertical shifts respectively, in our case in 10%
* sheer_range - used to control the angle in conuterclockwise direction as radians in wich we allow our 
    image to be sheared
* zoom_range - the floating poinr value we allows to zoom in and out accoriding uniform distribustion
    of the vlause [1-zoom_range, 1+zoom_range]
* horizontal_flip = bolean value if we allow or not to filp the image

Note1: for most CV app flip image dosen't cange the resulting class labels, but there are applications
where a horizontal or vertical dose change the semantic meaning of the image

Note2: it's important to remember we want to create more new data BASEd on our exisit data whithout
effecting the class label or change them or creating "bad" images'''

#consturcting the actual python generator
print("[INFO] generatin images...")
imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
	save_prefix=args["prefix"], save_format="jpg")

# loop over examples from our image data augmentation generator
for image in imageGen:
	# increment our counter
	total += 1

	# if we have reached 10 examples, break from the loop
	if total == 10:
		break
