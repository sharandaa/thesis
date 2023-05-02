import cv2
from cv2 import dnn_superres
import os

image_dir = 'C:/Users/shara/OneDrive/Documents/Scriptie/AID/mod_AID/downsample_4x'
output_dir = 'C:/Users/shara/OneDrive/Documents/Scriptie/AID/mod_AID/all_espcn_x4'

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read image
#image = cv2.imread('./mod_AID/downsample_2x_upscale_2x/airport_24x2.jpg')

# Read the desired model
path = "ESPCN_x4.pb"
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("espcn", 4)

# Upscale the image
#result = sr.upsample(image)

# Save the image
#cv2.imwrite("./espcnx2.jpg", result)

for _, _, image_names in os.walk(image_dir):
        #iterate through all the files in the image_dir
        for image_name in image_names:
            # check for extension .jpg
            #print(image_name)
            if '.jpg' in image_name:
                # get image read path(path should not contain spaces in them)
                filepath = os.path.join(image_dir, image_name)
                n = os.path.splitext(image_name)[0]
                #print(n)
                b = n + '_ESPCNx2' + '.jpg'
                #lr_name = image_name + '_lr'
                # get image write path
                dstpath = os.path.join(output_dir, b)
                #print(filepath, dstpath)
                # read the image
                image = cv2.imread(filepath)
                # do your processing
                result = sr.upsample(image)

                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                # write the image in a different path with the same name
                cv2.imwrite(dstpath, result)
                #print(image.shape, resized_img.shape)