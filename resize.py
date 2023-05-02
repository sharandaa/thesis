import cv2
import matplotlib.pyplot as plt
import os

# https://www.tutorialspoint.com/how-to-resize-an-image-in-opencv-using-python

factor = 4

#img = cv2.imread('C:/Users/shara/OneDrive/Documents/Scriptie/AID/mod_AID/Airport/airport_24.jpg')
#h, w, c = img.shape
#print(f"Height and width of original image: {h}, {w}" )

h = 600
w = 600

new_height = int(h / factor)
new_width = int(w / factor)
        
# resize the image - down
#resized_img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR)
#h, w, c = resized_img.shape
#print(f"Height and width of new image: {h}, {w}" )

#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

#plt.imshow(resized_img)
#plt.show()


# https://stackoverflow.com/questions/66444804/save-an-image-with-the-same-name-after-editing-in-python-opencv-module
image_dir = 'C:/Users/shara/OneDrive/Documents/Scriptie/datasets/test_AID'
output_dir = 'C:/Users/shara/OneDrive/Documents/Scriptie/datasets/test_AID_x4'

source = "C:/Users/shara/OneDrive/Documents/Scriptie/AID"

# Source folders containing pictures
"""
folders = [source + "/mod_AID/Airport", source + "/mod_AID/BareLand", source + "/mod_AID/BaseballField", source + "/mod_AID/Beach",
               source + "/mod_AID/Bridge", source + "/mod_AID/Center", source + "/mod_AID/Church", source + "/mod_AID/Commercial",
               source + "/mod_AID/DenseResidential", source + "/mod_AID/Desert", source + "/mod_AID/Farmland", source + "/mod_AID/Forest",
               source + "/mod_AID/Industrial", source + "/mod_AID/Meadow", source + "/mod_AID/MediumResidential", source + "/mod_AID/Mountain",
               source + "/mod_AID/Park", source + "/mod_AID/Parking", source + "/mod_AID/Playground", source + "/mod_AID/Pond", source + "/mod_AID/Port",
               source + "/mod_AID/RailwayStation", source + "/mod_AID/Resort", source + "/mod_AID/River", source + "/mod_AID/School",
               source + "/mod_AID/SparseResidential", source + "/mod_AID/Square", source + "/mod_AID/Stadium", source + "/mod_AID/StorageTanks",
               source + "/mod_AID/Viaduct"]
"""
"""
#for downsizing:
for folder in folders:
    print(folder)
    #iterate through all the files in the image directory
    for _, _, image_names in os.walk(folder):
        #iterate through all the files in the image_dir
        for image_name in image_names:
            # check for extension .jpg
            #print(image_name)
            if '.jpg' in image_name:
                # get image read path(path should not contain spaces in them)
                filepath = os.path.join(folder, image_name)
                n = os.path.splitext(image_name)[0]
                #print(n)
                b = n + 'x4' + '.jpg'
                #lr_name = image_name + '_lr'
                # get image write path
                dstpath = os.path.join(output_dir, b)
                #print(filepath, dstpath)
                # read the image
                image = cv2.imread(filepath)
                # do your processing
                resized_img = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_CUBIC)
                h, w, c = resized_img.shape
                #print(f"Height and width of new image: {h}, {w}" )

                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                # write the image in a different path with the same name
                cv2.imwrite(dstpath, resized_img)
                #print(image.shape, resized_img.shape)
"""


#for upsizing:
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
            b = n + 'x4' + '.jpg'
            #lr_name = image_name + '_lr'
            # get image write path
            dstpath = os.path.join(output_dir, b)
            #print(filepath, dstpath)
            # read the image
            image = cv2.imread(filepath)
            # do your processing
            resized_img = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_CUBIC)
            #h, w, c = resized_img.shape
            #print(f"Height and width of new image: {h}, {w}" )

            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            # write the image in a different path with the same name
            cv2.imwrite(dstpath, resized_img)
            #print(image.shape, resized_img.shape)

#python main_test_swinir.py --task lightweight_sr --scale 2 --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth --folder_lq testsets/BareLand_lr --folder_gt testsets/BareLand
            #python main_test_swinir.py --task classical_sr --scale 2 --training_patch_size 48 --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth --folder_lq testsets/downsample_2x_upscale_2x --folder_gt testsets/Airport

"""
des = 'C:/Users/shara/OneDrive/Documents/Scriptie/AID/mod_AID/all_hr'

for folder in folders:
    print(folder)
    #iterate through all the files in the image directory
    for _, _, image_names in os.walk(folder):
        #iterate through all the files in the image_dir
        for image_name in image_names:
            # check for extension .jpg
            #print(image_name)
            if '.jpg' in image_name:
                # get image read path(path should not contain spaces in them)
                filepath = os.path.join(folder, image_name)
                n = os.path.splitext(image_name)[0]
                #print(n)
                b = n + '.jpg'
                #lr_name = image_name + '_lr'
                # get image write path
                dstpath = os.path.join(des, b)
                #print(filepath, dstpath)
                # read the image
                image = cv2.imread(filepath)
                # do your processing
                #resized_img = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_CUBIC)
                #h, w, c = resized_img.shape
                #print(f"Height and width of new image: {h}, {w}" )

                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                # write the image in a different path with the same name
                cv2.imwrite(dstpath, image)

                python test_sample.py --config configs/SRx2_EDTT_ImageNet200K.py --model pretrained/SRx2_EDTT_ImageNet200K.pth --input test_sets/SR/Airport_downsample --output result --gt test_sets/SR/Airport
                python inference_realesrgan.py -n RealESRGAN_x4plus -i Airport_downsample --outscale 2 -o Airport
                """