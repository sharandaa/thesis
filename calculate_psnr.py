from math import log10, sqrt
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

n_image = 1000
og_dir = 'test_AID'
upsampled_dir = '../Real-ESRGAN/results/x4'
  
def main():
    total = 0
    print("real esrgan x4")
    total_psnr = 0
    total_ssim = 0
    # Get list of files in each folder
    og_files = os.listdir(og_dir)
    upsampled_files = os.listdir(upsampled_dir)

    # Sort the files alphabetically to match the pairs
    og_files.sort()
    upsampled_files.sort()

    # Loop over the files in the folders and read the image pairs
    for file1, file2 in zip(og_files, upsampled_files):
        og_img = cv2.imread(os.path.join(og_dir, file1))
        up_img = cv2.imread(os.path.join(upsampled_dir, file2))
        
        score = cv2.PSNR(og_img, up_img)
        print(score)
        print(og_img, up_img)
        total_psnr += score

        # Convert the images to grayscale
        gray1 = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(up_img, cv2.COLOR_BGR2GRAY)
        ssim_value = ssim(gray1, gray2)
        total_ssim += ssim_value

        total += 1
        # Do something with the image pairs
        # For example, display them side by side
        #cv2.imshow('Image Pair', cv2.hconcat([og_img, up_img]))
    
    mean_psnr = total_psnr / n_image
    mean_ssim = total_ssim / n_image
    print(f"mean PSNR value is {mean_psnr} dB")
    print(f"mean SSIM value is {mean_ssim}")
    print("total n:")
    print(total)

def single_image():
    og_img = cv2.imread('C:/Users/shara/OneDrive/Documents/Scriptie/AID/mod_AID/all_bicubic_2x_300/airport_24x2.jpg')
    up_img = cv2.imread('C:/Users/shara/OneDrive/Documents/Scriptie/AID/mod_AID/Airport/airport_24.jpg')
        
    score = cv2.PSNR(og_img, up_img)
    print(score)

def size():
    img = cv2.imread('pond_61x2_out.jpg')
    h, w, c = img.shape
    print(f"Height and width of new image: {h}, {w}" )

if __name__ == "__main__":
    main()
    #single_image()
    #size()