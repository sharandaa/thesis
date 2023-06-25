from math import log10, sqrt
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from natsort import natsorted
import matplotlib.pyplot as plt

n_image = 1000
og_dir = 'test_AID'
upsampled_dirs = ['/home/s2630575/thesis/espcnmodel/ESPCN_x2', '/home/s2630575/Real-ESRGAN/results/x2', 
                  '/home/s2630575/SwinIR/results/swinir_classical_sr_x2', 
                  '/home/s2630575/SwinIR/results/swinir_lightweight_sr_x2', 
                  '/home/s2630575/swin2sr/results/swin2sr_classical_sr_x2']
names = ['ESPCN', 'Real-ESRGAN', 'SwinIR Classical', 'SwinIR Lightweight', 'Swin2SR']
psnr_dict = {}
ssim_dict = {}

upsampled_dirs = ['/home/s2630575/thesis/espcnmodel/ESPCN_x2', '/home/s2630575/SwinIR/results/swinir_lightweight_sr_x2']
names = ['espcn', 'lightweight swinir x2']
  
def main():
    for i in range(2):
        upsampled_dir = upsampled_dirs[i] 
        print(upsampled_dir)
        psnr_list = []
        ssim_list = []
        total = 0
        total_psnr = 0
        total_ssim = 0
        # Get list of files in each folder
        og_files = os.listdir(og_dir)
        upsampled_files = os.listdir(upsampled_dir)

        # Sort the files alphabetically to match the pairs
        og_files = natsorted(og_files)
        upsampled_files = natsorted(upsampled_files)

        # Loop over the files in the folders and read the image pairs
        for file1, file2 in zip(og_files, upsampled_files):
            og_img = cv2.imread(os.path.join(og_dir, file1))
            up_img = cv2.imread(os.path.join(upsampled_dir, file2))
            
            #print(file1, file2)
            score = cv2.PSNR(og_img, up_img)
            #print(score)
            psnr_list.append(score)
            #print(file1, file2)
            total_psnr += score

            # Convert the images to grayscale
            gray1 = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(up_img, cv2.COLOR_BGR2GRAY)
            ssim_value = ssim(gray1, gray2)
            ssim_list.append(ssim_value)
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

        psnr_dict[names[i]] = psnr_list
        ssim_dict[names[i]] = ssim_list

    group_data = [values for key, values in psnr_dict.items()]
    group_data_ssim = [values for key, values in ssim_dict.items()]

    print('psnr dict')
    print(psnr_dict)
    print('ssim dict')
    print(ssim_dict)
    #print(group_data)

    # psnr
    fig, ax = plt.subplots()
    ax.boxplot(group_data)

    # Add labels and title
    ax.set_xticklabels(psnr_dict.keys())
    #ax.set_xlabel('Groups')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('PSNR scores of scale x2')
    plt.xticks(rotation = 45)

    plt.savefig("psnrplotx2.pdf", format="pdf", bbox_inches="tight")

    # ssim
    fig, ax = plt.subplots()
    ax.boxplot(group_data_ssim)

    # Add labels and title
    ax.set_xticklabels(ssim_dict.keys())
    #ax.set_xlabel('Groups')
    ax.set_ylabel('SSIM')
    ax.set_title('SSIM scores of scale x2')
    plt.xticks(rotation = 45)

    plt.savefig("ssimplotx2.pdf", format="pdf", bbox_inches="tight")

def single_image():
    og_img = cv2.imread('C:/Users/shara/OneDrive/Documents/Scriptie/AID/mod_AID/all_bicubic_2x_300/airport_24x2.jpg')
    up_img = cv2.imread('C:/Users/shara/OneDrive/Documents/Scriptie/AID/mod_AID/Airport/airport_24.jpg')
        
    score = cv2.PSNR(og_img, up_img)
    print(score)

def size():
    """
    img = cv2.imread('pond_61x2_out.jpg')
    h, w, c = img.shape
    print(f"Height and width of new image: {h}, {w}" )
    """
    og_files = os.listdir("test_AID")
    print(og_files)
    natsort_file_names = natsorted(og_files)
    print(natsort_file_names)


if __name__ == "__main__":
    main()
    #single_image()
    #size()