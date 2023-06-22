import tensorflow as tf

import os
import math
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import array_to_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory

from IPython.display import display

from huggingface_hub import from_pretrained_keras

from natsort import natsorted
import cv2
from skimage.metrics import structural_similarity as ssim
import os

model = from_pretrained_keras("keras-io/super-resolution")

import PIL

def get_lowres_image(img, upscale_factor):
    """Return low-resolution image to use as model input."""
    return img.resize(
        (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
        PIL.Image.BICUBIC,
    )


def upscale_image(model, img):
    """Predict the result based on input image and restore the image as RGB."""
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0

    input = np.expand_dims(y, axis=0)
    out = model.predict(input)

    out_img_y = out[0]
    out_img_y *= 255.0

    # Restore the image in RGB color space.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
        "RGB"
    )
    return out_img

# The model weights (that are considered the best) are loaded into the model.
model.load_weights("finetuned_espcn_x3_35.h5")


highres_img_paths = sorted(
    [
        os.path.join("/scratch/s2630575/thesis/test_AID", fname)
        for fname in os.listdir("/scratch/s2630575/thesis/test_AID")
        if fname.endswith(".jpg")
    ]
)
print(highres_img_paths[0])

output_folder = '/scratch/s2630575/datasets/finetuned_espcn'
upscale_factor = 3

for index, highres_img_path in enumerate(highres_img_paths):
    highres_img = load_img(highres_img_path)
    #lowres_input = get_lowres_image(highres_img, upscale_factor)
    #w = lowres_input.size[0] * upscale_factor
    #h = lowres_input.size[1] * upscale_factor
    prediction = upscale_image(model, highres_img)
    
    # Get the filename and extension from the original image path
    file_name = os.path.basename(highres_img_path)
    file_name, ext = os.path.splitext(file_name)
    
    # Construct the output image path
    output_path = os.path.join(output_folder, f"{file_name}{ext}")
    
    # Save the upscaled image
    prediction.save(output_path)
