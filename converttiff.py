import os
from PIL import Image

def convert_jpg_to_tiff(input_folder, output_folder):

    # Get a list of all JPG files in the input folder
    jpg_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

    for jpg_file in jpg_files:
        # Remove the "x2" part from the filename
        jpg_name = os.path.splitext(jpg_file)[0]
        jpg_name = jpg_name.replace("x2", "")

        # Open the JPG file
        jpg_path = os.path.join(input_folder, jpg_file)
        with Image.open(jpg_path) as img:
            # Generate the output TIFF file path
            tiff_file = jpg_name + '.tiff'
            tiff_path = os.path.join(output_folder, tiff_file)

            # Convert and save the image as TIFF
            img.save(tiff_path, format='TIFF')

# Specify the input and output folders
input_folder = '../datasets/test_AID'
output_folder = '../datasets/test_AID_tiff'

# Call the function to convert JPG files to TIFF
convert_jpg_to_tiff(input_folder, output_folder)
