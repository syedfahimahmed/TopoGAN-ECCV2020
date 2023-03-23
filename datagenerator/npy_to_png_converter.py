import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Directory containing the .npy files
npy_dir = 'C:/Users/syed_fahim_ahmed/Desktop/Coding_With_Fahim/Unet_MC/dataset/training/boundaries'

# Directory to save the .png files
png_dir = 'C:/Users/syed_fahim_ahmed/Desktop/Coding_With_Fahim/Unet_MC/dataset/training_png/boundaries'

# Loop through the .npy files in the directory
for npy_file in os.listdir(npy_dir):
    if npy_file.endswith('.npy'):
        # Load the .npy file
        arr = np.load(os.path.join(npy_dir, npy_file))
        data = 2 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) - 1
        
        # Split the filename into the name and extension
        split_filename = os.path.splitext(npy_file)
        
        # Join filename without extension with the directory to save the .png file
        file_path = png_dir+"/"+split_filename[0]
        
        # Save the image as a .png file
        #plt.imsave(file_path+'.png', data, cmap='gray', vmin=-1, vmax=1)
        data = np.uint8((data + 1) * 127.5)
        
        # Create an image from the data
        image = Image.fromarray(data)
        
        image.save(file_path+'.png')
        
        # Load the .png file
        #data_copy = np.load(os.path.splitext(file_path)[0]+'.png')
        
        # Plot the data as an image
        #plt.imshow(data_copy, cmap='gray')
