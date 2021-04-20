import os
import matplotlib.pyplot as plt
import numpy as np

import pdb
# ===============
# Create directory if not exist
# ===============
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ===============
# Save input image, ground truth and prediction for each volume to separate directory as png
# ===============    
def save_volume(image, label, prediction, volume_id, path):
    path_to_volume = path + ('/%s')%volume_id
    create_directory(path_to_volume)
    
    for i in range(label.shape[0]):
        plt.imsave('%s/%d_image.png'%(path_to_volume, i), np.rot90(np.squeeze(image[i])), cmap = plt.cm.bone)
        plt.imsave('%s/%d_label.png'%(path_to_volume, i), np.rot90(np.squeeze(label[i])), cmap = plt.get_cmap('tab10'))
        plt.imsave('%s/%d_prediction.png'%(path_to_volume, i), np.rot90(np.squeeze(prediction[i])), cmap = plt.get_cmap('tab10'))
    
    