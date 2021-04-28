import os
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn

import pdb

# ===============
# Setting the Seed
# ===============
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ===============
# Dice Loss function (from TransUNet)
# ===============
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

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

# ===============
# Change path name by adding "_i" if file/folder exist
# ===============

def file_new(path):

    path_raw, extension = os.path.splitext(path)[0], os.path.splitext(path)[1]
    
    path = path_raw
    k = 0
    
    if extension == '':
        while os.path.isdir(path):
            path = path_raw.rstrip('/') + '_' + str(k) + '/'
            k += 1        
    else:
        while os.path.isfile(path + extension):
            path = path_raw + '_' + str(k)
            k += 1
    
    return path + extension

# ===============
# Add a line to the beginning of a file
# ===============

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)

# ===============
# Training data save function for better inspection
# ===============

def pre_training_save(config_file, exp_config):
    dir_save = file_new('./results/' + exp_config.data_identifier_source + '/train/' + exp_config.train_id + '/')
    create_directory(dir_save)
    cfg_path = os.path.join(dir_save, 'cfg.py')
    copyfile(config_file, cfg_path)
    
    return dir_save
    
def save_training(training_data, config_file, exp_config, dir_save):
    
    create_directory(dir_save)
    csv_path = os.path.join(dir_save, 'epoch_data.csv')
    cfg_path = os.path.join(dir_save, 'cfg.py')
    
    training_data['epoch_data'].to_csv(csv_path)
    
    divSTR = '\n#?======================================================================='
    idSTR = '\n#!\t' + exp_config.data_identifier_source + '\t|\t' + exp_config.train_id
    dateSTR = '\n#* Finished at ' + training_data['end_date']
    timSTR = '\n#* Elapsed {:.2f} seconds'.format(training_data['total_time'])
    pthSTR = '\n#* Model parameters saved to ' + training_data['pth_save'] 
    bestSTR = '\n#* Best Validation Score:\n#* ' + training_data['best_str']
    cfgSTR = divSTR + '\n#? Exact configuration file can be found below' + divSTR + '\n \n'
    sumSTR = idSTR + dateSTR + timSTR + pthSTR + bestSTR + cfgSTR
    
    line_prepender(cfg_path, sumSTR)