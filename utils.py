import os
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import pickle5 as pickle
from importlib.machinery import SourceFileLoader

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

def pre_training_save(exp_config):
    dir_save = file_new('./results/' + exp_config.data_identifier_source + '/train/' + exp_config.train_id + '/')
    
    train_id = dir_save.rsplit('/')[-2]
    
    if train_id != exp_config.train_id:
        print('Train ID changed from {} to {}'.format(exp_config.train_id, train_id))
        exp_config.train_id = train_id
    
    create_directory(dir_save)
    cfg_path = os.path.join(dir_save, 'cfg.py')
    copyfile(exp_config.__file__, cfg_path)
    
    setattr(exp_config, 'save_path', dir_save)
    
    return exp_config
    
def save_training(exp_config, training_data):
    
    dir_save = exp_config.save_path
    
    create_directory(dir_save)
    csv_path = os.path.join(dir_save, 'epoch_data.csv')
    cfg_path = os.path.join(dir_save, 'cfg.py')
    pkl_path = os.path.join(dir_save, 'data.pickle')
    
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
    
    with open(pkl_path , 'wb') as f:
        pickle.dump(training_data, f, protocol=pickle.HIGHEST_PROTOCOL)

# ===============
# Save Test
# ===============

def pre_test_save(exp_config):
    
    if not hasattr(exp_config, 'save_path'):
        default_sp = './results/%s/test/%s'%(exp_config.data_identifier_source, exp_config.experiment_name)
        setattr(exp_config, 'save_path', default_sp)
    
    dir_save = file_new(exp_config.save_path)
    
    if dir_save != exp_config.save_path:
        print('Save path changed to ', dir_save)
    
    exp_config.save_path = dir_save
    create_directory(dir_save)
    
    cfg_path = os.path.join(dir_save, 'cfg.py')
    copyfile(exp_config.__file__, cfg_path)
    
    if hasattr(exp_config, '__fileTrain__'):
        cfg_train_path = os.path.join(dir_save, 'cfg_train.py')
        copyfile(exp_config.__fileTrain__, cfg_train_path)
    
    return exp_config

# ===============
# Merging Modules
# ===============

def merge_modules(*modules):
    
    module_list = list(modules)
    
    if len(module_list) == 1:
        return module_list[0]
    
    module_root = module_list[0]
    module_add = module_list[1]

    for attr in list(set(dir(module_root) + dir(module_add))):
        if not hasattr(module_root, attr):
            attr_value = getattr(module_add, attr, None)
            setattr(module_root, attr, attr_value)
    
    module_list[1] = module_root        
    module_list.pop(0)
    
    return merge_modules(*module_list)
    
# ===============
# Training to Test Module
# ===============

def train2test(test_config):
    
    train_id = getattr(test_config, 'train_id', None)
    if train_id is None:
        return test_config
    
    dir_save = './results/' + test_config.data_identifier_source + '/train/' + train_id + '/'
    cfg_tr_path = dir_save + 'cfg.py'
    pkl_tr_path = dir_save + 'data.pickle'
    
    cfg_tr_name = cfg_tr_path.split('/')[-1].rstrip('.py')
    train_config = SourceFileLoader(cfg_tr_name, cfg_tr_path).load_module()
    
    cfg = merge_modules(test_config, train_config)
    setattr(cfg, '__fileTrain__', train_config.__file__)
    
    attr_pth = 'model_path'
    
    if not hasattr(cfg, attr_pth):
        with open(pkl_tr_path, 'rb') as f:
            training_data = pickle.load(f)
        setattr(cfg, attr_pth, training_data['pth_save'])
    
    return cfg