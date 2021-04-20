from importlib.machinery import SourceFileLoader
from data import data_loader
import argparse
import torch
from models.ctun import Ctun
#import torch.nn as nn
#import torch.optim as optim
import sklearn.metrics as met
import numpy as np
from utils import *

import matplotlib
matplotlib.use('Agg')

import pdb

def test_segmentation_network(exp_config, model, loader_test):
    
    create_directory(exp_config.path_to_save_images)
    f = open(('%s/dice_score.txt')%exp_config.path_to_save_images, "w")
    
    model.eval()
    
    counter = 0
    counter_volume = 0
    dice_total = 0
    for data, target in loader_test:
        data = data.cuda()
        
        _, _, pred_argmax = model(data)
        
        if counter == 0:
            data_volume = torch.squeeze(data).detach().cpu().numpy()
            target_volume = torch.squeeze(target).numpy()
            pred_volume = pred_argmax.detach().cpu().numpy()
        else:
            data_volume = np.concatenate((data_volume, torch.squeeze(data).detach().cpu().numpy()), axis = 0)
            target_volume = np.concatenate((target_volume, torch.squeeze(target).numpy()), axis = 0)
            pred_volume = np.concatenate((pred_volume, pred_argmax.detach().cpu().numpy()), axis = 0)
        
        counter += data.shape[0]
        
        if counter == exp_config.image_size[2]:
            if exp_config.save_images:
                save_volume(data_volume, target_volume, pred_volume, counter_volume, exp_config.path_to_save_images)
            dice_for_each_class = met.f1_score(target_volume.flatten(), pred_volume.flatten(), average = None)
            dice_volume = np.mean(dice_for_each_class)
            dice_total += dice_volume
            
            print('Dice score of volume %d = %f'%(counter_volume, dice_volume))
            f.write(('Vol_%d\t%.10f \n')%(counter_volume, dice_volume))
            
            counter = 0
            counter_volume += 1
            
    print('Average Dice score of all volumes = %f'%(dice_total / counter_volume))
    f.write(('Average\t%.10f \n')%(dice_volume))
    
    f.close()
        
def main(exp_config):

    # =====================
    # Define network architecture
    # =====================    
    model = Ctun(exp_config).cuda()
    model.cuda()
    
    # =========================
    # Load source dataset
    # =========================
    _, test_loader, _ = data_loader.load_datasets(exp_config.data_identifier_target, exp_config.batch_size)
    
    # =========================
    # Load pre_trained model
    # =========================
    model.load_state_dict(torch.load(exp_config.path_to_load_pretrained_model))
    
    # =========================
    # Test on source data
    # =========================
    test_segmentation_network(exp_config, model, test_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment config file")
    args = parser.parse_args()

    config_file = args.EXP_PATH
    config_module = config_file.split('/')[-1].rstrip('.py')
        
    exp_config = SourceFileLoader(config_module, config_file).load_module() # exp_config stores configurations in the given config file under experiments folder.

    main(exp_config=exp_config)

