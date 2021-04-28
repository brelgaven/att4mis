from importlib.machinery import SourceFileLoader
from data import data_loader
import argparse
import torch
from models.ctun import Ctun
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as met
import numpy as np

import matplotlib
matplotlib.use('Agg')

import pdb

from utils import *
import pandas as pd
from timeit import default_timer as timer
from datetime import datetime

def train_segmentation_network(exp_config, model, trainable_parameters, loader_train, loader_val, path_to_save_model):

    optimizer = optim.Adam(trainable_parameters, lr = 1e-4, weight_decay = 0.0) # Define optimizer - update all paramaters
    
    loss_mult = getattr(exp_config, 'loss_mult', [0.5, 0.5])
    ce_mult, dice_mult = loss_mult[0], loss_mult[1]
    
    if ce_mult != 0:
        ce_loss = nn.CrossEntropyLoss()
    if dice_mult != 0:
        dice_loss = DiceLoss(exp_config.num_classes)
    
    
    best_loss_val = 0
    
    pth_save_path = file_new(('%s/%s_model_segmentation_%s.pth')%
                             (path_to_save_model, exp_config.data_identifier_source, exp_config.train_id))
    epoch_data = []    
    start_time = timer()
    
    for epoch in range(exp_config.number_of_epoch):
        
        model.train() # Switch on training mode            
        running_loss_train = 0.0
        running_dice_train = 0.0
        counter = 0
        for data, target in loader_train:
            data, target = data.cuda(), torch.squeeze(target).cuda().long()
            
            optimizer.zero_grad()
            pred_logits, pred_softmax, pred_argmax = model(data)
            
            loss_ce = ce_loss(pred_logits, target) if ce_mult != 0 else 0.0
            loss_dice = dice_loss(pred_logits, target, softmax=True) if dice_mult != 0 else 0.0
            loss = ce_mult * loss_ce + dice_mult * loss_dice
            loss.backward()
            optimizer.step()
            
            dice_each_class = met.f1_score(torch.flatten(target).detach().cpu().numpy(), torch.flatten(pred_argmax).detach().cpu().numpy(), average = None)
            running_dice_train += np.mean(dice_each_class)            
            running_loss_train += loss.item()
            
            counter += 1
        running_loss_train = running_loss_train / counter
        running_dice_train = running_dice_train / counter
        
        model.eval() # Switch on evaluation mode
        running_loss_val = 0.0
        running_dice_val = 0.0
        counter = 0
        for data, target in loader_val:
            data, target = data.cuda(), torch.squeeze(target).cuda().long()
            
            pred_logits, pred_softmax, pred_argmax = model(data)

            loss_ce = ce_loss(pred_logits, target) if ce_mult != 0 else 0.0
            loss_dice = dice_loss(pred_logits, target, softmax=True) if dice_mult != 0 else 0.0
            loss = ce_mult * loss_ce + dice_mult * loss_dice
            dice_each_class = met.f1_score(torch.flatten(target).detach().cpu().numpy(), torch.flatten(pred_argmax).detach().cpu().numpy(), average = None)
            running_dice_val += np.mean(dice_each_class)
            running_loss_val += loss.item()
            
            counter += 1
        running_loss_val = running_loss_val / counter
        running_dice_val = running_dice_val / counter
        
        if epoch == 0 or best_loss_val > running_loss_val:
            best_loss_val = running_loss_val
            torch.save(model.state_dict(), pth_save_path) 
            best_str = 'epoch:%d - loss_tr: %.10f loss_val: %.10f - dice_tr: %.10f dice_val: %.10f' % (epoch, running_loss_train, running_loss_val, running_dice_train, running_dice_val) 
            print(best_str, '- Last Saved')
            save_flag = 1 
        else:
            print('epoch:%d - loss_tr: %.10f loss_val: %.10f - dice_tr: %.10f dice_val: %.10f' %
                  (epoch, running_loss_train, running_loss_val, running_dice_train, running_dice_val))
            save_flag = 0 
            
        epoch_data.append([running_loss_train, running_loss_val, running_dice_train, running_dice_val, save_flag]) 

    training_data = {   'epoch_data': pd.DataFrame(epoch_data, columns =['loss_tr', 'loss_val', 'dice_tr', 'dice_val', 'save']),
                        'total_time': timer() - start_time,
                        'end_date'  : datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                        'pth_save'  : pth_save_path,
                        'best_str'  : best_str, }

    return training_data 

def main(exp_config):
    
    print('Train ID \t:', exp_config.train_id, '\nDataset \t:', exp_config.data_identifier_source)
    
    # =====================
    # Set determinism of the algorithm
    # =====================   
    deterministic = getattr(exp_config, 'deterministic', True)
    seed = getattr(exp_config, 'seed', 0)

    if not deterministic:
        print('The code runs nondeterministicaly.')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        print('The code runs deterministicaly.')
        set_seed(seed)
    
    # =====================
    # Define network architecture
    # =====================    
    model = Ctun(exp_config).cuda()
    model.cuda()
    
    # =========================
    # Load source dataset
    # =========================
    source_train_loader, source_test_loader, source_val_loader = data_loader.load_datasets(exp_config.data_identifier_source, exp_config.batch_size)
    
    # =========================
    # Train on source data
    # =========================
    training_data = train_segmentation_network(exp_config, model, model.parameters(), source_train_loader, source_val_loader, exp_config.path_to_save_trained_model)

    return training_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment config file")
    args = parser.parse_args()

    config_file = args.EXP_PATH
    config_module = config_file.split('/')[-1].rstrip('.py')
        
    exp_config = SourceFileLoader(config_module, config_file).load_module() # exp_config stores configurations in the given config file under experiments folder.

    dir_save = pre_training_save(config_file, exp_config)
    
    training_data = main(exp_config=exp_config)
    
    save_training(training_data, config_file, exp_config, dir_save)