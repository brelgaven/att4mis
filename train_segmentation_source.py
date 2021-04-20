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

def train_segmentation_network(exp_config, model, trainable_parameters, loader_train, loader_val, path_to_save_model):

    optimizer = optim.Adam(trainable_parameters, lr = 1e-4, weight_decay = 0.0) # Define optimizer - update all paramaters

    criterion = nn.CrossEntropyLoss()
    
    best_loss_val = 0
    
    for epoch in range(exp_config.number_of_epoch):
        
        model.train() # Switch on training mode            
        running_loss_train = 0.0
        running_dice_train = 0.0
        counter = 0
        for data, target in loader_train:
            data, target = data.cuda(), torch.squeeze(target).cuda().long()
            
            optimizer.zero_grad()
            pred_logits, pred_softmax, pred_argmax = model(data)
            
            loss = criterion(pred_logits, target)
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
                
            loss = criterion(pred_logits, target)
            dice_each_class = met.f1_score(torch.flatten(target).detach().cpu().numpy(), torch.flatten(pred_argmax).detach().cpu().numpy(), average = None)
            running_dice_val += np.mean(dice_each_class)
            running_loss_val += loss.item()
            
            counter += 1
        running_loss_val = running_loss_val / counter
        running_dice_val = running_dice_val / counter
        
        if epoch == 0 or best_loss_val > running_loss_val:
            best_loss_val = running_loss_val
            torch.save(model.state_dict(), ('%s/%s_model_segmentation%s.pth')%(path_to_save_model, exp_config.data_identifier_source, exp_config.train_id))
            print('epoch:%d - loss_tr: %.10f loss_val: %.10f - dice_tr: %.10f dice_val: %.10f - Last Saved' %
                  (epoch, running_loss_train, running_loss_val, running_dice_train, running_dice_val))
        else:
            print('epoch:%d - loss_tr: %.10f loss_val: %.10f - dice_tr: %.10f dice_val: %.10f' %
                  (epoch, running_loss_train, running_loss_val, running_dice_train, running_dice_val))

def main(exp_config):

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
    train_segmentation_network(exp_config, model, model.parameters(), source_train_loader, source_val_loader, exp_config.path_to_save_trained_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment config file")
    args = parser.parse_args()

    config_file = args.EXP_PATH
    config_module = config_file.split('/')[-1].rstrip('.py')
        
    exp_config = SourceFileLoader(config_module, config_file).load_module() # exp_config stores configurations in the given config file under experiments folder.

    main(exp_config=exp_config)

