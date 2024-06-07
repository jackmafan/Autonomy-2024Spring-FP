#built in libraries
import os
import time
import argparse
from datetime import datetime
from tqdm import tqdm

#torch libraries
import torch
import torch.nn as nn

#self defined libraies

from dataset import build_loader
from model import TSC_M,TSC_MA
from util import write_result_log,plot_learning_curve

def train(data_set_dir:str,exp_dir:str, lr:float, optim:str, bs:int, epochs:int, device:torch.device, model:nn.Module,cropROI:bool):
    train_loader,val_loader = build_loader(data_set_dir=data_set_dir,batch_size = bs,cropROI=cropROI)
    
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    if optim=='Adam':
        optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
    else:
        raise ValueError('Unknown type of optimizers')
    
    
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    best_acc = 0.0
    
    for epoch in range(epochs):
        ####################
        ##### TRAINING #####
        ####################
        train_start_time = time.time()
        train_loss, train_correct = 0.0, 0.0
        
        model.train()
        print(f'epochs: {epoch+1} / {epochs}')
        for batch, data in enumerate(tqdm(train_loader)):
            # Data loading.
            images, labels = data['images'].to(device), data['labels'].to(device) # 
            pred = model(images)

            # Calculate loss.
            loss = loss_func(pred, labels)

            # Backprop. (update model parameters)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Evaluate.
            train_correct += torch.sum(torch.argmax(pred, dim=1) == labels)
            train_loss += loss.item()
            
        # Print training result
        train_time = time.time() - train_start_time
        train_acc = train_correct / len(train_loader.dataset)
        train_loss /= len(train_loader)
        train_acc_list.append(train_acc.cpu())
        train_loss_list.append(train_loss)
        print()
        print(f' {train_time:.2f} sec(s) Train Acc: {train_acc:.5f} | Train Loss: {train_loss:.5f}')
        ######################
        ##### VALIDATION #####
        ######################
        model.eval()
        with torch.no_grad():
            val_start_time = time.time()
            val_loss,val_correct = 0.0, 0.0
            
            for batch, data in enumerate(tqdm(val_loader)):
                # Data loading.
                images, labels = data['images'].to(device), data['labels'].to(device) # (batch_size, 3, 32, 32), (batch_size)
                
                # Forward pass. input: (batch_size, 3, 32, 32), output: (batch_size, 10)
                pred = model(images)
                
                # Calculate loss.
                loss = loss_func(pred, labels)

                # Evaluate.
                val_correct += torch.sum(torch.argmax(pred, dim=1) == labels)
                val_loss += loss.item()
    
            ######################### TODO End ##########################
            
        # Print validation result
        val_time = time.time() - val_start_time
        val_acc = val_correct / len(val_loader.dataset)
        val_loss /= len(val_loader)
        val_acc_list.append(val_acc.cpu())
        val_loss_list.append(val_loss)
        print()
        print(f'{val_time:.2f} sec(s) Val Acc: {val_acc:.5f} | Val Loss: {val_loss:.5f}')
        

        ##### WRITE LOG #####
        is_better = val_acc >= best_acc
        epoch_time = train_time + val_time
        write_result_log(os.path.join(exp_dir, 'result_log.txt'), epoch, epoch_time, train_acc, val_acc, train_loss, val_loss, is_better,epochs)

        ##### SAVE THE BEST MODEL #####
        if is_better:
            print(f'[{epoch + 1}/{epochs}] Save best model to {exp_dir} ...')
            torch.save(model.state_dict(), os.path.join(exp_dir, 'model_best.pth'))
            best_acc = val_acc

        ##### PLOT LEARNING CURVE #####
        ##### TODO: check plot_learning_curve() in this file #####
        current_result_lists = {
            'train_acc': train_acc_list,
            'train_loss': train_loss_list,
            'val_acc': val_acc_list,
            'val_loss': val_loss_list
        }
        plot_learning_curve(exp_dir, current_result_lists)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', help='dataset directory', type=str, default='GTSRB')
    parser.add_argument('--lr',help='learning rate',type=float,default=0.001)
    parser.add_argument('--drop_out_prob',help='hyper parameter for dropout layers',type=float,default=0.1)
    parser.add_argument('--optimizer',help='type of optimizer',type=str,default='Adam')
    parser.add_argument('--bs',help='batch size',type=int,default=50)
    parser.add_argument('--epochs',help='training epochs',type=int,default=70)
    parser.add_argument('--noROI',action='store_true',default=False)
    args = parser.parse_args()
    

    exp_name =  'TSC_M'+ datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
    exp_dir = os.path.join('./experiment', exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TSC_M(args.drop_out_prob)
    
    train(data_set_dir=args.dataset_dir,
          exp_dir = exp_dir,
          lr=args.lr,
          optim=args.optimizer,
          bs=args.bs,
          epochs=args.epochs,
          device=device,
          model=model,
          cropROI = not args.noROI
          )
    
    
    
    
    
