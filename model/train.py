'''
    This file takes in command line arguments for the training parameters
    and runs a training/test function. In general, this code tries to be agnostic
    to the model and dataset but assumes a standard supervised training setup.
'''

import argparse
import glob
import math
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from datasets import *
from models import *
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import optuna

np.set_printoptions(linewidth=np.nan)


def train(loss,from_checkpoint,optimizer,log_name,root_dir,batch_size,epochs,ese,lr,use_cuda,seed,subset, median_filter, augment_angles,
          model_type, model_hidden_dim_size_rnn, model_hidden_dim_size_trans, save_model_ckpt, model_num_layers_trans, model_num_heads_trans):

    writer = SummaryWriter("runs/" + log_name+"_"+str(time.time()))
    # log training parameters
    print("===========================================")
    for k,v in zip(locals().keys(),locals().values()):
        writer.add_text(f"locals/{k}", f"{v}")
        print(f"locals/{k}", f"{v}")
    print("===========================================")

    # ================== parse the arguments ==================
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Checkpoint path
    checkpoint_path = 'models/' + log_name + '.pth'

    # setup device
    use_cuda = use_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load datasets
    train_loader, val_loader, test_loader = load_nvgesture(batch_size, seed, root_dir=root_dir,subset=subset, median_filter=median_filter, augment_angles=augment_angles)

    # load the model
    if model_type == "RNN":
        model_params = dict(input_dim=63, hidden_dim=model_hidden_dim_size_rnn, layer_dim=1, output_dim=25, device='cuda')
        model = RNNModel(**model_params).to(device)
    else:
        model_params = dict(input_dim=63, num_classes=25, num_heads=model_num_heads_trans, hidden_dim=model_hidden_dim_size_trans, num_layers=model_num_layers_trans)
        model = TransformerClassifier(**model_params).to(device)

    # set loss function
    if loss == "CE":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError()

    # set optimizer
    if optimizer == "SGD":
        opt = torch.optim.SGD(params=model.parameters(), lr=lr)
    elif optimizer == "Adam":
        opt = torch.optim.Adam(params=model.parameters(), lr=lr)
    elif optimizer == "AdamW":
        opt = torch.optim.AdamW(params=model.parameters(), lr=lr)
    else:
        raise NotImplementedError()

    # continue training a model
    if from_checkpoint != None:
        checkpoint = torch.load(from_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("resume from:", checkpoint['val_acc'])
        best_val_acc = checkpoint['val_acc']

        # # Log kwargs
        # for k, v in kwargs.items():
        #     if type(v) == dict:
        #         for vk, vv in v:
        #             writer.add_text(f"{k}/{vk}", str(vv))
        #     else:
        #         writer.add_text(f"{k}", str(v))
    else:
        best_val_acc = 0
        lowest_loss = 1e6



    # ================== training loop ==================
    model.train()
    model = model.to(device)
    batch_iter = 0

    # how many epochs the validation loss did not decrease, 
    # used for early stopping
    num_epochs_worse = 0
    for e in range(epochs):
        if num_epochs_worse == ese:
            break
        for batch_idx, (data, target, _id) in enumerate(train_loader):
            # stop training, run on the test set
            if num_epochs_worse == ese:
                break

            data, target = data.to(device), target.to(device)

            opt.zero_grad()
            model.train()

            output = model(data.float())

            train_loss = loss_fn(output, target)
            writer.add_scalar("Metric/train_" + loss, train_loss, batch_iter)
            train_loss.backward()
            opt.step()

            if (batch_iter % 10) == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}'.format(
                    e, batch_idx * len(data), len(train_loader.dataset),
                       100.0 * batch_idx / len(train_loader), train_loss))  # , scheduler1.get_last_lr()[0]))
            # if batch_idx > 0:
            #     scheduler1.step()
            batch_iter += 1
        # scheduler2.step()

        if num_epochs_worse == ese:
            print(f"Stopping training because accuracy did not improve after {num_epochs_worse} epochs")
            break

        # evaluate on the validation set
        val_acc, val_loss = validate(model, val_loader, device, loss_fn)

        writer.add_scalar("Metric/val_acc", val_acc, e)
        writer.add_scalar("Metric/val_loss", val_loss, e)

        print('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}, val acc: {:.3f}, val loss: {:.3f}'.format(
            e, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), train_loss, val_acc, val_loss))
        # scheduler1.step()

        if best_val_acc < val_acc:
        # if lowest_loss > val_loss:
            print("==================== best validation metric ====================")
            print("epoch: {}, val acc: {}, val loss: {}".format(e, val_acc, val_loss))
            best_val_acc = val_acc
            lowest_loss = val_loss
            torch.save({
                'epoch': e + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            num_epochs_worse = 0
        # scheduler2.step()
        else:
            print(f"WARNING: {num_epochs_worse} num epochs without improving")
            num_epochs_worse += 1

    # evaluate on test set
    print("\n\n\n==================== TRAINING FINISHED ====================")
    print("Evaluating on test set")

    # load the best model
    # model = RNNModel(63, 256, 1, 25, 'cuda')
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    model.eval()
    test_acc, test_loss = validate(model, test_loader, device, loss_fn)

    print('test acc: {:.3f}, test loss: {:.3f}'.format(test_acc, test_loss))
    writer.add_scalar("Metric/test_acc", test_acc, e)
    writer.add_text("test_accuracy",f"{test_acc}")

    if not save_model_ckpt:
        os.remove(checkpoint_path)

    return test_acc, test_loss


def validate(model, val_loader, device, loss_fn):
    model.eval()
    model = model.to(device)

    val_loss = 0
    num_correct = 0
    total = 0
    preds = torch.zeros(len(val_loader.dataset))
    gt = torch.zeros(len(val_loader.dataset))

    with torch.no_grad():
        i = 0
        for idx, (data, target, id) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            out = model(data.float())
            val_loss += loss_fn(out, target)
            num_correct += sum(torch.argmax(out, dim=1) == target)
            total += len(target)
            i += 1

            preds[idx * val_loader.batch_size:idx * val_loader.batch_size + len(target)] = torch.argmax(out, dim=1)
            gt[idx * val_loader.batch_size:idx * val_loader.batch_size + len(target)] = target

        # Compute loss and accuracy
        val_loss /= i
        val_acc = num_correct / total

        # cm = confusion_matrix(gt, preds)
        # print(cm)
        return val_acc, val_loss


# ===================================== Main =====================================
if __name__ == "__main__":
    # training parameters
    losses = ["CE"]
    from_checkpoint = [None]
    opts = ["Adam","SGD"]
    log_name = ["baseline_"]
    root_dirs = ["../csvs/front","../csvs/top"]
    batch_sizes = [32,8,16,64]
    epochs = [100]
    ese = [20]
    lr = [0.001,0.01]
    use_cuda = [True,False]
    seed = [42]
    subsets = [list(np.arange(25)),[0,2,4,6,8,10,12,14,16,18,20,22,24]]

    for root_dir in root_dirs:
        train_params = {'loss':losses[0],'from_checkpoint':from_checkpoint[0],\
                        'optimizer':opts[0],'log_name':log_name[0]+os.path.basename(root_dir),'root_dir':root_dir,\
                        'batch_size':batch_sizes[0],'epochs':epochs[0],'ese':ese[0],'lr':lr[0],\
                        'use_cuda':use_cuda[0],'seed':seed[0],'subset':subsets[1]}

        train(**train_params)
