'''
    This file takes in command line arguments for the training parameters
    and runs a training/test function. In general, this code tries to be agnostic
    to the model and dataset but assumes a standard supervised training setup.
'''

import argparse
import functools
import glob
import inspect
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
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from tqdm import tqdm
import time
from datasets import *
from models import *
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import optuna
from utils import utils

np.set_printoptions(linewidth=np.nan)



@utils.ignore_unmatched_kwargs
def train(loss,from_checkpoint,optimizer,log_name,root_dir,batch_size,epochs,ese,lr,use_cuda,seed,subset, median_filter, augment_angles,
          model_type, model_hidden_dim_size_rnn, save_model_ckpt, model_hidden_dim_size_trans=None, model_num_layers_trans=None, model_num_heads_trans=None, model_lambda=None):

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
        model_params = dict(input_dim=63, hidden_dim=model_hidden_dim_size_rnn, layer_dim=1, output_dim=25, device=device)
        model = RNNModel(**model_params).to(device)
    elif model_type == "AttentionRNN":
        model_params = dict(input_dim=63, hidden_dim=model_hidden_dim_size_rnn, layer_dim=1, output_dim=25, device=device,fc=True)
        model = AttentionRNNModel(**model_params).to(device)
    elif model_type == "Transformer":
        model_params = dict(input_dim=63, num_classes=25, num_heads=model_num_heads_trans, hidden_dim=model_hidden_dim_size_trans, num_layers=model_num_layers_trans)
        model = TransformerClassifier(**model_params).to(device)
    else:
        model = model_lambda()

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

    print("Generate Embeddings")
    hidden_size = model_hidden_dim_size_rnn if (model_type == "RNN" or model_type == "AttentionRNN") else model_hidden_dim_size_trans
    generate_embeddings(test_loader,model,writer,model_type,device,hidden_size)

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


def generate_embeddings(test_loader,model,tb_writer,model_type,device,hidden_size):
    model.eval().to(device)

    # get batch from test loader
    batch_imgs, batch_labels, idxs = next(iter(test_loader))
    batch_imgs = batch_imgs.to(device)

    # store output embedding for each timestep, so we can average
    activation = {}

    if model_type == "RNN" or model_type == "AttentionRNN":
        # forward hook
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = input[0].detach().to('cpu')
            return hook

        # register the forward hook as input to fully connected layer
        model.fc.register_forward_hook(get_activation('emb'))

        # forward pass on the batch
        with torch.no_grad():
            preds = model(batch_imgs)

            # get the embedings and divide by sequence length
            embds = activation['emb'].squeeze(1).to('cpu')

            # get the labels
            # halfway = len(test_loader.dataset.datasets[0])
            batch_label_strings = []
            for idx,label in enumerate(batch_labels):
                batch_label_strings.append(str(label.item()))
                # if idx >= halfway:
                #     dir = os.path.basename(Path(test_loader.dataset.datasets[1].img_paths[idx-halfway]).parents[1])
                #     batch_label_strings.append(str(label.item())+"_"+dir)
                # else:
                #     dir = os.path.basename(Path(test_loader.dataset.datasets[0].img_paths[idx-halfway]).parents[1])
                #     batch_label_strings.append(str(label.item())+"_"+dir)

            # store embeddings to tensorboard
            tb_writer.add_embedding(embds,metadata=batch_label_strings)

    elif model_type == "Transformer":
        # forward hook
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = input[0].detach().to('cpu')
            return hook

        # register the forward hook as input to fully connected layer
        model.output_layer.register_forward_hook(get_activation('emb'))

        # forward pass on the batch
        with torch.no_grad():
            preds = model(batch_imgs)
            embds = activation['emb'].squeeze(1).to('cpu')

            # get the labels
            batch_label_strings = []
            for idx,label in enumerate(batch_labels):
                batch_label_strings.append(str(label.item()))

            # store embeddings to tensorboard
            tb_writer.add_embedding(embds,metadata=batch_label_strings)


# ===================================== Main =====================================
if __name__ == "__main__":

    train_params = {'loss': "CE", 'from_checkpoint': None, 'optimizer': "AdamW", 'log_name': "collected", 'root_dir': "../csvs/collected_data",
                    'batch_size': 64, 'epochs': 50, 'ese': 5, 'lr': 0.0015, 'use_cuda': True, 'seed': 42, 'subset': tuple(np.arange(25)), 'median_filter': False, 'augment_angles': True,
                    'model_type': "AttentionRNN", 'model_hidden_dim_size_rnn': 256, 'model_hidden_dim_size_trans': 276, 'save_model_ckpt': True,
                    'model_num_layers_trans': 1, 'model_num_heads_trans': 6}

    train(**train_params)
