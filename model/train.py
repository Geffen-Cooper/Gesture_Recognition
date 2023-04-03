'''
    This file takes in command line arguments for the training parameters
    and runs a training/test function. In general, this code tries to be agnostic
    to the model and dataset but assumes a standard supervised training setup.
'''

import argparse
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

np.set_printoptions(linewidth=np.nan)


def train(model, train_loader, val_loader, test_loader, device, loss_fn, optimizer, args, **kwargs):

    # continue training a model
    if args.checkpoint != "None":
        # init tensorboard
        args.log_name = args.log_name + "resume"
        writer = SummaryWriter("runs/" + args.log_name)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("resume from:", checkpoint['val_metric'])
        best_val_acc = checkpoint['val_metric']

        # Log kwargs
        for k, v in kwargs.items():
            if type(v) == dict:
                for vk, vv in v:
                    writer.add_text(f"{k}/{vk}", str(vv))
            else:
                writer.add_text(f"{k}", str(v))
    else:
        writer = SummaryWriter("runs/" + args.log_name)
        best_val_acc = 0

    model.train()
    model = model.to(device)
    batch_iter = 0

    # how many epochs the validation loss did not decrease, 
    # used for early stopping
    num_epochs_worse = 0
    for e in range(args.epochs):
        if num_epochs_worse == args.ese:
            break
        for batch_idx, (data, target, id) in enumerate(train_loader):
            # stop training, run on the test set
            if num_epochs_worse == args.ese:
                break

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            model.train()

            output = model(data)

            loss = loss_fn(output, target)
            writer.add_scalar("Metric/train_" + args.loss, loss, batch_iter)
            loss.backward()
            optimizer.step()

            if (batch_iter % 10) == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}'.format(
                    e, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss))  # , scheduler1.get_last_lr()[0]))
            # if batch_idx > 0:
            #     scheduler1.step()
            batch_iter += 1
        # scheduler2.step()

        if num_epochs_worse == args.ese:
            break

        # evaluate on the validation set
        val_acc, val_loss = validate(model, val_loader, device, loss_fn)

        writer.add_scalar("Metric/val_acc", val_acc, e)
        writer.add_scalar("Metric/val_loss", val_loss, e)

        print('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}, val acc: {:.3f}, val loss: {:.3f}'.format(
            e, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss, val_acc, val_loss))
        # scheduler1.step()

        if best_val_acc < val_acc:
            print("==================== best validation metric ====================")
            print("epoch: {}, val acc: {}".format(e, val_acc))
            best_val_acc = val_acc
            torch.save({
                'epoch': e + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, 'models/' + args.log_name + '.pth')
            num_epochs_worse = 0
        # scheduler2.step()
        else:
            num_epochs_worse += 1

    # evaluate on test set
    print("\n\n\n==================== TRAINING FINISHED ====================")
    print("Evaluating on test set")

    # load the best model
    # model = RNNModel(63, 256, 1, 25, 'cuda')
    model.load_state_dict(torch.load('models/' + args.log_name + '.pth')['model_state_dict'])
    model.eval()
    test_acc, test_loss = validate(model, test_loader, device, loss_fn)

    print('test acc: {:.3f}, test loss: {:.3f}'.format(
        test_acc, test_loss))


def validate(model, val_loader, device, loss_fn):
    model.eval()
    model = model.to(device)

    val_loss = 0
    num_correct = 0
    total = 0
    preds = torch.zeros(len(test_loader.dataset))
    gt = torch.zeros(len(test_loader.dataset))

    with torch.no_grad():
        i = 0
        for idx, (data, target, id) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            out = model(data)
            val_loss += loss_fn(out, target)
            num_correct += sum(torch.argmax(out, dim=1) == target)
            total += len(target)
            i += 1

            preds[idx * test_loader.batch_size:idx * test_loader.batch_size + len(target)] = torch.argmax(out, dim=1)
            gt[idx * test_loader.batch_size:idx * test_loader.batch_size + len(target)] = target

        # Compute loss and accuracy
        val_loss /= i
        val_acc = num_correct / total

        # cm = confusion_matrix(gt, preds)
        # print(cm)
        return val_acc, val_loss


# ===================================== Command Line Arguments =====================================
def parse_args():
    parser = argparse.ArgumentParser(description="Training and Evaluation")

    # logging details
    parser.add_argument('--loss', type=str, default='CE', help='loss function')
    parser.add_argument('--checkpoint', type=str, default='None', help='checkpoint to resume from')
    parser.add_argument('--opt', type=str, default='Adam', help='optimizer')
    parser.add_argument('--log_name', type=str, default='default', help='checkpoint file name')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--ese', type=int, default=5, metavar='N',
                        help='early stopping epochs')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    print(args)

    return args


# ===================================== Main =====================================
if __name__ == "__main__":

    # get arguments
    args = parse_args()

    # torch.manual_seed(args.seed)

    # setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load datasets
    train_loader, val_loader, test_loader = load_nvgesture(args.batch_size, args.seed, root_dir='../data/nvGesture_v1 ')

    # load the model
    model_params = dict(input_dim=63, hidden_dim=256, layer_dim=1, output_dim=25, device='cuda')
    model = RNNModel(**model_params).to(device)
    # model_params = dict(input_dim=63, num_classes=25, num_heads=4, hidden_dim=32, num_layers=2)
    # model = TransformerClassifier(**model_params).to(device)

    # set loss function
    if args.loss == "CE":
        loss = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError()

    # set optimizer
    if args.opt == "SGD":
        opt = torch.optim.SGD(params=model.parameters(), lr=args.lr)
    elif args.opt == "Adam":
        opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError()

    train(model, train_loader, val_loader, test_loader, device, loss, opt, args, model_params=model_params)
