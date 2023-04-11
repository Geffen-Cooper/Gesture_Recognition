import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna._callbacks import RetryFailedTrialCallback
from torch.utils.tensorboard import SummaryWriter

from model.PrototypeLoss import PrototypicalLoss
from model.datasets import load_nvgesture
from model.models import RNNModel, RNNFeatureExtractor
from utils.utils import ignore_unmatched_kwargs

@ignore_unmatched_kwargs
def train_prototype(loss_min_count=2, from_checkpoint=None, optimizer="AdamW", log_name="default", root_dir="ds", batch_size=128, epochs=200, ese=20, lr=1e-4, use_cuda=True, seed=42, subset=None, median_filter=False, augment_angles=False, model_type="RNN", model_params=None, save_model_ckpt=False):
    if model_params is None:
        model_params = {}
    assert batch_size > 25

    writer = SummaryWriter("runs/" + log_name + "_" + str(time.time()))
    # log training parameters
    print("===========================================")
    for k, v in zip(locals().keys(), locals().values()):
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
    train_loader, val_loader, test_loader = load_nvgesture(batch_size, seed, root_dir=root_dir, subset=subset, median_filter=median_filter, augment_angles=augment_angles)

    # load the model
    if model_type == "RNN":
        model = RNNFeatureExtractor(**model_params).to(device)
    else:
        raise NotImplementedError("Wrong model")

    loss = PrototypicalLoss(min_count=loss_min_count)

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
    else:
        best_val_acc = 0
        lowest_loss = 1e6


    model.train()
    model = model.to(device)
    batch_iter = 0

    num_epochs_worse = 0
    for epoch in range(epochs):

        for batch_idx, (data, target, _id) in enumerate(train_loader):
            data, target = data.to(device).float(), target.to(device)
            model.train()
            opt.zero_grad()

            embeddings = model(data)
            train_loss, train_acc = loss(embeddings, target)

            train_loss.backward()
            opt.step()
            batch_iter += 1

            # Log Stuff
            if (batch_iter % 10) == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)] train loss: {train_loss:.3f} train acc: {train_acc*100:.3f}%')
            writer.add_scalar("Metric/train_loss", train_loss, batch_iter)

        if num_epochs_worse >= ese:
            print(f"Stopping training because accuracy did not improve after {num_epochs_worse} epochs")
            break

        # evaluate on the validation set
        val_acc, val_loss = validate(model, val_loader, device, loss)

        writer.add_scalar("Metric/val_acc", val_acc, epoch)
        writer.add_scalar("Metric/val_loss", val_loss, epoch)

        print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)] train loss: {train_loss:.3f}, val acc: {val_acc*100:.3f}%, val loss: {val_loss:.3f}')

        # Check ESE
        if best_val_acc < val_acc:
            print("==================== best validation metric ====================")
            print("epoch: {}, val acc: {}, val loss: {}".format(epoch, val_acc, val_loss))
            best_val_acc = val_acc
            lowest_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            num_epochs_worse = 0
            # scheduler2.step()
        else:
            print(f"WARNING: {num_epochs_worse} num epochs without improving")
            num_epochs_worse += 1

    print("\n\n\n==================== TRAINING FINISHED ====================")
    print("Evaluating on test set")

    # load the best model
    # model = RNNModel(63, 256, 1, 25, 'cuda')
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    model.eval()
    test_acc, test_loss = validate(model, test_loader, device, loss)

    print('test acc: {:.3f}%, test loss: {:.3f}'.format(test_acc*100, test_loss))
    writer.add_scalar("Metric/test_acc", test_acc, epoch)
    writer.add_text("test_accuracy", f"{test_acc}")

    if not save_model_ckpt:
        os.remove(checkpoint_path)

    return test_acc, test_loss

def validate(model, val_loader, device, loss_fn):
    model.eval()
    model = model.to(device)

    val_loss = 0
    val_acc = 0
    total = 0

    with torch.no_grad():
        i = 0
        for idx, (data, target, id) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            out = model(data.float())
            batch_loss, batch_acc = loss_fn(out, target)
            val_loss += batch_loss
            val_acc += batch_acc
            total += len(target)
            i += 1

        # Compute loss and accuracy
        val_loss /= i
        val_acc /= i

        return val_acc, val_loss

if __name__ == '__main__':
    # model_params = dict(input_dim=63, hidden_dim=256, latent_dim=64, layer_dim=1, device='cuda')
    # params = {"loss": "CE", "from_checkpoint": None, "log_name": "prototype_v0", 'epochs': 400, 'ese': 20, 'use_cuda': True, 'seed': 42, 'subset': None, "augment_angles": True, "batch_size": 213, "lm_type": "w", "lr": 0.0068914481421098244, "median_filter": False, "save_model_ckpt": False, "model_type": "RNN", "optimizer": "AdamW", "resolution_method": "f", "sensor": "c", "use_clahe": 0, "video_mode": 1}
    # params['root_dir'] = f"../csvs/ds_L{params['lm_type']}_S{params['sensor']}_C{params['use_clahe']}_V{params['video_mode']}_R{params['resolution_method']}"
    # params['model_params'] = model_params
    #
    # test_acc, test_loss = train(**params)
    # print(f"My model: {test_acc*100:.4f}%, loss: {test_loss}")


    model_params = dict(input_dim=63, hidden_dim=256, latent_dim=64, layer_dim=1, device='cuda')
    params = dict(loss_min_count=2, from_checkpoint=None, optimizer="AdamW", log_name="default", root_dir="ds", batch_size=128, epochs=200, ese=20, lr=1e-4, use_cuda=True, seed=42, subset=None, median_filter=False, augment_angles=False, model_type="RNN", model_params=None, save_model_ckpt=False)

    def suggest_params(trial: optuna.Trial):
        # Model params
        model_params = dict()
        model_params['input_dim'] = 63
        model_params['hidden_dim'] = trial.suggest_int('hidden_dim', 64, 400)  # 256
        model_params['latent_dim'] = trial.suggest_int('latent_dim', 10, 128)  # 64
        model_params['layer_dim'] = trial.suggest_int('layer_dim', 1, 2) # 1
        model_params['rnn_type'] = trial.suggest_categorical('rnn_type', ["GRU", "LSTM"])
        model_params['rnn_use_last'] = trial.suggest_categorical('rnn_use_last', [False, True])
        model_params['device'] = 'cuda'

        # Datasets
        lm_type = trial.suggest_categorical("lm_type", ['n', 'wp', 'w'])
        sensor = "c"
        use_clahe = trial.suggest_categorical("use_clahe", [0, 1])
        video_mode = trial.suggest_categorical("video_mode", [0, 1])
        resolution_method = trial.suggest_categorical("resolution_method", ['z', 'f'])  # ['i', 'z', 'f']
        ds_name = f"ds_L{lm_type}_S{sensor}_C{use_clahe}_V{video_mode}_R{resolution_method}"


        # Train params
        params = dict()
        params['loss_min_count'] = trial.suggest_int('loss_min_count', 2, 4)
        params['from_checkpoint'] = None
        params['optimizer'] = trial.suggest_categorical('optimizer', ["AdamW", "Adam", "SGD"])
        params['log_name'] = f"proto_{trial.number}"
        params['root_dir'] = f"../csvs/{ds_name}"
        params['batch_size'] = trial.suggest_int('batch_size', 100, 256)
        params['epochs'] = 400
        params['ese'] = 20
        params['lr'] = trial.suggest_float('lr', 1e-3, 1e-2)
        params['use_cuda'] = True
        params['seed'] = 42
        params['subset'] = None
        params['model_type'] = "RNN"
        params['model_params'] = model_params
        params['median_filter'] = False
        params['augment_angles'] = trial.suggest_categorical('augment_angles', [True, False]) if lm_type != 'wp' else False
        params['save_model_ckpt'] = False

        return params


    def objective(trial):
        params = suggest_params(trial)
        try:
            test_acc, test_loss = train_prototype(**params)
            torch.cuda.empty_cache()
        except Exception as e:
            for i in range(10):
                print()
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            logger.error(str(e), exc_info=True)
            return float('nan')
        return test_loss, test_acc

    ##### SQL Server #####
    user = "admin"
    password = "plzdonthackme"
    url = "optuna.czje4evrsrqy.us-east-2.rds.amazonaws.com"
    database = "db1"
    ######################

    STUDY_NAME = "prototype_study_v0"

    storage = optuna.storages.RDBStorage(
        # url="sqlite:///:memory:",
        url=f"mysql://{user}:{password}@{url}/{database}",
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
    )

    print(f"SQL URL: mysql://{user}:{password}@{url}/{database}")
    print()

    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, constant_liar=True)
    study = optuna.create_study(study_name=STUDY_NAME, sampler=sampler, storage=storage, load_if_exists=True, directions=['minimize', 'maximize'])
    study.optimize(objective, timeout=3600 * 10, gc_after_trial=True)