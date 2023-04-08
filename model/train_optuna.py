import glob
import os
import re
from pathlib import Path

import numpy as np
import optuna
import logging
from optuna.storages import RetryFailedTrialCallback

from train import train


def suggest_params(trial: optuna.Trial):
    # Model
    model_type = trial.suggest_categorical("model_type", ["RNN", "Transformer"])
    model_hidden_dim_size_rnn = None
    model_hidden_dim_size_trans = None

    if model_type == "RNN":
        model_hidden_dim_size_rnn = trial.suggest_int("model_hidden_dim_size_rnn", low=64, high=600)
    else:
        model_hidden_dim_size_trans = trial.suggest_int("model_hidden_dim_size_trans_multiplier", low=4, high=16) * 4

    # Training
    save_model_ckpt = False
    loss = "CE"
    from_checkpoint = None
    epochs = 100
    ese = 20
    use_cuda = True
    seed = 42
    opt = "Adam"
    batch_size_idx = trial.suggest_int("batch_size_idx", low=0, high=3)
    batch_size = (8, 16, 32, 64)[batch_size_idx]

    lr_idx = trial.suggest_int("lr_idx", low=0, high=2)
    lr = (0.001, 0.005, 0.01)[lr_idx]

    # Datasets
    subsets_type = 'all'
    lm_type = trial.suggest_categorical("lm_type", ['n', 'wp', 'w'])
    sensor = trial.suggest_categorical("sensor", ['c', 'd'])
    use_clahe = trial.suggest_categorical("use_clahe", [0, 1])
    video_mode = 0
    resolution_method = trial.suggest_categorical("resolution_method", ['i', 'z', 'f'])
    median_filter = trial.suggest_categorical("median_filter", [True, False])

    if lm_type == 'wp':
        augment_angles = False
    else:
        augment_angles = trial.suggest_categorical("augement_angles", [True, False])

    ds_regex = f"ds_L{lm_type}_S{sensor}_C{use_clahe}_V{video_mode}_R{resolution_method}"
    root_dir = f"../csvs/{ds_regex}"
    assert Path(root_dir).is_dir(), f"root dir: {root_dir} doesn't exist!"

    subsets_lst = [tuple(np.arange(25)), (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)]
    subset = subsets_lst[0 if subsets_type == "all" else 1]

    log_name = f"optuna_MODEL{model_type[0].lower()}_HIDDEN_{model_hidden_dim_size_rnn if model_hidden_dim_size_rnn else model_hidden_dim_size_trans}_O{opt[0].lower()}_B{batch_size}_S{subsets_type}_M{int(median_filter)}_A{int(augment_angles)}_LR{lr:.5f}_{Path(root_dir).stem}"

    train_params = {'loss': loss, 'from_checkpoint': from_checkpoint, 'optimizer': opt, 'log_name': log_name, 'root_dir': root_dir,
                    'batch_size': batch_size, 'epochs': epochs, 'ese': ese, 'lr': lr, 'use_cuda': use_cuda, 'seed': seed, 'subset': subset, 'median_filter': median_filter, 'augment_angles': augment_angles,
                    'model_type': model_type, 'model_hidden_dim_size_rnn': model_hidden_dim_size_rnn, 'model_hidden_dim_size_trans': model_hidden_dim_size_trans, 'save_model_ckpt': save_model_ckpt}
    return train_params


def objective(trial):
    params = suggest_params(trial)
    try:
        val_acc, val_loss = train(**params)
    except Exception as e:
        for i in range(10):
            print()
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.error(str(e), exc_info=True)
        return float('nan')
    return val_loss


if __name__ == '__main__':
    ##### SQL Server #####
    # admin, plzdonthackme
    user = "admin"
    password = "plzdonthackme"
    url = "optuna.czje4evrsrqy.us-east-2.rds.amazonaws.com"
    database = "db1"
    ######################

    STUDY_NAME = "actual_actual_initial_study_v2"

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
    study = optuna.create_study(study_name=STUDY_NAME, sampler=sampler, storage=storage, load_if_exists=True)
    study.optimize(objective, timeout=3600 * 10, gc_after_trial=True)
