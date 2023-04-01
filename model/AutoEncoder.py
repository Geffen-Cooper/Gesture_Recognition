import glob
import os.path
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from torch.utils import data
import pandas as pd
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(63, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 20))

        self.decoder = nn.Sequential(
            nn.Linear(20, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 63),
            nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AEDataset(data.Dataset):
    def __init__(self, csvs_dir, cache={}, validate=False, csv_len=80):
        super(AEDataset, self).__init__()
        self.csvs = list(Path(csvs_dir).glob("*.csv"))

        assert csv_len is not None, "csv_len must not be None"
        self.CSV_LEN = csv_len
        self.cache = cache

        if validate:
            for csv in tqdm(self.csvs):
                df = pd.read_csv(csv)
                assert len(df) == self.CSV_LEN

    def __len__(self):
        return self.CSV_LEN * len(self.csvs)

    def __getitem__(self, item):
        csv_idx = item // self.CSV_LEN
        if csv_idx in self.cache:
            df = self.cache[csv_idx]
        else:
            df = pd.read_csv(self.csvs[csv_idx])
            self.cache[csv_idx] = df
        return torch.Tensor(df.loc[item % self.CSV_LEN, :].values)


if __name__ == '__main__':
    ds_cache = {}

    train_ds1 = AEDataset(csvs_dir='../data/nvGesture_v1/lm_train', cache=ds_cache)
    train_ds2 = AEDataset(csvs_dir='../data/ae_unlabled_ds', cache=ds_cache, csv_len=3616)  # Data I collected
    train_ds = ConcatDataset([train_ds1, train_ds2])

    test_ds = AEDataset(csvs_dir='../data/nvGesture_v1/lm_test', cache=ds_cache)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # instantiate the model
    model = Autoencoder().to(device)

    # define the loss function
    criterion = nn.MSELoss()

    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # define the number of epochs
    num_epochs = 100
    batch_size = 32

    train_dl = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_dl = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # set up early stopping
    max_rounds_wo_improvement = 5
    min_test_loss = float('inf')
    epochs_without_improvement = 0

    # iterate over the dataset for the specified number of epochs
    for epoch in tqdm(range(num_epochs), desc='Epoch', leave=False):
        train_loss = 0.0
        test_loss = 0.0

        # train the model
        model.train()
        for data in tqdm(train_dl, desc='Batch', leave=False):
            # get the inputs and wrap them in variables
            inputs = data
            inputs = inputs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()

        # evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            for data in tqdm(test_dl, desc='Test', leave=False):
                inputs = data
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                test_loss += loss.item()

        # print the average losses for the epoch
        epoch_train_loss = train_loss / len(train_dl)
        epoch_test_loss = test_loss / len(test_dl)
        print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_train_loss, epoch_test_loss))

        # check if the test loss has improved
        if epoch_test_loss < min_test_loss:
            min_test_loss = epoch_test_loss
            torch.save(model.state_dict(), 'models/ae_model.pt')
            epochs_without_improvement = 0
        else:
            # increment the epochs without improvement counter
            epochs_without_improvement += 1

            # check if early stopping criterion has been met
            if epochs_without_improvement >= max_rounds_wo_improvement:
                print('Early stopping criterion met, stopping training.')
                break
