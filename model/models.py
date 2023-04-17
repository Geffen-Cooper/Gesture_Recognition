import abc
import functools
import warnings
from abc import ABC
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


# Create RNN Model
class RNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(RNNModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.device = device

        # RNN
        # self.rnn = torch.nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        # Readout layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros (# layers, batch size, feature size)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)

        # output shape is (batch size, sequence length, feature dim)
        # it stores the hidden state for all timesteps
        output, (hn, cn) = self.rnn(x, (h0, c0))

        # average over sequence
        avg_hidden = torch.mean(output, dim=1)
        return self.fc(avg_hidden)


# Create RNN Model with attention
class AttentionRNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device, fc=False):
        super(AttentionRNNModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.device = device

        # RNN
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # attention
        self.attention = torch.nn.Linear(hidden_dim, 1)

        if fc == True:
            self.fc = torch.nn.Linear(hidden_dim, output_dim)
        else:
            self.fc = None

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)

        # (batch_size,sequence length) since we want a weight for each hidden state in the sequence
        attention_weights = torch.zeros((x.shape[0], x.shape[1])).to(self.device)

        # output shape is (batch size, sequence length, feature dim)
        output, (hn, cn) = self.rnn(x, (h0, c0))

        # for each time step, get the attention weight (do this over the batch)
        for i in range(x.shape[1]):
            attention_weights[:, i] = self.attention(output[:, i, :]).view(-1)
        attention_weights = F.softmax(attention_weights, dim=1)

        # apply attention weights for each element in batch
        attended = torch.zeros(output.shape[0], output.shape[2]).to(self.device)
        for i in range(x.shape[0]):
            attended[i, :] = attention_weights[i] @ output[i, :, :]

        if self.fc is not None:
            return self.fc(attended)
        else:
            return attended
class ModelWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ModelWrapper, self).__init__()
        self.fc = nn.LazyLinear(25)
        self.model = RNNFeatureExtractor(*args, **kwargs)
        self.fc.to(self.model.device)
    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x

    def state_dict(self):
        return self.model.state_dict()


class RNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, layer_dim, rnn_type, rnn_use_last, device):
        super(RNNFeatureExtractor, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Use last output instead of average output
        self.rnn_use_last = rnn_use_last

        self.device = device
        self.rnn_type = rnn_type

        # RNN
        if rnn_type == "GRU":
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)
        elif rnn_type == "AttentionRNN":
            self.rnn = AttentionRNNModel(input_dim, hidden_dim, layer_dim, latent_dim, device, fc=False)
        else:
            raise NotImplementedError()
        self.embedder = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = x.float()
        if self.rnn_type != "AttentionRNN":
            rnn_out, _ = self.rnn(x)
            if not self.rnn_use_last:
                rnn_mean = torch.mean(rnn_out, dim=1)
            else:
                rnn_mean = rnn_out[:, -1, :]
        else:
            rnn_mean = self.rnn(x)

        output = self.embedder(rnn_mean)
        return output


class FewShotModel(nn.Module, ABC):
    def __init__(self, input_dim, hidden_dim, latent_dim, layer_dim, rnn_type, rnn_use_last, checkpoint_file=None, device='cuda'):
        super(FewShotModel, self).__init__()
        self.checkpoint_file = checkpoint_file
        self.device = device
        self.fe_model = RNNFeatureExtractor(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, layer_dim=layer_dim, rnn_type=rnn_type, rnn_use_last=rnn_use_last,
                                            device=device).to(device)

        if self.checkpoint_file:
            checkpoint = torch.load(self.checkpoint_file)
            print(f"Loading RNNFeatureExtractor with epoch: {checkpoint['epoch']}, val_acc: {checkpoint['val_acc']}, val_loss: {checkpoint['val_loss']}")
            self.fe_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            warnings.warn("Checkpoint not specified, loading untrained model!")

    def get_embeddings(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            embeddings = self.fe_model(x)
        return embeddings

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def num_classes(self):
        pass

    @abc.abstractmethod
    def delete_centroid_if_exists(self, class_num):
        pass

    @abc.abstractmethod
    def add_data_for_class(self, data, class_num):
        pass

    @abc.abstractmethod
    def do_train(self):
        pass


class SklearnModel(FewShotModel):

    def __init__(self, input_dim, hidden_dim, latent_dim, layer_dim, rnn_type, rnn_use_last, checkpoint_file=None, device='cuda', model=None):
        super().__init__(input_dim, hidden_dim, latent_dim, layer_dim, rnn_type, rnn_use_last, checkpoint_file=checkpoint_file, device='cuda')
        self.device = device

        assert model is not None
        self.model = model
        self.model_invalid = False
        self.data_by_class = {}
        self.class_idx2class_num = {}

    def forward(self, x):
        assert not self.model_invalid
        embeddings = self.get_embeddings(x).cpu().numpy()
        pred_proba = self.model.predict_proba(embeddings)

        assert functools.reduce(lambda i, j: i and j, map(lambda m, k: m == k, self.get_class_nums_sorted(), list(self.model.classes_)), True)
        return np.argmax(pred_proba, axis=1), pred_proba

    def num_classes(self):
        return len(self.data_by_class)

    def delete_centroid_if_exists(self, class_num):
        self.model_invalid = True
        if class_num in self.data_by_class:
            del self.data_by_class[class_num]
            return class_num
        return None

    def add_data_for_class(self, data, class_num):
        self.model_invalid = True
        embeddings = self.get_embeddings(data)
        embeddings = embeddings.cpu().numpy()

        if class_num in self.data_by_class:
            self.data_by_class[class_num] = np.concatenate([self.data_by_class[class_num], embeddings], axis=0)
        else:
            self.data_by_class[class_num] = embeddings

    def get_class_nums_sorted(self):
        return sorted(self.data_by_class.keys())

    def do_train(self):
        print("Training Model")
        self.model_invalid = False

        all_data = [self.data_by_class[class_num] for class_num in self.get_class_nums_sorted()]
        all_targets = [(torch.ones(all_data[class_idx].shape[0]) * class_num).long().numpy() for class_idx, class_num in enumerate(self.get_class_nums_sorted())]
        self.class_idx2class_num = {class_idx: class_num for class_idx, class_num in enumerate(self.get_class_nums_sorted())}

        # Concat into one np array
        all_data = np.concatenate(all_data, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        self.model.fit(all_data, all_targets)
        train_acc = self.model.score(all_data, all_targets)

        y_pred = self.model.predict(all_data)
        cm = confusion_matrix(all_targets, y_pred)

        print(f"Train acc: {train_acc}")
        return train_acc, cm


class CentroidModel(FewShotModel):
    def __init__(self, input_dim, hidden_dim, latent_dim, layer_dim, rnn_type, rnn_use_last, checkpoint_file=None, device='cuda'):
        super().__init__(input_dim, hidden_dim, latent_dim, layer_dim, rnn_type, rnn_use_last, checkpoint_file=checkpoint_file, device='cuda')
        self.device = device

        # Centroids
        self.centroids = defaultdict(lambda: torch.zeros(latent_dim).to(self.device))
        self.class_counts = defaultdict(lambda: 0)

    def forward(self, x):
        if self.num_classes() <= 0:
            return None
        embeddings = self.get_embeddings(x)
        centroids, class_nums = self.get_centroids_as_tensor()
        dists = self.euclidean_dist(embeddings, centroids)
        closest_class_idxs = dists.argmin(dim=1)
        decoded_class_nums = class_nums.to(self.device).gather(0, closest_class_idxs)
        return decoded_class_nums, F.softmax(-dists, dim=1)

    def num_classes(self):
        return len(self.centroids)

    def delete_centroid_if_exists(self, class_num):
        if class_num in self.centroids:
            del self.centroids[class_num]
            del self.class_counts[class_num]
            return class_num
        else:
            return None

    def add_data_for_class(self, x, class_num):
        embeddings = self.get_embeddings(x)
        self.centroids[class_num] = (self.centroids[class_num] * self.class_counts[class_num] + embeddings.mean(dim=0)) / (self.class_counts[class_num] + embeddings.size(0))
        self.class_counts[class_num] += embeddings.size(0)

    def do_train(self):
        print("Training model")

    def _dont_call_this(self, x, class_num):
        x = x.cuda()

        self.centroids[class_num] = (self.centroids[class_num] * self.class_counts[class_num] + x.sum(dim=0)) / (self.class_counts[class_num] + x.size(0))
        self.class_counts[class_num] += x.size(0)

    def _dont_call_this_forward(self, x):
        centroids, class_nums = self.get_centroids_as_tensor()
        dists = self.euclidean_dist(x, centroids)
        closest_class_idxs = dists.argmin(dim=1)
        decoded_class_nums = class_nums.to(self.device).gather(0, closest_class_idxs)
        return decoded_class_nums, F.softmax(-dists, dim=1)

    def get_centroids_as_tensor(self):
        """
        Returns: Centroids, corresponding class list
        """
        centroid_lst = []
        class_num_lst = []
        for class_num, class_centroid in self.centroids.items():
            class_num_lst.append(class_num)
            centroid_lst.append(class_centroid)

        sorted_pairs = sorted(zip(class_num_lst, centroid_lst), key=lambda pair: pair[0])
        class_num_lst = [x for x, _ in sorted_pairs]
        centroid_lst = [x for _, x in sorted_pairs]

        centroids = torch.stack(centroid_lst)
        if len(centroids.size()) > 2:
            centroids = centroids.squeeze()

        return centroids, torch.Tensor(class_num_lst).long()

    def euclidean_dist(self, x, y):
        '''
        Compute euclidean distance between two tensors
        '''
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)


class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, num_heads):
        super(TransformerFeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.embedding_layer = nn.Linear(input_dim, hidden_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                                            dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=self.num_layers)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.embedding_layer(x)
        x = F.relu(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # aggregate the sequence into a single vector
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, num_heads):
        super(TransformerClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.transformer_feature_extractor = TransformerFeatureExtractor(input_dim=input_dim, hidden_dim=hidden_dim,
                                                                         num_classes=num_classes, num_layers=num_layers,
                                                                         num_heads=num_heads)
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.transformer_feature_extractor(x)
        x = self.output_layer(x)
        # x = F.softmax(x, dim=1)
        return x


if __name__ == '__main__':
    model = CentroidModel(input_dim=63, layer_dim=1, latent_dim=10, hidden_dim=128, rnn_type='GRU', rnn_use_last=False)

    data = torch.rand((200, 80, 63))
    print(model.get_embeddings(data))

    print()
    data = torch.rand((200, 10))
    model._dont_call_this(data * 0, 0)
    # model._dont_call_this(data * 0 + 1, 0)
    model._dont_call_this(data * 0 + 1, 1)
    model._dont_call_this(data * 0 + 2, 1)
    model._dont_call_this(data * 0 + 2, 2)
    model._dont_call_this(data * 0 + 3, 2)
    model._dont_call_this(data * 0 + 3, 3)
    model._dont_call_this(data * 0 + 4, 3)
    model._dont_call_this(data * 0 + 4, 4)

    model.delete_centroid_if_exists(0)

    model._dont_call_this(data * 0, 0)
    model._dont_call_this(data * 0 + 1, 0)

    centroids, vals = model.get_centroids_as_tensor()
    print()

    # data = torch.rand((200, 80, 63)) * 4
    # print(model(data))

    data = torch.rand((1, 10)) + .5
    pred, proba = model._dont_call_this_forward(data.cuda())
    print()
