import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import logging


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
        avg_hidden = torch.mean(output,dim=1)
        return self.fc(avg_hidden)


# Create RNN Model with attention
class AttentionRNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
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

        # Readout layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)

        # (batch_size,sequence length) since we want a weight for each hidden state in the sequence
        attention_weights = torch.zeros((x.shape[0],x.shape[1])).to(self.device)

        # output shape is (batch size, sequence length, feature dim)
        output, (hn, cn) = self.rnn(x, (h0, c0))

        # for each time step, get the attention weight (do this over the batch)
        for i in range(x.shape[1]):
            attention_weights[:,i] = self.attention(output[:,i,:]).view(-1)
        attention_weights = F.softmax(attention_weights,dim=1)

        # apply attention weights for each element in batch
        attended = torch.zeros(output.shape[0],output.shape[2]).to(self.device)
        for i in range(x.shape[0]):
            attended[i,:] = attention_weights[i]@output[i,:,:]

        return self.fc(attended)


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

        # RNN
        if rnn_type == "GRU":
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)
        self.embedder = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        if not self.rnn_use_last:
            rnn_mean = torch.mean(rnn_out, dim=1)
        else:
            rnn_mean = rnn_out[:, -1, :]
        output = self.embedder(rnn_mean)
        return output


class CentroidModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, layer_dim, rnn_type, rnn_use_last, checkpoint_file=None, device='cuda'):
        super().__init__()
        self.checkpoint_file = checkpoint_file
        self.device = device
        self.fe_model = RNNFeatureExtractor(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, layer_dim=layer_dim, rnn_type=rnn_type, rnn_use_last=rnn_use_last, device=device).to(device)

        if self.checkpoint_file:
            logging.warning("Checkpoint not specified, loading untrained model!")
            checkpoint = torch.load(self.checkpoint_file)
            print(f"Loading RNNFeatureExtractor with epoch: {checkpoint['epoch']}, val_acc: {checkpoint['val_acc']}, val_loss: {checkpoint['val_loss']}")
            self.fe_model.load_state_dict(checkpoint['model_state_dict'])

        # Centroids
        self.centroids = defaultdict(lambda: torch.zeros(latent_dim).to(self.device))
        self.class_counts = defaultdict(lambda: 0)

    def get_embeddings(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            embeddings = self.fe_model(x)
        return embeddings

    def forward(self, x):
        embeddings = self.get_embeddings(x)
        centroids, class_nums = self.get_centroids_as_tensor()
        dists = self.euclidean_dist(embeddings, centroids)
        closest_class_idxs = dists.argmin(dim=1)
        decoded_class_nums = class_nums.to(self.device).gather(0, closest_class_idxs)
        return decoded_class_nums

    def num_classes(self):
        return len(self.centroids)

    def learn_centroid(self, x, class_num):
        embeddings = self.get_embeddings(x)
        self.centroids[class_num] = (self.centroids[class_num] * self.class_counts[class_num] + embeddings.mean(dim=0)) / (self.class_counts[class_num] + embeddings.size(0))
        self.class_counts[class_num] += embeddings.size(0)

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

        return torch.stack(centroid_lst).squeeze(), torch.Tensor(class_num_lst)


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

# class RNNTest(nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim, layer_dim, output_dim, device):
#         super(RNNTest, self).__init__()
#         self.feature_extractor = RNNFeatureExtractor(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
#                                                      layer_dim=layer_dim, device=device)
#         self.fc = nn.Linear(latent_dim, output_dim)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.feature_extractor(x)
#         x = self.relu(x)
#         x = self.fc(x)
#         return x



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
    data = torch.rand((200, 80, 63))
    model.learn_centroid(data * 0.01, 0)
    model.learn_centroid(data * 1, 1)
    model.learn_centroid(data * 2, 2)
    model.learn_centroid(data * 3, 3)
    model.learn_centroid(data * 4, 4)

    print(model.get_centroids_as_tensor())

    data = torch.rand((200, 80, 63)) * 4
    print(model(data))
