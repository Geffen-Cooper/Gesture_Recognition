import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)

        # loop
        out, (hi, ci) = self.rnn(x[:, 0, :].unsqueeze(1), (h0, c0))
        pred = self.fc(out[:, -1, :])
        for i in range(1, x.shape[1]):
            out, (hi, ci) = self.rnn(x[:, i, :].unsqueeze(1), (hi, ci))
            pred += self.fc(out[:, -1, :])
        return pred / x.shape[1]


class RNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, layer_dim, rnn_type, device):
        super(RNNFeatureExtractor, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.device = device

        # RNN
        if rnn_type == "GRU":
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)
        self.embedder = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        rnn_mean = torch.mean(rnn_out, dim=1)
        output = self.embedder(rnn_mean)
        return output


class RNNTest(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, layer_dim, output_dim, device):
        super(RNNTest, self).__init__()
        self.feature_extractor = RNNFeatureExtractor(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                                                     layer_dim=layer_dim, device=device)
        self.fc = nn.Linear(latent_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.relu(x)
        x = self.fc(x)
        return x


# # Create RNN Model
# class RNNModel(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
#         super(RNNModel, self).__init__()

#         # Number of hidden dimensions
#         self.hidden_dim = hidden_dim

#         # Number of hidden layers
#         self.layer_dim = layer_dim

#         # RNN
#         # self.rnn = torch.nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
#         self.rnn = torch.nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
#         # Readout layer
#         self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, 32) for i in range(80)])

#         self.final = torch.nn.Linear(32*80,output_dim)
#         self.do = torch.nn.Dropout()

#     def forward(self, x):
#         # Initialize hidden state with zeros
#         h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to('cuda')
#         c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to('cuda')

#         # loop
#         out, (hi,ci) = self.rnn(x[:,0,:].unsqueeze(1), (h0,c0))
#         pred = self.do(self.fc_layers[0](out[:,-1,:]))
#         for i in range(1,x.shape[1]):
#             out, (hi,ci) = self.rnn(x[:,i,:].unsqueeze(1), (hi,ci))
#             pred = torch.cat([pred,self.do(self.fc_layers[i](out[:,-1,:]))],dim=1)

#         # print(pred.shape)
#         return self.final(pred)


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
