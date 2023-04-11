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
        # Initialize hidden state with zeros (# layers, batch size, feature size)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)

        # loop
        # out, (hi, ci) = self.rnn(x[:, 0, :].unsqueeze(1), (h0, c0))
        # pred = self.fc(out[:, -1, :])
        # for i in range(1, x.shape[1]):
        #     out, (hi, ci) = self.rnn(x[:, i, :].unsqueeze(1), (hi, ci))
        #     pred += self.fc(out[:, -1, :])
        # return pred / x.shape[1]

        # output shape is (batch size, sequence length, feature dim)
        output, (hn, cn) = self.rnn(x, (h0, c0))
        
        # average over sequence
        pred = torch.zeros(x.shape[0],self.fc.out_features).to(self.device)
        for i in range(x.shape[1]):
            pred += self.fc(output[:,i,:])
        return pred / x.shape[1]
    

# Create RNN Model with attention
class AttentionRNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(RNNModel, self).__init__()

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

        attention_weights = torch.zeros((x.shape[0],x.shape[1])).to(self.device)

        # output shape is (batch size, sequence length, feature dim)
        output, (hn, cn) = self.rnn(x, (h0, c0))

        # we need to figure out how to apply on the batch dimension, this is not quite right
        pred = torch.zeros(x.shape[0],self.fc.out_features).to(self.device)
        for i in range(x.shape[1]):
            attention_weights[i] = self.attention(output[:,i,:])
        
        attention_weights = F.softmax(attention_weights,dim=1)
        pred = self.fc(output@attention_weights) # this is not quite right
        #return pred / x.shape[1]


class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, num_heads):
        super(TransformerFeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.embedding_layer = nn.Linear(input_dim, hidden_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
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

        self.transformer_feature_extractor = TransformerFeatureExtractor(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, num_layers=num_layers, num_heads=num_heads)
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.transformer_feature_extractor(x)
        x = self.output_layer(x)
        # x = F.softmax(x, dim=1)
        return x
