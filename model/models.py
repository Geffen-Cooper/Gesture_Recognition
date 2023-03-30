import torch


# Create RNN Model
class RNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,device):
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
        out, (hi,ci) = self.rnn(x[:,0,:].unsqueeze(1), (h0,c0))
        pred = self.fc(out[:,-1,:])
        for i in range(1,x.shape[1]):
            out, (hi,ci) = self.rnn(x[:,i,:].unsqueeze(1), (hi,ci))
            pred += self.fc(out[:,-1,:])
        return pred/x.shape[1]
    




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