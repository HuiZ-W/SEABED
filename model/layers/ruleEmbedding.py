import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RuleEmbedding(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RuleEmbedding, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 64
        self.num_layers = 2
        self.rnn = nn.RNN(self.input_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=False)
        self.linear = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, ori_lengths):
        '''
        input: x: (batch_size, max_length, input_dim) --> the input sequence
               ori_lengths: (batch_size) --> the length of each sequence in the batch
        output: (batch_size, output_dim) --> the last hidden output of the RNN
        '''
        packed_x = pack_padded_sequence(x, ori_lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_x)
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)

        hidden = hidden[-1].squeeze()
        output = self.linear(hidden)

        return output
    
    def padding_data(self, x):
        '''
        input: x: [(length, input_dim)] --> the list of input sequences
        output: (batch_size, max_length, input_dim) --> the padded input sequence
        '''
        max_length = max(tensor.shape[1] for tensor in x)
        ori_lengths = [tensor.shape[1] for tensor in x]
        padded_tensor = [torch.cat([tensor, torch.zeros(max_length - tensor.shape[1], tensor.shape[2]).to(tensor.device).unsqueeze(0)], dim=1) for tensor in x]
        padded_tensor = torch.cat(padded_tensor, dim=0)
        
        return padded_tensor, ori_lengths
    

class RuleRanker(torch.nn.Module):
    def __init__(self, input_dim):
        """
        input: input_dim(int)
        """
        self.hidden_dim1 = 128
        self.hidden_dim2 = 64
        super(RuleRanker, self).__init__()
        self.fc1 = nn.Linear(input_dim, self.hidden_dim1)
        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fc3 = nn.Linear(self.hidden_dim2, 1)

    def forward(self, x):
        """
        input: x(batch_size, dim)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        scores = self.fc3(x)
        scores = torch.sigmoid(scores)
        return scores.squeeze()
    
