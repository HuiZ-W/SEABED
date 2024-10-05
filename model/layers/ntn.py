import torch
import torch.nn as nn
import torch.nn.functional as F

class NTN(nn.Module):
    def __init__(self, input_dim, out_dim, args):
        super(NTN, self).__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.args = args

        self.W = nn.Parameter(torch.Tensor(input_dim, input_dim, out_dim))
        self.V = nn.Parameter(torch.Tensor(out_dim, 2 * input_dim))
        self.B = nn.Parameter(torch.Tensor(out_dim))
        self.U = nn.Parameter(torch.Tensor(out_dim, 1))

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.U)
        nn.init.zeros_(self.B)

    def forward(self, features1, features2):
        features1 = features1
        features2 = features2
        tensor_product = torch.einsum('ai,ijk,bj->abk', [features1, self.W, features2])
        features1_expanded = features1.unsqueeze(1).expand(features1.size(0),features2.size(0),features1.size(1))  # [a, 1, 200]
        features2_expanded = features2.unsqueeze(0).expand(features1.size(0),features2.size(0),features1.size(1))  # [a, 1, 200]
        concat_features = torch.cat((features1_expanded, features2_expanded),dim=2)
        Linear_term = torch.matmul(concat_features, self.V.t())
        x1 = (tensor_product + Linear_term + self.B).squeeze()
        x = F.relu(tensor_product + Linear_term + self.B).squeeze()
        #x = torch.matmul(x, self.U).squeeze()
        return x, x1