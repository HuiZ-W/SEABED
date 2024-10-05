import torch
import torch.nn as nn
import torch.nn.functional as F
import math
'''
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()

        self.dim = input_dim
        self.query = nn.Linear(self.dim, self.dim)
        self.key = nn.Linear(self.dim, self.dim)
        self.value = nn.Linear(self.dim, self.dim)
        self.softmax = nn.Softmax(dim=-1)
        #self.cls_token = nn.Parameter(torch.zeros(1, self.dim))
        self.cls_token = nn.Parameter(torch.randn(1, self.dim))
        nn.init.kaiming_uniform_(self.query.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.key.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.value.weight, a=math.sqrt(5))
        if self.query.bias is not None:
            nn.init.constant_(self.query.bias, 0)
        if self.key.bias is not None:
            nn.init.constant_(self.key.bias, 0)
        if self.value.bias is not None:
            nn.init.constant_(self.value.bias, 0)

    def forward(self, x):
        # x: (k, m)
        x = torch.cat((self.cls_token, x), dim=0)
        query = self.query(x)  # (k+1, m)
        key = self.key(x)      # (k+1, m)
        value = self.value(x)  # (k+1, m)
        
        #compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float32))  # (k+1, k+1)
        attention_weights = self.softmax(attention_scores)  # (k+1, k+1)
        
        #compute attention output
        attention_output = torch.matmul(attention_weights, value)  # (k+1, m)
        
        #get cls representation
        cls_representation = attention_output[0, :]  # (m)

        return cls_representation
'''  
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()

        self.dim = input_dim
        self.query = nn.Linear(self.dim, self.dim)
        self.key = nn.Linear(self.dim, self.dim)
        self.value = nn.Linear(self.dim, self.dim)
        self.softmax = nn.Softmax(dim=-1)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
            nn.ReLU()
        )
        '''
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Dropout(p=0.2),
        )'''
        nn.init.kaiming_uniform_(self.query.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.key.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.value.weight, a=math.sqrt(5))
        if self.query.bias is not None:
            nn.init.constant_(self.query.bias, 0)
        if self.key.bias is not None:
            nn.init.constant_(self.key.bias, 0)
        if self.value.bias is not None:
            nn.init.constant_(self.value.bias, 0)

    def forward(self, x, query):
        # x: (k, m)
        query = self.query(query)  # (1, m)
        key = self.key(x)      # (k, m)
        value = self.value(x)  # (k, m)
        
        #compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float32))  # (1, k)
        attention_weights = self.softmax(attention_scores)  # (1, k)
        '''
        if attention_weights[0].item() != 0.03125:
            print(attention_weights)'''
        #compute attention output
        attention_output = torch.matmul(attention_weights, value)  # (1, m)
        
        #get cls representation
        result = self.mlp(attention_output)  # (m)

        return result, attention_weights