

import torch
import torch.nn as nn


class UpdateGate(nn.Module):
    def __init__(self, in_channel, num_hidden, filter_size, stride, layer_norm=False, activation='tanh'):
        super(UpdateGate, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.activation = activation
        self.conv_x = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 3,
                    kernel_size=filter_size, stride=stride, padding=self.padding))
        self.conv_h = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 3, 
                    kernel_size=filter_size, stride=stride, padding=self.padding))
  

    def forward(self, x_t, h_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)

        r_x, z_x, n_x = torch.split(x_concat, self.num_hidden, dim=1)
        r_h, z_h, n_h = torch.split(h_concat, self.num_hidden, dim=1)

        r_t = torch.sigmoid(r_x + r_h)
        z_t = torch.sigmoid(z_x + z_h + self._forget_bias)
        if self.activation == 'lrelu':
            n_t = nn.functional.leaky_relu(n_x + r_t * n_h, negative_slope=0.2)
        else:
            n_t = torch.tanh(n_x + r_t * n_h)
        h_new = (1 - z_t) * n_t + z_t * h_t

        return h_new

class ConvGate(nn.Module):
    def __init__(self, in_channel, num_hidden, filter_size, stride, layer_norm=False, activation='tanh'):
        super(ConvGate, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.activation = activation
        self.conv_x = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 2,
                    kernel_size=filter_size, stride=stride, padding=self.padding))
        self.conv_h = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 2, 
                    kernel_size=filter_size, stride=stride, padding=self.padding))
  

    def forward(self, x_t, h_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)

        z_x, n_x = torch.split(x_concat, self.num_hidden, dim=1)
        z_h, n_h = torch.split(h_concat, self.num_hidden, dim=1)

        z_t = torch.sigmoid(z_x + z_h + self._forget_bias)
        n_t = torch.tanh(n_x +   n_h)
        h_new = (1 - z_t) * n_t + z_t * h_t

        return h_new

class UpdateGateThree(nn.Module):
    def __init__(self, in_channel, num_hidden, filter_size, stride, layer_norm=False, activation='tanh'):
        super(UpdateGateThree, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.activation = activation
        self.conv_x = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 2,
                    kernel_size=filter_size, stride=stride, padding=self.padding))
        self.conv_h = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 2, 
                    kernel_size=filter_size, stride=stride, padding=self.padding))
        self.conv_d = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 2, 
                    kernel_size=filter_size, stride=stride, padding=self.padding))                    
  

    def forward(self, x_t, h_t, d_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        d_concat = self.conv_d(d_t)
        x_z, x_d = torch.split(x_concat, self.num_hidden, dim=1)
        h_z, h_d = torch.split(h_concat, self.num_hidden, dim=1)
        d_z, d_d = torch.split(d_concat, self.num_hidden, dim=1)
 
        D_Z = torch.sigmoid(h_z + x_z + d_z)
        D_ = torch.tanh(h_d + x_d + d_d)
        h_new = (1 - D_Z) * D_ + D_Z * x_t

        return h_new
    

class LinearGate(nn.Module):
    def __init__(self, input_dim, rule_dim, num_hidden, layer_norm=False, activation='relu'):
        super(LinearGate, self).__init__()

        self.num_hidden = num_hidden
        self._forget_bias = 1.0
        self.activation = activation
        self.linear_x = nn.Linear(input_dim, num_hidden * 2)
        self.linear_h = nn.Linear(rule_dim, num_hidden * 2)
        self.layer_norm = nn.LayerNorm(num_hidden * 2) if layer_norm else None

    def forward(self, x_t, h_t):
        #x_t: graph vector
        #h_t: rule vector
        x_concat = self.linear_x(x_t)
        h_concat = self.linear_h(h_t)

        if self.layer_norm:
            x_concat = self.layer_norm(x_concat)
            h_concat = self.layer_norm(h_concat)

        z_x, n_x = torch.split(x_concat, self.num_hidden, dim=0)
        z_h, n_h = torch.split(h_concat, self.num_hidden, dim=0)

        z_t = torch.sigmoid(z_x + z_h + self._forget_bias)
        n_t = torch.relu(n_x + n_h)
        h_new = (1 - z_t) * n_t + z_t * h_t

        return h_new