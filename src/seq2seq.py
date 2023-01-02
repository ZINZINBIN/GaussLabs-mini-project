# simple neural network
import torch
import torch.nn as nn
import random
from torch.autograd import Variable
from typing import Optional
from pytorch_model_summary import summary

class RnnEncoder(nn.Module):
    def __init__(self, input_dim : int, hidden_dim : int, n_layers : int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = n_layers, bidirectional = True, batch_first = False)
    
    def forward(self, x : torch.Tensor):
        x = x.permute(1,0,2)
        output, (h, c) = self.lstm(x)
        output = output.permute(1,0,2)
        h = h.permute(1,0,2)
        c = c.permute(1,0,2)

        return output, (h,c)

class RnnDecoder(nn.Module):
    def __init__(self, input_dim : int, hidden_dim : int, output_dim : int, n_layers : int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = n_layers, bidirectional = True, batch_first = False)
        self.mlp = nn.Linear(2 * hidden_dim, output_dim)
    
    def forward(self, x : torch.Tensor, h: torch.Tensor, c : torch.Tensor):
        
        x = x.permute(1,0,2)
        h = h.permute(1,0,2)
        c = c.permute(1,0,2)
        output, (h, c) = self.lstm(x, (h,c))
        self.h = h.permute(1,0,2)

        h = h.permute(1,0,2)
        c = c.permute(1,0,2)
        output = output.permute(1,0,2).squeeze(1)

        output = self.mlp(output)

        return output, (h,c)

class SimpleRNN(nn.Module):
    def __init__(self, input_dim : int, hidden_dim : int, output_dim : int, n_layers : int, target_len : int, teacher_forcing_ratio : float):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.target_len = target_len
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        self.encoder = RnnEncoder(input_dim, hidden_dim, n_layers)
        self.decoder = RnnDecoder(input_dim, hidden_dim, output_dim, n_layers)
    
    # teacher forcing
    def forward(self, x : torch.Tensor, x_target : torch.Tensor, target_len : Optional[int] = None, teacher_forcing_ratio : Optional[float] = None):
        
        if target_len is None:
            target_len = self.target_len
        
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = self.teacher_forcing_ratio
        
        batch_size = x.size()[0]
        input_size = x.size()[2]

        output = torch.zeros(batch_size, target_len, input_size).to(x.device)

        _, (h, c) = self.encoder(x)
        decoder_input = x[:,-1,:].view(batch_size, 1, input_size)

        for t in range(target_len):
            out, (h,c) = self.decoder(decoder_input, h,c)

            if random.random() < teacher_forcing_ratio:
                decoder_input = x_target[:,t,:].unsqueeze(1)
            else:
                decoder_input = out.unsqueeze(1)
            
            output[:,t,:] = out
        
        return output
    
    def predict(self, x : torch.Tensor, target_len : Optional[int] = None):
        
        if target_len is None:
            target_len = self.target_len
        
        with torch.no_grad():
            
            if x.ndim == 2:
                x = x.unsqueeze(0)
            
            batch_size = x.size()[0]
            input_size = x.size()[2]
            output = torch.zeros(batch_size, target_len, input_size).to(x.device)
            _, (h,c) = self.encoder(x)

            decoder_input = x[:,-1,:].unsqueeze(1)

            for t in range(target_len):
                out, (h, c) = self.decoder(decoder_input, h,c)
                out = out.unsqueeze(1)
                decoder_input = out
                output[:,t,:] = out.squeeze(1)
            
            return output
        
    def summary(self):
        sample_data = torch.zeros((1, 1, self.input_dim))
        return summary(self, sample_data, sample_data, 1, 0.5, batch_size = 1, show_input = True, show_hierarchical=False,print_summary=True)