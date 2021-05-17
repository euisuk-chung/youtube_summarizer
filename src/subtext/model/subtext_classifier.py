import torch
import torch.nn as nn


class SubtextClassifier(nn.Module):
    
    def __init__(self, window_size):
        super(SubtextClassifier, self).__init__()
        
        if window_size == 1:
            conv_kernel_size = 2
            linear_channel = 1
        else:
            conv_kernel_size = window_size*2-2
            linear_channel = 3
        
        # block1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=128, kernel_size=conv_kernel_size),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(128*linear_channel, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        
        # forward block 1
        out = self.block1(x)
        batch_size = out.size()[0]
        
        out = out.view(batch_size, -1)
        out = self.block2(out)
        
        return out.view(-1)