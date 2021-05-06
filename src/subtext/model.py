import torch
import torch.nn as nn

class ChunkClassifier(nn.Module):
    
    def __init__(self, x_features):
        super(ChunkClassifier, self).__init__()
        
        # block1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=128, kernel_size=6),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(128*3, 16),
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