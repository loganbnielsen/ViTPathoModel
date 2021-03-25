import log
import logging
logger = logging.getLogger('root') 

import torch
import torch.nn as nn

from vit import ViT



class ViTPathoModel(nn.Module):
    def __init__(self, vit_configs, res_block_configs):
        super().__init__()
        self.vit = ViT(**vit_configs)
        self.res_blocks = ResidualBlocks(**res_block_configs)
        ## TODO I'm here.
        ##      4) RPN
        ##      5) output

    def forward(self, X):
        # ViT
        X = self.vit(X) # TODO ViT produces a single dimension output. stack multiple ViT outputs?
        X = X.unsqueeze(1)
        # Residual Blocks
        X = self.res_blocks(X)
        # RPN
        # OUTPUT
        return X



class ResidualBlocks(nn.Module):
    def __init__(self, block_channels):
        """
            block_channels: [(1,2,1),(1,2,1)]. each tuple of 3 numbers represent a block
            
            TODO make it so input and output channels for a block don't need to match.
        """
        super().__init__()
        self.blocks = nn.Sequential(*[ResBlock(*c) for c in block_channels])
    
    def forward(self, X):
        return self.blocks(X)


class ResBlock(nn.Module):
    def __init__(self, c1, c2, c3):
        super().__init__()
        self.activ = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(1)

    def forward(self, X):
        orig_X = X

        X = self.conv1(X)
        X = self.activ(self.bn1(X))

        X = self.conv2(X)
        X = self.activ(self.bn2(X))

        return orig_X + X

        
