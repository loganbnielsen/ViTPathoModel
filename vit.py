# Some edits. Src: https://github.com/lucidrains/vit-pytorch
import log
import logging
logger = logging.getLogger('root') 

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, outpatch_els, depth, heads, dim_head, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                Residual(PreNorm(outpatch_els, Attention(outpatch_els, heads = heads, dim_head = dim_head, dropout = dropout)))
            )
    def forward(self, x):
        for attn in self.layers:
            x = attn(x)
        return x

class ViT(nn.Module):
    def __init__(self, img_len, patch_len, outpatch_len, depth, heads, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert img_len % patch_len == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (img_len // patch_len) ** 2
        patch_els = channels * patch_len ** 2
        outpatch_els = outpatch_len ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_len, p2 = patch_len), # h is the number of patches across the x axis, w is the same but across the y axis
            nn.Linear(patch_els, outpatch_els),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, outpatch_els))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(outpatch_els, depth, heads, dim_head, dropout)

        self.tile_x_dim = self.tile_y_dim = outpatch_len
        self.num_x_splits = self.num_y_splits = img_len // patch_len

    def _combine_tiles(self, X):
        """
            INPUT:
                tensor([[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                         [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                         [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                         [4., 4., 4., 4., 4., 4., 4., 4., 4., 4.]]])
            OUTPUT:
                tensor([[[1., 1., 1., 1., 1., 2., 2., 2., 2., 2.],
                         [1., 1., 1., 1., 1., 2., 2., 2., 2., 2.],
                         [3., 3., 3., 3., 3., 4., 4., 4., 4., 4.],
                         [3., 3., 3., 3., 3., 4., 4., 4., 4., 4.]]])
        """

        X = X.reshape(-1, self.num_x_splits, self.num_y_splits, self.tile_x_dim, self.tile_y_dim) # batch x layers_to_stack x blocks_for_row x tile_x_dim x tile_y_dim
        X = X.permute(0,1,3,2,4).reshape(-1, self.tile_x_dim * self.num_x_splits,
                                             self.tile_y_dim * self.num_y_splits)
        return X

    def forward(self, X):
        X = self.to_patch_embedding(X)
        b, n, _ = X.shape # batch_size x num_tiles x embedding_dimension
        X += self.pos_embedding[:, :n]
        X = self.dropout(X)

        X = self.transformer(X)
        X = self._combine_tiles(X)

        
        return X
