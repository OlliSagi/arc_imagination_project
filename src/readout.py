import torch, torch.nn as nn
from einops import rearrange

class Readout(nn.Module):
    def __init__(self, C=96, H=32, W=32, dim=256, n_queries=8):
        super().__init__()
        self.proj_in = nn.Conv2d(C, dim, 1)
        self.queries = nn.Parameter(torch.randn(n_queries, dim))
        self.proj_out = nn.Linear(dim, dim)
    def forward(self, S):
        B,C,H,W = S.shape
        x = self.proj_in(S)                        # (B,D,H,W)
        x = rearrange(x, 'b d h w -> b (h w) d')   # (B,N,D)
        q = self.queries.unsqueeze(0).expand(B,-1,-1)  # (B,Q,D)
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        q_norm = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        attn = torch.einsum('bnd,bqd->bnq', x_norm, q_norm).softmax(dim=1)  # (B,N,Q)
        pooled = torch.einsum('bnd,bnq->bqd', x, attn).mean(dim=1)  # (B,D)
        return self.proj_out(pooled)               # (B,D)
