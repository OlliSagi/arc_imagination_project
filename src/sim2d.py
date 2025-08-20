import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange

class SimBlock(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln1  = nn.LayerNorm(d)
        self.ffn  = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.ln2  = nn.LayerNorm(d)
    def forward(self, x):
        # x: (B,N,D)
        h,_ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x

class Sim2D(nn.Module):
    """
    Always-on 2D latent field S \in R^{B,C,H,W} with transformer updates and soft-brush edits.
    """
    def __init__(self, H=32, W=32, C=96, dim=256, layers=6, heads=8):
        super().__init__()
        self.H, self.W, self.C = H, W, C
        self.proj_in  = nn.Conv2d(C, dim, 1)
        self.blocks   = nn.ModuleList([SimBlock(dim, heads) for _ in range(layers)])
        self.proj_out = nn.Conv2d(dim, C, 1)
        self.delta    = nn.Conv2d(C, C, 3, padding=1)
    def step(self, S, edit_tokens=None):
        B,C,H,W = S.shape
        x = self.proj_in(S)                    # (B,D,H,W)
        x = rearrange(x, 'b d h w -> b (h w) d')
        for blk in self.blocks:
            x = blk(x)
        x = rearrange(x, 'b n d -> b d n')
        x = rearrange(x, 'b d (h w) -> b d h w', h=H, w=W)
        x = self.proj_out(x)
        S = S + 0.1 * torch.tanh(self.delta(x))  # keep it alive
        if edit_tokens is not None:
            S = apply_soft_brushes(S, edit_tokens)
        return S

def apply_soft_brushes(S, edit_tokens):
    """
    edit_tokens: (B,K,4+C) -> [cx, cy, r, scale, d1..dC]
    """
    B,C,H,W = S.shape
    device = S.device
    ys = torch.linspace(0, 1, H, device=device)
    xs = torch.linspace(0, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([xx, yy], dim=0)  # (2,H,W)
    for k in range(edit_tokens.shape[1]):
        params = edit_tokens[:,k,:]
        cx, cy, r, scale = params[:,0], params[:,1], params[:,2], params[:,3]
        delta = params[:,4:4+C]
        masks = []
        for b in range(B):
            g = torch.exp(-((grid[0]-cx[b])**2 + (grid[1]-cy[b])**2) / (2.0*(r[b]**2 + 1e-6)))
            masks.append(g)
        mask = torch.stack(masks, dim=0).unsqueeze(1)  # (B,1,H,W)
        mask = mask / (mask.amax(dim=(-2,-1), keepdim=True) + 1e-6)
        S = S + mask * (delta.view(B,C,1,1) * scale.view(B,1,1,1))
    return S

def init_field(B, C, H, W, device):
    return 0.01 * torch.randn(B, C, H, W, device=device)
