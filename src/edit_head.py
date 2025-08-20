import torch, torch.nn as nn

class EditHead(nn.Module):
    def __init__(self, hidden_size, C=96, K=3):
        super().__init__()
        self.C, self.K = C, K
        self.proj = nn.Linear(hidden_size, K*(4+C))
    def forward(self, h_last):
        out = self.proj(h_last)                    # (B, K*(4+C))
        out = out.view(out.size(0), self.K, 4+self.C)
        out[:,:,0:2] = out[:,:,0:2].sigmoid()     # cx, cy in [0,1]
        out[:,:,2]   = out[:,:,2].sigmoid()*0.5 + 0.05
        out[:,:,3]   = torch.tanh(out[:,:,3])     # scale
        return out
