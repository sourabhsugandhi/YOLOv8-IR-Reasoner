import torch
import torch.nn as nn
import torch.nn.functional as F

class Reasoner(nn.Module):
    """
    IR-Reasoner block adapted from YOLOv7-Reasoner.
    Applies spatial self-attention on a feature map and 
    returns a refined fetures map with residual connection 
    """

    def __init__(self, c1=None, reduction=8):
        """ 
        c1: input channels
        """
        super().__init__()
        self.reduction = reduction

        #layer will be initialized lazily
        self.q = None
        self.k = None
        self.v = None
        self.proj = None
        self.scale = None

        #if channels are provided, initialize immediately

        if c1 is not None:
            self._init_layers(c1)

    def _init_layers(self,c):
        hidden_dim = c // self.reduction

        self.q = nn.Conv2d(c,hidden_dim, 1, bias=False)
        self.k = nn.Conv2d(c,hidden_dim, 1, bias=False)
        self.v = nn.Conv2d(c,c, 1, bias=False)
        self.proj = nn.Conv2d(c,c,1, bias=False)

        self.scale = hidden_dim ** -0.5

    def forward(self,x):
        print("Reasoner forward:", x.shape)
        b,c,h,w = x.shape

        #lazy init (first forward)

        if self.q is None:
            self._init_layers(c)

        n = h * w

        q = self.q(x).reshape(b, -1, n).permute(0,2,1)
        k = self.k(x).reshape(b, -1, n)
        v = self.v(x).reshape(b, -1, n).permute(0,2,1)

        attn = torch.bmm(q, k) * self.scale
        attn = F.softmax(attn, dim=1)

        out = torch.bmm(attn, v)
        out = out.permute(0,2,1).reshape(b,c,h,w)

        return x + self.proj(out)




