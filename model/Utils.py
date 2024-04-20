import torch
import torch.nn.functional as F
import math
from torch.autograd import Variable
from config.Config import hparams


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def kl(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)


def kl_gnn(mu):
    return torch.mean(-0.5 * torch.sum(- mu ** 2, dim=1), dim=0)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(Variable(mask) == 0, -1e9).to(hparams['device'])
    p_attn = F.softmax(scores, dim=-1)  # torch.Size([1326, 8, 20, 20])
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    import dgl
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    dgl.seed(seed)

