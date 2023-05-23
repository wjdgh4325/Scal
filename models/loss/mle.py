import pdb
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import util

sys.path.append("...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_log(x, EPS = 1e-4):
    return (x + EPS).log()

class MLE(nn.Module):
    def __init__(self, args):
        super(MLE, self).__init__()
        self.log_base = torch.FloatTensor([np.e]).to(DEVICE)
        self.eps = 1e-4
        self.args = args

    # Fast forward
    def forward(self, pred_params, tgt, model_dist):
        tte, is_dead = tgt[:, 0], tgt[:, 1]

        is_dead = is_dead.long().to(DEVICE)
    
        mask = torch.ones(tte.shape[0], tte.shape[0])
        tte = tte.reshape(-1, 1)
        mask[(tte - tte.T) > 0] = 0
        mask = mask.to(DEVICE)
        
        log_pl = mask * torch.exp(pred_params).reshape(-1)
        log_pl = torch.sum(log_pl, dim=1) / torch.sum(mask, dim=1)
        log_pl = torch.log(log_pl).reshape(-1, 1)
        
        neg_log_pl = - torch.sum((pred_params - log_pl).reshape(-1) * is_dead) / torch.sum(is_dead)
        
        def bad(x):
            return torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))

        if bad(log_pl):
            print("BAD LOG PDF in MLE")

        return neg_log_pl