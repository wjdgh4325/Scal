import torch
import torch.nn.functional as F
import torch.utils.data as data
import util
import random
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from args import TestArgParser
from logger import TestLogger
from saver import ModelSaver
import numpy as np
from lifelines.utils import concordance_index
from evaluator import ModelEvaluator
import warnings

def concordance(args, test_loader, model):
    tte_per_batch = []
    is_dead_per_batch = []
    pred_risks_per_batch = []

    for i, (src, tgt) in enumerate(test_loader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tte = tgt[:, 0]
        is_dead = tgt[:, 1]
        #order = torch.argsort(tte)
        tte_per_batch.append(tte)
        is_dead_per_batch.append(is_dead)

        pred_params = -model.forward(src.to(args.device))
        pred_risks_per_batch.append(pred_params)
    
    ttes = torch.cat(tte_per_batch)
    #order = torch.argsort(ttes)
    ttes = ttes.cpu().numpy()
    is_deads = torch.cat(is_dead_per_batch).long().cpu().numpy()
    pred_risks = torch.cat(pred_risks_per_batch).cpu().numpy()
    
    concordance = concordance_index(ttes, pred_risks, is_deads)

    return concordance