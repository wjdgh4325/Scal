import random
import sys

import torch
import torch.nn.functional as F
import torch.utils.data as data

import util

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pickle
import warnings

import numpy as np
from lifelines.utils import concordance_index
from tqdm import tqdm

from args import TestArgParser
from evaluator import ModelEvaluator
from logger import TestLogger
from saver import ModelSaver


def pred_params_to_cat(pred_params, args):
    pred = util.CatDist(pred_params, args)

    return pred

def pred_params_to_weibull(pred_params):
    pre_scale = pred_params[:, 0]
    scale = pre_scale + 1.
    pre_k = pred_params[:, 1]
    k = pre_k.sigmoid() + 1.0
    pred = torch.distributions.Weibull(scale, k)

    return pred

def pred_params_to_lognormal_params(pred_params):
    mu = pred_params[:, 0]
    pre_log_sigma = pred_params[:, 1]
    log_sigma = F.softplus(pre_log_sigma) - 0.5
    sigma = log_sigma.clamp(max = 10).exp()
    sigma = sigma + 1e-4

    return mu, sigma

def pred_params_to_lognormal(pred_params):
    mu, sigma = pred_params_to_lognormal_params(pred_params)
    pred = torch.distributions.LogNormal(mu, sigma)

    return pred

def pred_params_to_dist(pred_params, tgt, args):
    if args.model_dist=='lognormal':
        pred = pred_params_to_lognormal(pred_params)
    
    elif args.model_dist=='weibull':
        pred = pred_params_to_weibull(pred_params)

    elif args.model_dist == 'mtlr':
        pred = pred_params_to_cat(pred_params, args)

    else:
        pred = pred_params_to_cox(pred_params, tgt)

    return pred

def pred_params_to_cox(pred_params, tgt):
    risk_score = torch.exp(pred_params)
    tte, is_dead = tgt[:, 0], tgt[:, 1]
    """
    while True:
        tie_breaking = torch.rand(len(tte)) * 1e-6
        while len(tte) != len(torch.unique(tte + tie_breaking.to(DEVICE))):
            tie_breaking = torch.rand(len(tte)) * 1e-6

        break
     
    tie_breaking = tie_breaking.to(DEVICE)
    tte = tte + tie_breaking
    """
    order = torch.argsort(tte)
    tte = tte[order].reshape(-1, 1)
    is_dead = is_dead[order]
    mask = torch.ones([tte.shape[0], tte.shape[0]])
    mask[(tte - tte.T) > 0] = 0
    mask = mask.to(DEVICE)
    risk_score = risk_score[order].reshape(-1)
    risk_value = torch.sum(mask * risk_score, dim=1) + 1e-13
    value = is_dead / risk_value
    
    H = torch.cumsum(value, dim=0) # Breslow estimator
    S = torch.exp(-H) + 1e-13 # Baseline survival function
    
    cdf = 1 - S ** risk_score

    return cdf

def get_cdf_val(pred_params, tgt, args):
    
    pred = pred_params_to_dist(pred_params, tgt, args)

    if args.model_dist in ['cat','mtlr']:
        tte, is_dead, ratio = tgt[:, 0], tgt[:, 1], tgt[:, 2]
        cdf = pred.cdf(tte, ratio)

    elif args.model_dist == 'lognormal':
        tte, is_dead = tgt[:, 0], tgt[:, 1]
        cdf = pred.cdf(tte + 1e-4)
        
    else:
        cdf = pred

    return cdf
    
def get_predict_time(pred, args):
    if args.model_dist in ['cat','mtlr']:
        return pred.predict_time()
    
    elif args.model_dist == 'lognormal':
        if args.pred_type == 'mean':
            pred_time = pred.mean

        elif args.pred_type == 'mode':
            pred_time = util.log_normal_mode(pred)

    elif args.model_dist == 'weibull':
        logtwo = torch.tensor([2.0]).to(DEVICE).log()
        inverse_concentration = 1.0 / pred.concentration
        pred_time = pred.scale * logtwo.pow(inverse_concentration)
        
        if torch.any(torch.isnan(pred_time)) or torch.any(torch.isinf(pred_time)):
            print(":(")
    
    else:
        assert False, "wrong dist or pred type in predict time in utils"
    
    return pred_time

def get_logpdf_val(pred_params, tgt, args):

    pred = pred_params_to_dist(pred_params, tgt, args)
    tte = tgt[:,0]

    if args.model_dist == 'lognormal' or args.model_dist == 'weibull':
        tte = tte + 1e-4
    log_prob = pred.log_prob(tte)
    
    return log_prob

def log_normal_mode(pytorch_distribution_object):
    return (pytorch_distribution_object.loc - pytorch_distribution_object.scale.pow(2)).exp()
