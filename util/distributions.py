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

import util
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

def pred_params_to_dist(pred_params, args):
    if args.model_dist=='lognormal':
        pred = pred_params_to_lognormal(pred_params)
    elif args.model_dist=='weibull':
        pred = pred_params_to_weibull(pred_params)
    else:
        pred = pred_params_to_cat(pred_params,args)
    return pred

def compute_counts(event, time, order=None):
    """Count right censored and uncensored samples at each unique time point.
    Parameters
    ----------
    event : array
        Boolean event indicator.
    time : array
        Survival time or time of censoring.
    order : array or None
        Indices to order time in ascending order.
        If None, order will be computed.
    Returns
    -------
    times : array
        Unique time points.
    n_events : array
        Number of events at each time point.
    n_at_risk : array
        Number of samples that have not been censored or have not had an event at each time point.
    n_censored : array
        Number of censored samples at each time point.
    """
    n_samples = event.shape[0]

    if order is None:
        order = torch.argsort(time, kind="mergesort")

    failure_times = torch.empty(n_samples, dtype=time.dtype)
    uniq_events = torch.empty(n_samples, dtype=int)
    uniq_counts = torch.empty(n_samples, dtype=int)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        failure_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    times = np.resize(failure_times, j)
    n_events = np.resize(uniq_events, j)
    total_count = np.resize(uniq_counts, j)
    n_censored = total_count - n_events
    # offset cumulative sum by one
    total_count = torch.cat((torch.tensor([0]), torch.tensor(total_count)), dim=-1)
    n_at_risk = n_samples - torch.cumsum(total_count, dim = 0)

    return times, n_events, n_at_risk[:-1], n_censored

def get_cdf_val(pred_params, tgt, args):
    risk_score = torch.exp(pred_params)
    tte, is_dead = tgt[:, 0], tgt[:, 1]
    
    if args.model_dist in ['cat','mtlr']:
        tte, is_alive, ratio = tgt[:, 0], tgt[:, 1], tgt[:, 2]
        cdf = pred.cdf(tte, ratio)

    while True:
        tie_breaking = torch.rand(len(tte)) * 1e-6
        while len(tte) != len(torch.unique(tte + tie_breaking.to(DEVICE))):
            tie_breaking = torch.rand(len(tte)) * 1e-6

        break
    
    tie_breaking = tie_breaking.to(DEVICE)
    tte = tte + tie_breaking

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

def pred_params_to_dist(pred_params, args):
    if args.model_dist=='lognormal':
        pred = pred_params_to_lognormal(pred_params)

    elif args.model_dist=='weibull':
        pred = pred_params_to_weibull(pred_params)

    else:
        pred = pred_params_to_cat(pred_params, args)

    return pred

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