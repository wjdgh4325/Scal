import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import pdb

Uniform = torch.distributions.Uniform
Beta = torch.distributions.Beta
MVN = torch.distributions.MultivariateNormal
Normal = torch.distributions.Normal
LogNormal = torch.distributions.LogNormal
Bernoulli = torch.distributions.Bernoulli
Weibull = torch.distributions.Weibull
Gamma = torch.distributions.Gamma
RELU = torch.nn.functional.relu
RELU6 = torch.nn.functional.relu6
grad = torch.autograd.grad
kl = torch.distributions.kl_divergence
EPS = 1e-4

# torch.multiprocessing.set_start_method('spawn', force = True)
def censor_time_to_indicators(survival_time, censor_time):
    """
    assumes time and censor_time has been sampled continuously
    creates a censored indicator that is true if censoring
    occured before event. If so, time is replaced with censoring time
    """

    N = survival_time.size()[0]
    time = torch.zeros_like(survival_time)
    dead = torch.zeros_like(censor_time)
    for i in range(N):
        if survival_time[i] < censor_time[i]:
            dead[i] = 1
            time[i] = survival_time[i]

        else:
            dead[i] = 0
            time[i] = censor_time[i]

    dead = dead.long()

    return time, dead

class SyntheticDataset(data.Dataset):
    def __init__(self, x, surv_t, censor_t, phase, is_training=True, censor=True):
        #print(x.size())
        #print(surv_t.size())
        #print(censor_t.size())
        assert x.size()[0] == surv_t.size()[0]
        assert surv_t.size()[0] == censor_t.size()[0]

        if censor:
            print("Synthetic Dataset is Applying Censoring to Data")
            t, dead = censor_time_to_indicators(surv_t, censor_t)
            print(t.size())
            print(dead.size())

        else:
            print("Synthetic Dataset Hardcoded To Not Censor")
            dead = torch.ones_like(surv_t).long()
            t = surv_t
            print(t.size())
            print(dead.size())

        target = torch.zeros((t.size()[0], 2))
        target[:, 0] = t.squeeze()
        target[:, 1] = dead.squeeze()
        target = target

        self.src = x.float()
        self.tgt = target.float()
        self.survival_times = surv_t.float()
        self.censor_times = censor_t.float()

        self.is_training = is_training
        self.num_examples = x.size()[0]
        self.D_in = x.size()[1]

        self.phase = phase

    def __getitem__(self, index):
        x = self.src[index]
        y = self.tgt[index]
        t = self.survival_times[index]
        c = self.censor_times[index]

        return x, y

    def __len__(self):
        return len(self.src)

def get_synthetic_loader(args, phase, is_training=True, shuffle=False, dist='lognormal', censor=True):
    print('-' * 69)
    print("Making Dataloader for {}".format(phase))

    if phase == 'train':
        suffix = 'tr'

    elif phase == 'valid':
        suffix = 'va'

    elif phase == 'test':
        suffix = 'te'

    else:
        assert False, "wrong phase"

    prefix = 'data/'
    xfilename = prefix + '{}_x_{}.pt'.format(dist, suffix)
    survtfilename = prefix + '{}_surv_t_{}.pt'.format(dist, suffix)
    censortfilename = prefix + '{}_censor_t_{}.pt'.format(dist, suffix)

    print("Loading file: {}".format(xfilename))
    print("Loading file: {}".format(survtfilename))
    print("Loading file: {}".format(censortfilename))

    x = torch.load(xfilename)
    surv_t = torch.load(survtfilename)
    censor_t = torch.load(censortfilename)

    if phase == 'test':
        print('*' * 39)
        #print('YOU ARE ONLY USING 10000 for test')
        print('*' * 39)
        #x = x[:50000]
        #surv_t = surv_t[:10000]
        #censor_t = censor_t[:10000]

    dataset = SyntheticDataset(x=x, surv_t=surv_t, censor_t=censor_t, phase=phase, is_training=is_training, censor=censor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=shuffle)

    loader.phase = phase
    loader.D_in = dataset.D_in
    print('-' * 69)

    return loader