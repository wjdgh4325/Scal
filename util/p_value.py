from scipy import stats
import torch
import numpy as np


def get_p_value(cdf, tte, is_dead, B=1000, device='cpu'):
    cdf = cdf.to(device).view(-1, 1)
    tte = tte.to(device).view(-1, 1)

    points_dead = cdf[is_dead == 1]
    #points_dead = points_dead.expand(points_dead.shape[0], B)
    points_cens = cdf[is_dead == 0]
    #predict_dead = (1 - points_cens) * torch.rand(B).to(device) + points_cens
    predict_dead = (points_cens + 1) / 2

    statistics_dead = torch.cat([points_dead, predict_dead], dim=0)
    uniform_dead = torch.sort(statistics_dead, dim=0)[0].cpu()
    #ecdf_dead = (((torch.arange(cdf.shape[0]) + 1) / cdf.shape[0]).view(-1, 1).expand(cdf.shape[0], B)).cpu()
    ecdf_dead = (((torch.arange(cdf.shape[0]) + 1) / cdf.shape[0]).view(-1, 1)).cpu()

    difference = abs(uniform_dead - ecdf_dead)
    #KS_statistic = (torch.max(difference, dim=0)[0]).mean()
    KS_statistic = torch.max(difference, dim=0)[0][0]
    
    z = KS_statistic * np.sqrt((cdf.shape[0]**2) / (2*cdf.shape[0]))
    p = 0
    for i in range(1, 101):
        p += ((-1) ** (i-1)) * np.exp((-2) * (i**2) * (z**2))
    
    p_value = 2 * p
    
    return KS_statistic, p_value
    
"""
if __name__ == '__main__':
    p_value = get_p_value(0.08, 19, 10000)

    print('p_value', p_value)
"""