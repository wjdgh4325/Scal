import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def brier_score(cdf, tte):
    surv_function = 1 - cdf
    tte = tte.reshape(-1, 1)
    mask = torch.ones([tte.shape[0], tte.shape[0]])
    mask[(tte - tte.T) >= 0] = 0
    mask = mask.to(DEVICE)
    surv_function = surv_function.reshape(-1, 1)
    
    return torch.sum(torch.sum((mask - surv_function)**2, dim=1) / surv_function.shape[0], dim=0) / max(tte)
