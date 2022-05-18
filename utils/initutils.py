import os
import random
import numpy as np
import torch
import torch.nn as nn


def random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def weights_init(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    device = torch.device(device)
    return device
