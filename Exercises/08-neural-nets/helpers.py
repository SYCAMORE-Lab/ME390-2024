import numpy as np
import torch
import random


def set_seed(seed):
    """
    Set the seed for reproducibility.
    
    Parameters:
    seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
