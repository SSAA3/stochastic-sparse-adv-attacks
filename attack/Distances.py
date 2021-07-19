"""
Useful functions to compute distances between original sample and adversarial sample
"""

import torch

  
def l0_distance(x, y):

    """
    For a batch of images, computing the L0 distance between two images.
    """

    return torch.sum(torch.logical_not(torch.isclose(x, y)), dim=1).type(torch.int64)

def l1_distance(x, y):

    """
    For a batch of images, computing the L1 distance between two images.
    """

    return torch.sum(torch.abs(y-x), dim=1)


def l2_distance(x, y):

    """
    For a batch of images, computing the L2 distance between two images.
    """

    return torch.sqrt(torch.sum(torch.square(y-x), dim=1))



def compute_distances(x_in, x_out, batch_size=1):
    
    x_in = x_in.view(batch_size, -1)
    x_out = x_out.view(batch_size, -1)
    
    L0 = l0_distance(x_in, x_out).view(-1,1)
    L1 = l1_distance(x_in, x_out).view(-1,1)
    L2 = l2_distance(x_in, x_out).view(-1,1)
    
    return torch.cat([L0, L1, L2], dim=1)