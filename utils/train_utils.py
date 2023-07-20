import torch


def weight_reset(m):
    # if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()