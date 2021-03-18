from collections import OrderedDict

import torch


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
    # build url
    url = 'https://github.com/S-aiueo32/PerceptualSimilarity/tree/82ea5c4826444549cdbd09501955b44b87f0d800' \
        + f'/models/weights/v{version}/{net_type}.pth'

    # download
    old_state_dict = torch.load('lpips_pytorch/squeeze.pth')

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict
