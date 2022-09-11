import os
import sys

sys.path.append("../")


import torch
import numpy as np

from train import *
from models import make_model


args = parse_args()

# initialize our model
model = make_model(args)
print("# of model parameters is: " + str(utility.count_network_parameters(model)))

pretrained_dict = torch.load(args.checkpoint)
if "state_dict" in pretrained_dict:
    model.load_state_dict(pretrained_dict["state_dict"])
else:
    model.load_state_dict(pretrained_dict)

print(
    "%35s" % "Name"
    + "%24s" % "Density"
    + "%4s" % "In"
    + "%5s" % "Out"
    + "%6s" % "In'"
    + "%5s" % "Out'"
)
for name, w in model.named_parameters():
    if name.find("weight") > 0:
        out_ch, in_ch = w.shape[0], w.shape[1]
        density = torch.sum(w != 0).item() / np.prod(w.shape)
        sqrt_density = np.sqrt(density)

        print("%50s" % name, end="   ")
        print(
            "%.3f" % density
            + "  "
            + "%3d" % in_ch
            + "  "
            + "%3d" % out_ch
            + "  "
            + "%3d" % int(sqrt_density * in_ch)
            + "  "
            + "%3d" % int(sqrt_density * out_ch)
        )
