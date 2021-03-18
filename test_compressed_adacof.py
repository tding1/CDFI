import os
import sys
import argparse

import torch

from test import Middlebury_other, ucf_dvf
from models.compressed_adacof import AdaCoFNet
from utility import count_network_parameters


#########################################

def parse_args():
    parser = argparse.ArgumentParser(description='Compressed AdaCoF Evaluation')

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--kernel_size', choices=[5, 11], type=int, default=5)
    parser.add_argument('--dilation', choices=[1, 2], type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)

    if args.kernel_size == 5 and args.dilation == 1:
        args.checkpoint = './checkpoints/compressed_5.pth'
        args.out_dir = './test_output/compressed_adacof_F_5_D_1'
    elif args.kernel_size == 11 and args.dilation == 2:
        args.checkpoint = './checkpoints/compressed_adacof_F_11_D_2.pth'
        args.out_dir = './test_output/compressed_adacof_F_11_D_2'
    else:
        sys.exit('Kernel size and Dilation do not match with our compressed model')

    model = AdaCoFNet(args).cuda()
    print('===============================')
    print("# of model parameters is: " + str(count_network_parameters(model)))

    print('Loading the model...')
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    print('===============================')
    print('Test: Middlebury_others')
    test_dir = args.out_dir + '/middlebury_others'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_db = Middlebury_other('./test_data/middlebury_others/input', './test_data/middlebury_others/gt')
    test_db.test(model, test_dir)

    print('===============================')
    print('Test: UCF101-DVF')
    test_dir = args.out_dir + '/ucf101-dvf'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_db = ucf_dvf('./test_data/ucf101_interp_ours')
    test_db.test(model, test_dir)


if __name__ == "__main__":
    main()