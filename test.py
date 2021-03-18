import os
import argparse

import torch
import numpy as np
from PIL import Image
import skimage.metrics
from torchvision import transforms
from torchvision.utils import save_image as imwrite

from lpips_pytorch import lpips
from models.cdfi_adacof import CDFI_adacof
from utility import print_and_save, count_network_parameters


class MyTest:
    def __init__(self):
        self.im_list = []
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.input0_list = []
        self.input1_list = []
        self.gt_list = []

    def test(self, model, output_dir, output_name='output', file_stream=None):
        model.eval()
        with torch.no_grad():
            av_ssim = 0
            av_psnr = 0
            av_lpips = 0
            print('%25s%21s%21s' % ('PSNR', 'SSIM', 'lpips'))
            for idx in range(len(self.im_list)):
                if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                    os.makedirs(output_dir + '/' + self.im_list[idx])

                in0, in1 = self.input0_list[idx].unsqueeze(0).cuda(), self.input1_list[idx].unsqueeze(0).cuda()
                frame_out = model(in0, in1)

                lps = lpips(self.gt_list[idx].cuda(), frame_out, net_type='squeeze')

                imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name + '.png', range=(0, 1))

                frame_out = frame_out.squeeze().detach().cpu().numpy()
                gt = self.gt_list[idx].numpy()

                psnr = skimage.metrics.peak_signal_noise_ratio(image_true=gt, image_test=frame_out)
                ssim = skimage.metrics.structural_similarity(np.transpose(gt, (1, 2, 0)),
                                                             np.transpose(frame_out, (1, 2, 0)), multichannel=True)

                av_psnr += psnr
                av_ssim += ssim
                av_lpips += lps.item()

                msg = '{:<15s}{:<20.16f}{:<23.16f}{:<23.16f}'.format(self.im_list[idx] + ': ', psnr, ssim, lps.item())
                if file_stream:
                    print_and_save(msg, file_stream)
                else:
                    print(msg)

                self.gt_list[idx].to('cpu')

        av_psnr /= len(self.im_list)
        av_ssim /= len(self.im_list)
        av_lpips /= len(self.im_list)
        msg = '{:<15s}{:<20.16f}{:<23.16f}{:<23.16f}'.format('Average: ', av_psnr, av_ssim, av_lpips)
        if file_stream:
            print_and_save(msg, file_stream)
        else:
            print(msg)

        return av_psnr


class Middlebury_other(MyTest):
    def __init__(self, input_dir, gt_dir):
        super(Middlebury_other, self).__init__()
        self.im_list = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper',
                        'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(self.transform(Image.open(input_dir + '/' + item + '/frame10.png')))
            self.input1_list.append(self.transform(Image.open(input_dir + '/' + item + '/frame11.png')))
            self.gt_list.append(self.transform(Image.open(gt_dir + '/' + item + '/frame10i11.png')))


class ucf_dvf(MyTest):
    def __init__(self, input_dir):
        super(ucf_dvf, self).__init__()
        self.im_list = [str(x) for x in list(range(1, 3791, 10))]
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(self.transform(Image.open(input_dir + '/' + item + '/frame_00.png')))
            self.input1_list.append(self.transform(Image.open(input_dir + '/' + item + '/frame_02.png')))
            self.gt_list.append(self.transform(Image.open(input_dir + '/' + item + '/frame_01_gt.png')))


#########################################

def parse_args():
    parser = argparse.ArgumentParser(description='Compression-Driven Frame Interpolation Evaluation')

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/CDFI_adacof.pth')
    parser.add_argument('--out_dir', type=str, default='./test_output/cdfi_adacof')
    parser.add_argument('--kernel_size', type=int, default=11)
    parser.add_argument('--dilation', type=int, default=2)

    return parser.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)

    model = CDFI_adacof(args).cuda()
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
