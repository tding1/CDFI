import argparse

import torch
import lpips
import numpy as np
import skimage.metrics
from torchvision.utils import save_image as imwrite
from torch.utils.data import DataLoader

from datasets import *
from models import make_model
from utility import print_and_save, count_network_parameters


class MyTest:
    def __init__(self):
        self.im_list = []
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.input0_list = []
        self.input1_list = []
        self.gt_list = []

    def test(self, model, output_dir, output_name="output", file_stream=None):
        model.eval()
        lpips_net = lpips.LPIPS(net="squeeze").cuda()
        with torch.no_grad():
            av_ssim = 0
            av_psnr = 0
            av_lpips = 0
            print("%25s%21s%21s" % ("PSNR", "SSIM", "lpips"))
            for idx in range(len(self.im_list)):
                if not os.path.exists(output_dir + "/" + self.im_list[idx]):
                    os.makedirs(output_dir + "/" + self.im_list[idx])

                in0, in1 = (
                    self.input0_list[idx].unsqueeze(0).cuda(),
                    self.input1_list[idx].unsqueeze(0).cuda(),
                )
                frame_out = model(in0, in1)

                imwrite(
                    frame_out,
                    output_dir + "/" + self.im_list[idx] + "/" + output_name + ".png",
                    range=(0, 1),
                )

                gt = self.gt_list[idx].numpy()
                ref = self.transform(
                    Image.open(
                        output_dir
                        + "/"
                        + self.im_list[idx]
                        + "/"
                        + output_name
                        + ".png"
                    )
                ).numpy()

                lps = lpips_net(
                    self.gt_list[idx].cuda(), torch.tensor(ref).unsqueeze(0).cuda()
                )
                psnr = skimage.metrics.peak_signal_noise_ratio(
                    image_true=gt, image_test=ref, data_range=1
                )
                ssim = skimage.metrics.structural_similarity(
                    np.transpose(gt, (1, 2, 0)),
                    np.transpose(ref, (1, 2, 0)),
                    data_range=1,
                    multichannel=True,
                )

                av_psnr += psnr
                av_ssim += ssim
                av_lpips += lps.item()

                msg = "{:<15s}{:<20.16f}{:<23.16f}{:<23.16f}".format(
                    self.im_list[idx] + ": ", psnr, ssim, lps.item()
                )
                if file_stream:
                    print_and_save(msg, file_stream)
                else:
                    print(msg)

        av_psnr /= len(self.im_list)
        av_ssim /= len(self.im_list)
        av_lpips /= len(self.im_list)
        msg = "\n{:<15s}{:<20.16f}{:<23.16f}{:<23.16f}".format(
            "Average: ", av_psnr, av_ssim, av_lpips
        )
        if file_stream:
            print_and_save(msg, file_stream)
        else:
            print(msg)

        return av_psnr


class Middlebury_other(MyTest):
    def __init__(self, input_dir, gt_dir):
        super(Middlebury_other, self).__init__()
        self.im_list = [
            "Beanbags",
            "Dimetrodon",
            "DogDance",
            "Grove2",
            "Grove3",
            "Hydrangea",
            "MiniCooper",
            "RubberWhale",
            "Urban2",
            "Urban3",
            "Venus",
            "Walking",
        ]
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(
                self.transform(Image.open(input_dir + "/" + item + "/frame10.png"))
            )
            self.input1_list.append(
                self.transform(Image.open(input_dir + "/" + item + "/frame11.png"))
            )
            self.gt_list.append(
                self.transform(Image.open(gt_dir + "/" + item + "/frame10i11.png"))
            )


class ucf_dvf(MyTest):
    def __init__(self, input_dir):
        super(ucf_dvf, self).__init__()
        self.im_list = [str(x) for x in list(range(1, 3791, 10))]
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(
                self.transform(Image.open(input_dir + "/" + item + "/frame_00.png"))
            )
            self.input1_list.append(
                self.transform(Image.open(input_dir + "/" + item + "/frame_02.png"))
            )
            self.gt_list.append(
                self.transform(Image.open(input_dir + "/" + item + "/frame_01_gt.png"))
            )


def Vimeo90K_test(args, model, out_dir):

    _, val_dataset = Vimeo90K_interp(args.vimeo_dir)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8
    )

    transform = transforms.Compose([transforms.ToTensor()])
    lpips_net = lpips.LPIPS(net="squeeze").cuda()

    img_out_dir = out_dir + "/vimeo90k"
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)

    av_psnr = 0
    av_ssim = 0
    av_lps = 0
    for batch_idx, (frame0, frame1, frame2) in enumerate(val_loader):
        frame0, frame2 = frame0.cuda(), frame2.cuda()
        output = model(frame0, frame2)

        imwrite(output, img_out_dir + "/" + str(batch_idx) + ".png", range=(0, 1))

        ref = transform(Image.open(img_out_dir + "/" + str(batch_idx) + ".png")).numpy()

        lps = lpips_net(frame1.cuda(), torch.tensor(ref).unsqueeze(0).cuda())
        psnr = skimage.metrics.peak_signal_noise_ratio(
            image_true=frame1.squeeze().numpy(), image_test=ref, data_range=1
        )
        ssim = skimage.metrics.structural_similarity(
            np.transpose(frame1.squeeze().numpy(), (1, 2, 0)),
            np.transpose(ref, (1, 2, 0)),
            data_range=1,
            multichannel=True,
        )

        print(
            "idx: %d, psnr: %f, ssim: %f, lpips: %f"
            % (batch_idx, psnr, ssim, lps.item())
        )

        av_psnr += psnr
        av_ssim += ssim
        av_lps += lps.item()

    av_psnr /= len(val_loader)
    av_ssim /= len(val_loader)
    av_lps /= len(val_loader)

    print(av_psnr, av_ssim, av_lps)


#########################################


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compression-Driven Frame Interpolation Evaluation"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="cdfi_adacof",
        choices=("cdfi_adacof", "compressed_adacof"),
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--vimeo_dir", type=str, default=None)
    parser.add_argument("--kernel_size", type=int, default=11)
    parser.add_argument("--dilation", type=int, default=2)

    return parser.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)

    model = make_model(args).cuda()
    print("===============================")
    print("# of model parameters is: " + str(count_network_parameters(model)))

    print("Loading model %s..." % args.model)
    print("Kernel size: %d, Dilation: %d" % (args.kernel_size, args.dilation))
    if args.model == "cdfi_adacof":
        assert args.kernel_size == 11 and args.dilation == 2
        checkpoint = torch.load("./checkpoints/CDFI_adacof.pth")
        out_dir = "./test_output/cdfi_adacof"
    elif args.model == "compressed_adacof":
        if args.kernel_size == 5 and args.dilation == 1:
            checkpoint = torch.load("./checkpoints/compressed_adacof_F_5_D_1.pth")
            out_dir = "./test_output/compressed_adacof_F_5_D_1"
        elif args.kernel_size == 11 and args.dilation == 2:
            checkpoint = torch.load("./checkpoints/compressed_adacof_F_11_D_2.pth")
            out_dir = "./test_output/compressed_adacof_F_11_D_2"
        else:
            exit("Invalid settings of kernel_size and dilation!")

    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    print("===============================")
    print("Test: Middlebury_others")
    img_out_dir = out_dir + "/middlebury_others"
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    test_db = Middlebury_other(
        "./test_data/middlebury_others/input", "./test_data/middlebury_others/gt"
    )
    test_db.test(model, img_out_dir)

    print("===============================")
    print("Test: UCF101-DVF")
    img_out_dir = out_dir + "/ucf101-dvf"
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    test_db = ucf_dvf("./test_data/ucf101_interp_ours")
    test_db.test(model, img_out_dir)

    if args.vimeo_dir is not None:
        print("===============================")
        print("Test: Vimeo-90K")
        Vimeo90K_test(args, model, out_dir)


if __name__ == "__main__":
    main()
