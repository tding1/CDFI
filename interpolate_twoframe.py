import argparse

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image as imwrite

from models.cdfi_adacof import CDFI_adacof


def parse_args():
    parser = argparse.ArgumentParser(description='Two-frame Interpolation')

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/CDFI_adacof.pth')

    parser.add_argument('--kernel_size', type=int, default=11)
    parser.add_argument('--dilation', type=int, default=2)

    parser.add_argument('--first_frame', type=str, default='./imgs/0.png')
    parser.add_argument('--second_frame', type=str, default='./imgs/1.png')
    parser.add_argument('--output_frame', type=str, default='./output.png')

    return parser.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)

    model = CDFI_adacof(args).cuda()

    print('Loading the model...')
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    frame_name1 = args.first_frame
    frame_name2 = args.second_frame

    transform = transforms.Compose([transforms.ToTensor()])
    frame1 = transform(Image.open(frame_name1)).unsqueeze(0).cuda()
    frame2 = transform(Image.open(frame_name2)).unsqueeze(0).cuda()

    model.eval()
    with torch.no_grad():
        frame_out = model(frame1, frame2)

    imwrite(frame_out.clone(), args.output_frame, range=(0, 1))


if __name__ == "__main__":
    main()
