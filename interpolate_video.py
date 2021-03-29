import os
import argparse

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image as imwrite

from models.cdfi_adacof import CDFI_adacof


def parse_args():
    parser = argparse.ArgumentParser(description='Video Interpolation')

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/CDFI_adacof.pth')

    parser.add_argument('--kernel_size', type=int, default=11)
    parser.add_argument('--dilation', type=int, default=2)

    parser.add_argument('--index_from', type=int, default=0, help='when index starts from 1 or 0 or else')
    parser.add_argument('--zpad', type=int, default=5, help='zero padding of frame name.')

    parser.add_argument('--img_format', type=str, default='.jpg')
    parser.add_argument('--input_video', type=str, default='./imgs/img_seq')
    parser.add_argument('--output_video', type=str, default='./interpolated_video')

    return parser.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)

    transform = transforms.Compose([transforms.ToTensor()])

    model = CDFI_adacof(args).cuda()

    print('Loading the model...')
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    base_dir = args.input_video

    if not os.path.exists(args.output_video):
        os.makedirs(args.output_video)

    frame_len = len([name for name in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, name))])

    for idx in range(frame_len - 1):
        idx += args.index_from
        print(idx, '/', frame_len - 1, end='\r')

        frame_name1 = base_dir + '/' + str(idx).zfill(args.zpad) + args.img_format
        frame_name2 = base_dir + '/' + str(idx + 1).zfill(args.zpad) + args.img_format

        frame1 = transform(Image.open(frame_name1)).unsqueeze(0).cuda()
        frame2 = transform(Image.open(frame_name2)).unsqueeze(0).cuda()

        model.eval()
        with torch.no_grad():
            frame_out = model(frame1, frame2)

        # interpolate
        imwrite(frame1.clone(), args.output_video + '/'
                + str((idx - args.index_from) * 2 + args.index_from).zfill(args.zpad) + args.img_format, range=(0, 1))
        imwrite(frame_out.clone(), args.output_video + '/'
                + str((idx - args.index_from) * 2 + 1 + args.index_from).zfill(args.zpad) + args.img_format, range=(0, 1))

    # last frame
    print(frame_len - 1, '/', frame_len - 1)
    frame_name_last = base_dir + '/' + str(frame_len + args.index_from - 1).zfill(args.zpad) + args.img_format
    frame_last = transform(Image.open(frame_name_last)).unsqueeze(0)
    imwrite(frame_last.clone(), args.output_video + '/' + str((frame_len - 1) * 2 + args.index_from).zfill(args.zpad)
            + args.img_format, range=(0, 1))


if __name__ == "__main__":
    main()
