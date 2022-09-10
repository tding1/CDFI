import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class DBreader_Vimeo90k(Dataset):
    def __init__(
        self,
        root,
        path_list,
        random_crop=None,
        resize=None,
        augment_s=True,
        augment_t=True,
    ):
        self.root = root
        self.path_list = path_list

        self.random_crop = random_crop
        self.resize = resize
        self.augment_s = augment_s
        self.augment_t = augment_t

    def __getitem__(self, index):
        path = self.path_list[index]
        return self.Vimeo90K_loader(path)

    def __len__(self):
        return len(self.path_list)

    def Vimeo90K_loader(self, im_path):
        abs_im_path = os.path.join(self.root, "sequences", im_path)

        transform_list = []
        if self.resize is not None:
            transform_list += [transforms.Resize(self.resize)]
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

        rawFrame0 = Image.open(os.path.join(abs_im_path, "im1.png"))
        rawFrame1 = Image.open(os.path.join(abs_im_path, "im2.png"))
        rawFrame2 = Image.open(os.path.join(abs_im_path, "im3.png"))

        if self.random_crop is not None:
            i, j, h, w = transforms.RandomCrop.get_params(
                rawFrame1, output_size=self.random_crop
            )
            rawFrame0 = TF.crop(rawFrame0, i, j, h, w)
            rawFrame1 = TF.crop(rawFrame1, i, j, h, w)
            rawFrame2 = TF.crop(rawFrame2, i, j, h, w)

        if self.augment_s:
            if random.randint(0, 1):
                rawFrame0 = TF.hflip(rawFrame0)
                rawFrame1 = TF.hflip(rawFrame1)
                rawFrame2 = TF.hflip(rawFrame2)
            if random.randint(0, 1):
                rawFrame0 = TF.vflip(rawFrame0)
                rawFrame1 = TF.vflip(rawFrame1)
                rawFrame2 = TF.vflip(rawFrame2)

        frame0 = self.transform(rawFrame0)
        frame1 = self.transform(rawFrame1)
        frame2 = self.transform(rawFrame2)

        if self.augment_t:
            if random.randint(0, 1):
                return frame2, frame1, frame0
            else:
                return frame0, frame1, frame2
        else:
            return frame0, frame1, frame2


def make_dataset(root, list_file, num_training_samples=-1):
    raw_im_list = open(os.path.join(root, list_file)).read().splitlines()
    raw_im_list = raw_im_list[:-1]  # the last line is invalid in test set
    assert len(raw_im_list) > 0
    random.shuffle(raw_im_list)
    if num_training_samples <= 0:
        return raw_im_list
    else:
        return raw_im_list[-num_training_samples:]


def Vimeo90K_interp(root, num_training_samples=-1):
    train_list = make_dataset(root, "tri_trainlist.txt", num_training_samples)
    val_list = make_dataset(root, "tri_testlist.txt")
    train_dataset = DBreader_Vimeo90k(root, train_list)
    val_dataset = DBreader_Vimeo90k(root, val_list)
    return train_dataset, val_dataset
