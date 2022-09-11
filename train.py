import os
import shutil
import datetime
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader

import utility
from loss import Loss
from datasets import Vimeo90K_interp
from mytest import Middlebury_other
from models.cdfi_adacof import CDFI_adacof


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compression-Driven Frame Interpolation Training"
    )

    # parameters
    # Model Selection
    parser.add_argument("--model", type=str, default="cdfi_adacof")

    # Hardware Setting
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--use_cudnn", type=bool, default=True)

    # Directory Setting
    parser.add_argument("--data_dir", type=str, default="./vimeo_triplet/")
    parser.add_argument("--uid", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--force", action="store_true", help="force to override the given uid"
    )
    parser.add_argument(
        "--test_input", type=str, default="./test_data/middlebury_others/input"
    )
    parser.add_argument(
        "--test_gt", type=str, default="./test_data/middlebury_others/gt"
    )

    # Learning Options
    parser.add_argument("--epochs", type=int, default=100, help="Max Epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--loss",
        type=str,
        default="1*Charb+0.01*g_Spatial+0.005*VGG",
        help="loss function configuration",
    )
    parser.add_argument(
        "--num_training_samples", type=int, default=-1, help="Traning sub dataset size"
    )

    # Optimization specifications
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--lr_decay", type=int, default=20, help="learning rate decay per N epochs"
    )
    parser.add_argument(
        "--decay_type", type=str, default="step", help="learning rate decay type"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="learning rate decay factor for step decay",
    )
    parser.add_argument(
        "--optimizer",
        default="ADAMax",
        choices=("SGD", "ADAM", "RMSprop", "ADAMax", "OBProxSG"),
        help="optimizer to use (SGD | ADAM | RMSprop | ADAMax | OBProxSG)",
    )
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument(
        "--lambda_",
        type=float,
        default=1e-4,
        help="regularization parameter for L1 optimization",
    )
    parser.add_argument(
        "--Np", type=int, default=5, help="number of P-steps in OBProxSG"
    )

    # Options for AdaCoF
    parser.add_argument("--kernel_size", type=int, default=11)
    parser.add_argument("--dilation", type=int, default=2)

    parser.add_argument("--inactive", action="store_true", default=None)

    args = parser.parse_args()

    if not args.inactive:

        if args.uid is None:
            unique_id = str(np.random.randint(0, 100000))
            print("revise the unique id to a random number " + str(unique_id))
            args.uid = unique_id
            timestamp = datetime.datetime.now().strftime("%a-%b-%d-%H-%M")
            save_path = "./model_weights/" + args.uid + "-" + timestamp
        else:
            save_path = "./model_weights/" + str(args.uid)

        if not os.path.exists(save_path + "/best" + ".pth"):
            os.makedirs(save_path, exist_ok=True)
        else:
            if not args.force:
                raise ("please use another uid ")
            else:
                print("override this uid" + args.uid)
                for m in range(1, 10):
                    if not os.path.exists(save_path + "/log.txt.bk" + str(m)):
                        shutil.copy(
                            save_path + "/log.txt", save_path + "/log.txt.bk" + str(m)
                        )
                        shutil.copy(
                            save_path + "/args.txt", save_path + "/args.txt.bk" + str(m)
                        )
                        break

        parser.add_argument(
            "--save_path", default=save_path, help="the output dir of weights"
        )
        parser.add_argument(
            "--log", default=save_path + "/log.txt", help="the log file in training"
        )
        parser.add_argument(
            "--arg", default=save_path + "/args.txt", help="the args used"
        )

        args = parser.parse_args()

        with open(args.log, "w") as f:
            f.close()
        with open(args.arg, "w") as f:
            print(args)
            print(args, file=f)
            f.close()

        if args.use_cudnn:
            print("cudnn is used")
            torch.backends.cudnn.benchmark = True
        else:
            print("cudnn is not used")
            torch.backends.cudnn.benchmark = False

    return args


class Trainer:
    def __init__(
        self, args, train_loader, test_loader, my_model, my_loss, start_epoch=1
    ):
        self.args = args
        self.train_loader = train_loader
        self.max_step = self.train_loader.__len__()
        self.test_loader = test_loader
        self.model = my_model
        self.loss = my_loss
        self.current_epoch = start_epoch
        self.save_path = args.save_path

        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        self.result_dir = args.save_path + "/results"
        self.ckpt_dir = args.save_path + "/checkpoints"

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.logfile = open(args.log, "w")

        # Initial Test
        self.model.eval()
        self.best_psnr = self.test_loader.test(
            self.model,
            self.result_dir,
            output_name=str(self.current_epoch).zfill(3),
            file_stream=self.logfile,
        )

    def train(self):
        # Train
        self.model.train()
        for batch_idx, (frame0, frame1, frame2) in enumerate(self.train_loader):
            frame0 = frame0.cuda()
            frame1 = frame1.cuda()
            frame2 = frame2.cuda()

            self.optimizer.zero_grad()

            output = self.model(frame0, frame2)
            loss = self.loss(output, frame1)
            loss.backward()
            self.optimizer.step()

            if batch_idx % 100 == 0:
                torch.save(
                    {
                        "epoch": self.current_epoch,
                        "state_dict": self.model.state_dict(),
                    },
                    self.ckpt_dir + "/real_time.pth",
                )
                utility.print_and_save(
                    "{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f}".format(
                        "Train Epoch: ",
                        "["
                        + str(self.current_epoch)
                        + "/"
                        + str(self.args.epochs)
                        + "]",
                        "Step: ",
                        "[" + str(batch_idx) + "/" + str(self.max_step) + "]",
                        "train loss: ",
                        loss.item(),
                    ),
                    self.logfile,
                )

        self.current_epoch += 1
        self.scheduler.step()
        utility.print_and_save(
            "===== current lr: %f =====" % (self.optimizer.param_groups[0]["lr"]),
            self.logfile,
        )

    def test(self):
        utility.print_and_save("Testing...", self.logfile)
        # Test
        self.model.eval()
        tmp_psnr = self.test_loader.test(
            self.model,
            self.result_dir,
            output_name=str(self.current_epoch).zfill(3),
            file_stream=self.logfile,
        )
        if tmp_psnr > self.best_psnr:
            self.best_psnr = tmp_psnr
            torch.save(
                {"epoch": self.current_epoch, "state_dict": self.model.state_dict()},
                self.ckpt_dir
                + "/model_epoch_"
                + str(self.current_epoch).zfill(3)
                + ".pth",
            )

    def terminate(self):
        return self.current_epoch >= self.args.epochs

    def close(self):
        self.logfile.close()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)

    # prepare training data
    train_dataset, val_dataset = Vimeo90K_interp(args.data_dir)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    # val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    # prepare test data
    test_db = Middlebury_other(args.test_input, args.test_gt)

    # initialize our model
    model = CDFI_adacof(args).cuda()
    print("# of model parameters is: " + str(utility.count_network_parameters(model)))

    # prepare the loss
    loss = Loss(args)

    # prepare the trainer
    my_trainer = Trainer(args, train_loader, test_db, model, loss)

    # start training
    while not my_trainer.terminate():
        my_trainer.train()
        my_trainer.test()

    my_trainer.close()


if __name__ == "__main__":
    main()
