import sys

sys.path.append("../")

from train import *
from models import make_model


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)

    # prepare training data
    train_dataset, val_dataset = Vimeo90K_interp(
        args.data_dir, args.num_training_samples
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    # val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    # prepare test data
    test_db = Middlebury_other(args.test_input, args.test_gt)

    # initialize our model
    model = make_model(args).cuda()
    print("# of model parameters is: " + str(utility.count_network_parameters(model)))

    # load pretrained model
    pretrained_dict = torch.load(args.checkpoint)
    model.load_state_dict(pretrained_dict["state_dict"])

    # prepare the loss
    loss = Loss(args)

    # prepare the trainer
    my_trainer = Trainer(args, train_loader, test_db, model, loss)

    # start training
    while not my_trainer.terminate():
        my_trainer.train()
        my_trainer.test()

        weights = [w for name, w in model.named_parameters() if w.requires_grad]
        density = sum([torch.sum(w != 0).item() for w in weights]) / sum(
            [w.numel() for w in weights]
        )

        print("************* Model density: %.7f *************" % density)

    my_trainer.close()


if __name__ == "__main__":
    main()
