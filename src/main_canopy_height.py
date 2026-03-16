from config import args as args_config
import os
import torch
from data import get as get_data
from importlib import import_module
from main import test, train
from model.ognidc import OGNIDC

def check_args(args):
    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain)

            # new_args = checkpoint['args']
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume
            new_args.start_epoch = checkpoint['epoch'] + 1

    return new_args


if __name__ == '__main__':
    args = check_args(args_config)
    args.train_data_name = "CanopyHeightDataset"
    args.test_data_name = "CanopyHeightDataset"
    args.val_data_name = "CanopyHeightDataset"
    
    args.epochs = 200
    args.batch_size = 8
    args.val_depth_pattern = "60000"
    # args.backbone_mode = "rgbd"
    args.backbone_mode = "sentineld"
    args.gpus = '0'
    # args.loss = '1.0*SeqL1+1.0*SeqL2+1.0*GradMatching+0.5*SeqLaplace'
    args.loss = '1.0*SeqL1+1.0*SeqL2'

    args.lr = 1e-4
    # args.milestones = [10, 20, 50]
    args.milestones = [50, 100, 150]
    args.gamma = 0.2

    # args.pretrain = 'model_best_72epochs.pt'

    train(0, args)

    # Evalution on NERCI inventory data
    # args.inventory_evaluation = True
    # args.pretrain = '/kaggle/input/datasets/khanhtq2101/canopy-height-kochi/Kochi_sentinel_GEDI_60kpoint.pt'   
    # test(args)


