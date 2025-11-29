from config import args as args_config
import os
import torch
from data import ibims, ibims_draft
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
    # print(args)
    # args.split_json = "./khanh"
    args.train_data_name = "iBims_Draft"
    args.test_data_name = "iBims_Draft"
    args.val_data_name = "iBims_Draft"
    args.epochs = 1
    args.batch_size = 2
    
    data_name = args.train_data_name
    module_name = 'data.' + data_name.lower()
    dataset_name = data_name
    module = import_module(module_name)

    print(module_name)

    print(module)

    dataset = getattr(module, dataset_name)(args, mode="test")
    dataset = getattr(module, dataset_name)(args, mode="train")

    data_train = get_data(args, 'train')

    net = OGNIDC(args)
    print("Getting sentinel data:")
    dataset[0]
    print(len(dataset))


    # print(dataset[0])
    # print("TRAINING BEGINS")

    train(0, args)
    test(args)
    

    # data_train = get_data(args, "train")
    # for param in net.parameters():
    #     param.requires_grad = False

    i = 0 
    for name, param in net.named_parameters():
        if i != 0: 
            param.requires_grad = False
        # if i == 0:
        #     print(name, param.requires_grad)
        i += 1
    print(i)
    print("Number of parameters:", sum(p.numel() for p in net.parameters()))
    print("Number of trained parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))
    print(type(net.state_dict()))
    print("Done")