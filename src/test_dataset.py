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
    args.epochs = 40
    args.batch_size = 8
    args.val_depth_pattern = "60000"
    args.lr = 1e-5
    args.backbone_model = "rgbd"
    args.gpus = '0'
    # args.loss = '1.0*SeqL1+1.0*SeqL2+1.0*GradMatching+0.5*SeqLaplace'
    args.loss = '1.0*SeqL1+1.0*SeqL2'
    args.milestones = ''

    # args.pretrain = 'model_best_72epochs.pt'

    #---------------
    #Draft Dec 24: Try to load trained model from Hugging Face
    # args.load_dav2 = 1
    # net = OGNIDC.from_pretrained("zuoym15/OMNI-DC", args=args, strict= False)
    # net = OGNIDC(args=args)
    # if args.pretrain is not None:
    #     assert os.path.exists(args.pretrain), \
    #         "file not found: {}".format(args.pretrain)

    #     # checkpoint = torch.load(args.pretrain, map_location={'cuda:0': 'cuda:%d' % gpu})
    #     checkpoint = torch.load(args.pretrain, map_location='cpu')

    #     model_dict = net.state_dict()
    #     state_dict = checkpoint['net']

    #     compatible_state_dict = {}
    #     skipped = []

    #     for k, v in state_dict.items():
    #         if k in model_dict and v.shape == model_dict[k].shape:
    #             compatible_state_dict[k] = v
    #         else:
    #             skipped.append(k)

    #     missing, unexpected = net.load_state_dict(
    #         compatible_state_dict,
    #         strict=False
    #     )
    #     print(compatible_state_dict.keys())
        
        # net.load_state_dict(checkpoint['net'], strict=False)

    #     print('Load network parameters from : {}'.format(args.pretrain))
    # print("Done")
    # exit()
    #---------------

    # print("Getting sentinel data:")
    # for i in range(10):
    #     dataset[i]
    # print(len(dataset))


    # print(dataset[0])
    # print("TRAINING BEGINS")

    train(0, args)
    # args.pretrain = "/kaggle/input/preliminary-omni-dc-canopy-height/experiments/251203_035333_trial/model_best.pt"
    # test(args)
    

    # data_train = get_data(args, "train")
    # for param in net.parameters():
    #     param.requires_grad = False

    net = OGNIDC(args=args)
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