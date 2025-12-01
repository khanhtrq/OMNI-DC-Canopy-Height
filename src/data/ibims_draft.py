import os
import tarfile
from io import BytesIO

import numpy as np
from scipy import io

from . import BaseDataset

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


from pathlib import Path
dataset_folder = "E:\CEI - Carbon Stock\experiments\data\IBims-1"

split_txt = "E:\CEI - Carbon Stock\experiments\data\IBims-1\imagelist.txt"

gedi_folder = "/kaggle/input/gedi-canopy-height-hoanglien/gedi_height"
sentinel_folder = "/kaggle/input/gedi-canopy-height-hoanglien/sentinel_image"
# gedi_folder = "E:\CEI - Carbon Stock\experiments\data\canopyheight_HoangLien\gedi_height"
# sentinel_folder = "E:\CEI - Carbon Stock\experiments\data\canopyheight_HoangLien\sentinel_image"


class iBims_Draft(BaseDataset):
    def __init__(self, args, mode):
        super(iBims_Draft, self).__init__(args, mode)

        self.args = args
        self.mode = mode

        self.height = 480
        self.width = 640

        # print('Loading iBims-1...')
        # with open(split_txt, "r") as f:
        #     self.filenames = [
        #         s.split() for s in f.readlines()
        #     ]
        # print("LOADING DONE.")
        # print(self.filenames)
        # print(len(self.filenames))

    def __len__(self):
        # return 32
        return len(os.listdir(gedi_folder))
    
    def __getitem__(self, idx):
        gedi_file = os.path.join(gedi_folder, f"{idx}.npy")
        sentinel_file = os.path.join(sentinel_folder, f"{idx}.npy")

        gedi = np.load(gedi_file)
        rgb = np.load(sentinel_file)

        gedi = gedi.astype(np.float32)
        rgb = rgb.astype(np.float32)
        # print(rgb.dtype)

        # print("Gedi and RGB shapes:")
        # print(gedi.shape, rgb.shape)


        K = torch.eye(3)

        t_rgb = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        t_rgb_np_raw = T.Compose([
            self.ToNumpy(),
        ])

        t_dep = T.Compose([
            # self.ToNumpy(),
            T.ToTensor()
        ])

        rgb_np_raw = t_rgb_np_raw(rgb)
        rgb = t_rgb(rgb)

        dep = t_dep(gedi)

        dep_sp, pattern_id = self.get_sparse_depth(dep,
                                                   self.args.val_depth_pattern,
                                                   match_density=True,
                                                   rgb_np=rgb_np_raw,
                                                   input_noise=self.args.val_depth_noise)

        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K, 'pattern': pattern_id}

        return output