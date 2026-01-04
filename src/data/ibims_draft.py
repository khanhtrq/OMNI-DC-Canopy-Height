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

gedi_folder = "/kaggle/input/gedi-canopy-height-hoanglien/canopy_data/GEDI"
sentinel_folder = "/kaggle/input/gedi-canopy-height-hoanglien/Sentinel-12band/Sentinel-12band"
regions = ["HoangLien", "CucPhuong", "BaBe"]
regions = ["CucPhuong", "BaBe"]

# gedi_folder = "E:\CEI - Carbon Stock\experiments\data\canopyheight_HoangLien\GEDI"
# sentinel_folder = "E:\CEI - Carbon Stock\experiments\data\canopyheight_HoangLien\Sentinel"
# regions = ["HoangLien"]


class iBims_Draft(BaseDataset):
    def __init__(self, args, mode):
        super(iBims_Draft, self).__init__(args, mode)

        self.args = args
        self.data_mode = args.backbone_mode
        self.mode = mode

        self.height = 480
        self.width = 640

        self.sen_max = [18192, 19056, 20224]
        self.sen_min = [0, 0, 0]
        self.sen_mean = [0.08068061134137483, 0.08042594856552854, 0.06597259674602372]
        self.sen_std = [0.018680129025066675, 0.012111958307734488, 0.008596667499397656]

        # print('Loading iBims-1...')
        # with open(split_txt, "r") as f:
        #     self.filenames = [
        #         s.split() for s in f.readlines()
        #     ]
        # print("LOADING DONE.")
        # print(self.filenames)
        # print(len(self.filenames))
        
        ratio_train = 0.8 

        self.sentinel_paths = []
        self.gedi_paths = []
        for r in regions:
            # self.gedi_paths += [os.path.join(r, file_name) for file_name in os.listdir(os.path.join(gedi_folder, r))]
            # self.sentinel_paths += [os.path.join(r, file_name) for file_name in os.listdir(os.path.join(sentinel_folder, r))]

            # filtering patches with not enough GEDI points
            gedi_paths_all = [os.path.join(r, file_name) for file_name in os.listdir(os.path.join(gedi_folder, r))]
            sentinel_paths_all = [os.path.join(r, file_name) for file_name in os.listdir(os.path.join(sentinel_folder, r))]
            for i in range(len(gedi_paths_all)):
                gedi_path = os.path.join(gedi_folder, gedi_paths_all[i])
                gedi = np.load(gedi_path)
                if np.sum(~np.isnan(gedi)) >= 50:
                    self.gedi_paths.append(gedi_paths_all[i])
                    self.sentinel_paths.append(sentinel_paths_all[i])

        #Spliting data into train and test set
        rng = np.random.default_rng(seed=42)   # fixed seed
        file_idx_all = rng.permutation(len(self.gedi_paths)) 

        if self.mode == "train":
            file_idx_train = file_idx_all[:int(ratio_train * len(self.gedi_paths))]
            self.file_idx = file_idx_train
        elif self.mode == "test" or self.mode == "val":
            file_idx_test = file_idx_all[int(ratio_train * len(self.gedi_paths)):]
            self.file_idx = file_idx_test

    def __len__(self):
        # return 32
        return len(self.file_idx)
    
    def __getitem__(self, idx):
        input_file_idx = self.file_idx[idx]

        gedi_path = os.path.join(gedi_folder, self.gedi_paths[input_file_idx])
        sentinel_path = os.path.join(sentinel_folder, self.sentinel_paths[input_file_idx])

        gedi = np.load(gedi_path)
        rgb = np.load(sentinel_path)

        gedi = gedi.astype(np.float32)
        rgb = rgb.astype(np.float32)

        # print("Max in raw gedi:", np.nanmax(gedi))

        # print(f"{rgb.max()}, {rgb.min()}")
        # print(rgb.dtype)

        # print("Gedi and RGB shapes:")
        # print(gedi.shape, rgb.shape)
        sen_max = []

        K = torch.eye(3)

        if 'rgb' in self.data_mode:
            t_rgb = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[1467.741681522291, 1532.5968758647118, 1334.2297965915839],
                            std=[339.82890722401294, 230.80547751218842, 173.8590035078182])
            ])
        elif 'sentinel' in self.data_mode:
            t_rgb = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[1445.2507821473719, 1494.0496883470857, 1695.0355912679454, 1564.6835706583588, 2027.4767477712417, 3214.2947198416723, 3624.9778571404872, 3675.896774446667, 3823.0191113997635, 3810.2311611275345, 2921.714671898899, 2096.318336982956],
                            std=[213.97085480597985, 236.57899563433898, 264.3780942144651, 334.3297911214344, 332.1163963382028, 504.0289211352316, 630.1565564561154, 696.8590235637502, 698.3457937838511, 635.1889167644749, 583.1581743442069, 504.8285260785615]
                )
            ])

        print("SHAPE:", rgb.shape)      

        t_rgb_np_raw = T.Compose([
            self.ToNumpy(),
        ])

        t_dep = T.Compose([
            # self.ToNumpy(),
            T.ToTensor()
        ])

        rgb_np_raw = t_rgb_np_raw(rgb)
        rgb = t_rgb(rgb)
        # print(f"{rgb.max()}, {rgb.min()}")

        dep = t_dep(gedi)

        dep_sp, pattern_id, mask_sp = self.get_sparse_depth(dep,
                                                   self.args.val_depth_pattern,
                                                   match_density=True,
                                                   rgb_np=rgb_np_raw,
                                                   input_noise=self.args.val_depth_noise,
                                                   return_mask= True)

        # print("Shape of sparse depth:", dep_sp.shape)
        # print("Number of points in sparse depth:", (dep_sp > 0).sum().item())
        # print("Number of points in ground truth depth:", (dep > 0).sum().item())
        # print("Type of sparse mask:", mask_sp.dtype)
        # print("Number of points in sparse mask:", mask_sp.sum().item())

        dep_ex_sp = dep * (~mask_sp.to(torch.bool)).type_as(dep)
        dep_ex_sp[dep_ex_sp == 0] = float('nan')

        # print("Number of points depth exclusive sparse:", (dep_ex_sp > 0).sum().item())
        # print("Max in depth exclusive sparse:", np.nanmax(dep_ex_sp.numpy()))
        # print("Max in depth sparse:", np.nanmax(dep_sp.numpy()))
        # print("Max all:", np.nanmax(dep.numpy()))

        # Return ground truth depth exclusive sparse points for evaluation
        # if self.mode == "test" or self.mode == "val":

        dep = dep_ex_sp
    

        # print("Number of points after excluding points in sparse depth:", (dep > 0).sum().item())

        # print("Nubmer of not nan values in sparse depth:", torch.sum(~torch.isnan(dep_sp)).item())
        # print("Number of not nan values in ground truth depth:", torch.sum(~torch.isnan(dep)).item())
        # print(">0:", (dep > 0).sum().item())

        # print("\n")
        # print("Max in raw rgb channels:", rgb_np_raw[0].max(), rgb_np_raw[1].max(), rgb_np_raw[2].max())
        # print("Min in raw rgb channels:", rgb_np_raw[0].min(), rgb_np_raw[1].min(), rgb_np_raw[2].min())
        # print("Mean in raw rgb channels:", rgb_np_raw[0].mean(), rgb_np_raw[1].mean(), rgb_np_raw[2].mean())
        # print("\n")
        # print("Max in rgb channels:", rgb[0].max(), rgb[1].max(), rgb[2].max())
        # print("Min in rgb channels:", rgb[0].min(), rgb[1].min(), rgb[2].min())
        # print("Mean in rgb channels:", rgb[0].mean(), rgb[1].mean(), rgb[2].mean())

        # rgb = rgb[0:1, :, :]  # only use band 1 (red band)
        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K, 'pattern': pattern_id}
        

        return output