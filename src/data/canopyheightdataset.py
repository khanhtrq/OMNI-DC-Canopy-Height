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

# gedi_folder = "/kaggle/input/gedi-canopy-height-hoanglien/canopy_data/GEDI"
gedi_folder = "/kaggle/input/gedi-canopy-height-hoanglien/GEDI_filtered/GEDI_filtered"

# sentinel_folder = "/kaggle/input/gedi-canopy-height-hoanglien/canopy_data/Sentinel"
# sentinel_folder = "/kaggle/input/gedi-canopy-height-hoanglien/Sentinel-12band/Sentinel-12band"
# sentinel_folder = "/kaggle/input/gedi-canopy-height-hoanglien/Sentinel-10band/Sentinel-10band"

sentinel_folder = "/kaggle/input/gedi-canopy-height-hoanglien/Sentinel-10band-SVD5/Sentinel-10band-SVD5"

regions = ["HoangLien", "CucPhuong", "BaBe"]
regions = ["CucPhuong", "BaBe"]

# gedi_folder = "E:\CEI - Carbon Stock\experiments\data\canopyheight_HoangLien\GEDI"
# sentinel_folder = "E:\CEI - Carbon Stock\experiments\data\canopyheight_HoangLien\Sentinel"
# regions = ["HoangLien"]


class CanopyHeightDataset(BaseDataset):
    def __init__(self, args, mode):
        super(CanopyHeightDataset, self).__init__(args, mode)

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

                # 12 bands
                # ------------
                # T.Normalize(mean=[1445.2507821473719, 1494.0496883470857, 1695.0355912679454, 1564.6835706583588, 2027.4767477712417, 3214.2947198416723, 3624.9778571404872, 3675.896774446667, 3823.0191113997635, 3810.2311611275345, 2921.714671898899, 2096.318336982956],
                #             std=[213.97085480597985, 236.57899563433898, 264.3780942144651, 334.3297911214344, 332.1163963382028, 504.0289211352316, 630.1565564561154, 696.8590235637502, 698.3457937838511, 635.1889167644749, 583.1581743442069, 504.8285260785615]
                # )

                # 10 bands
                #  excluduing band 1 and band 9
                # ------------
                # T.Normalize(mean=[1494.0496883470857, 1695.0355912679454, 1564.6835706583588, 2027.4767477712417, 3214.2947198416723, 3624.9778571404872, 3675.896774446667, 3810.2311611275345, 2921.714671898899, 2096.318336982956],
                #             std=[236.57899563433898, 264.3780942144651, 334.3297911214344, 332.1163963382028, 504.0289211352316, 630.1565564561154, 696.8590235637502, 635.1889167644749, 583.1581743442069, 504.8285260785615]
                # )

                # SVD 5 - 10 bands
                T.Normalize(mean=[8.78252562e+03, 8.31726226e+00, 1.05245297e+00, 2.84851171e-01, 2.91621488e-03],
                            std=[1447.67461915, 716.00579042, 327.87595153, 168.70943227, 102.73989055]
                )   
                
            ])
            rgb = rgb.transpose(1, 2, 0)

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

        # Exclude sparse points from ground truth depth, for validation and testing
        dep_ex_sp = dep * (~mask_sp.to(torch.bool)).type_as(dep)
        dep_ex_sp[dep_ex_sp == 0] = float('nan')

        # Jan 23, 2026: all GEDI points as ground truth
        # for validation and testing 
        # ------------
        if self.mode == "test" or self.mode == "val":
            dep = dep_ex_sp

        # rgb = rgb[0:1, :, :]  # only use band 1 (red band)
        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K, 'pattern': pattern_id}
        

        return output