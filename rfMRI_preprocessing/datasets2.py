# 4D_fMRI_Transformer
import os
import torch
from torch.utils.data import Dataset

# import augmentations #commented out because of cv errors
import pandas as pd
from pathlib import Path
import numpy as np
import torchio as tio


class BaseDataset(Dataset):
    def __init__(
            self,
            root='/mnt/ssd/processed/S1200',
            sequence_length=20,
            use_augmentations=False,
            contrastive=False,
            with_voxel_norm=False
        ):
        super().__init__()

        # if use_augmentations or contrastive:
        #     raise NotImplementedError()
        self.data = self._set_data(root, sequence_length)
        self.with_voxel_norm = with_voxel_norm

    def load_sequence(self, subject_path, start_frame, session_duration):
        y = []
        load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+session_duration)]
        if self.with_voxel_norm:
            load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
            
        for fname in load_fnames:
            img_path = os.path.join(subject_path, fname)
            y_i = torch.load(img_path).unsqueeze(0)
            y.append(y_i)
        y = torch.cat(y, dim=4)
        return y

    def __len__(self):
        return  len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError("Required function")

    def _set_data(self, root, sequence_length):
        raise NotImplementedError("Required function")
        

class S1200(BaseDataset):
    def __init__(
            self,
            root='/mnt/ssd/processed/S1200',
            sequence_length=20,
            use_augmentations=False,
            contrastive=False,
            with_voxel_norm=False,
            dtype='float16',
        ):

        super(S1200, self).__init__(
            root=root,
            sequence_length=sequence_length,
            use_augmentations=use_augmentations,
            contrastive=contrastive,
            with_voxel_norm=with_voxel_norm)

        self.dtype=dtype

    def _set_data(self, root, sequence_length):
        data = []
        self.meta_data = pd.read_csv(os.path.join(root, "metadata", "HCP_1200_gender.csv"))
        self.meta_data_residual = pd.read_csv(os.path.join(root, "metadata", "HCP_1200_precise_age.csv"))

        img_root = os.path.join(root, 'img')
        subject_list = os.listdir(img_root)
        for i, subject in enumerate(subject_list):
            try:
                age = torch.tensor(
                    self.meta_data_residual[self.meta_data_residual["subject"] == int(subject)]["age"].values[0]
                )
            except Exception:
                # deal with discrepency that a few subjects don't have exact age, so we take the mean of the age range as the exact age proxy
                age = self.meta_data[self.meta_data["Subject"] == int(subject)]["Age"].values[0]
                age = torch.tensor([float(x) for x in age.replace("+", "-").split("-")]).mean()

            # age = torch.tensor(self.meta_data_residual[self.meta_data_residual["subject"] == int(subject)]["age"].values[0])
            sex = self.meta_data[self.meta_data["Subject"] == int(subject)]["Gender"].values[0]

            subject_path = os.path.join(img_root, subject)

            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - sequence_length + 1

            for start_frame in range(0, session_duration, sequence_length):
                data_tuple = (i, subject, subject_path, start_frame, sequence_length, age, sex)
                data.append(data_tuple)
        return data

    def __getitem__(self, index):
        _, subject, subject_path, start_frame, sequence_length, age, sex = self.data[index]
        age = self.label_dict[age] if isinstance(age, str) else age.float()

        y = self.load_sequence(subject_path, start_frame, sequence_length)

        y = y.permute(0,4,1,2,3)
        # TODO: background_value is also padded on the mean and std value. Should be modified that the mean and std value should pad their boundary
        # background_value = y.flatten()[0]
        # y = torch.nn.functional.pad(y, (8, 7, 2, 1, 11, 10), value=background_value)
        # TODO: the above TODO is fixed by inserting mode='replicate' in functinoal.pad.
        y = torch.nn.functional.pad(y, (8, 7, 2, 1, 11, 10), mode='replicate')

        y = y.permute(0,2,3,4,1)

        return {
            "fmri_sequence": y,
            "subject": subject,
            "sex": 1.0 if sex == 'M' else 0,
            "age": age,
            "TR": start_frame,
        } 

