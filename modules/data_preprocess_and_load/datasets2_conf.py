# 4D_fMRI_Transformer
import os
import torch
from torch.utils.data import Dataset

# import augmentations #commented out because of cv errors
import pandas as pd
from pathlib import Path
import numpy as np
import torchio as tio
import scipy


class BaseDataset(Dataset):
    def __init__(
            self,
            root='/mnt/ssd/processed/S1200',
            img_cache=None,
            sequence_length=20,
            use_augmentations=False,
            contrastive=False,
            with_voxel_norm=False
        ):
        super().__init__()

        # if use_augmentations or contrastive:
        #     raise NotImplementedError()
        self.cache = img_cache
        self.index_l = []
        self.data = self._set_data(root, sequence_length) # get list (element is tuple)
        self.with_voxel_norm = with_voxel_norm

    def load_sequence(self, subject_path, start_frame, session_duration):
        # shared cache
        # def  torchio_cache(path):
        #     if path not in self.cache:
        #         self.cache[path] = torch.load(path).unsqueeze(0)
        #     return self.cache[path]

        # y = []
        # load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+session_duration)]
        # if self.with_voxel_norm:
        #     load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
            
        # for fname in load_fnames:
        #     img_path = os.path.join(subject_path, fname)
        #     # shared cache
        #     y_i = torchio_cache(img_path)
        #     y.append(y_i)
        # y = torch.cat(y, dim=4)


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
        
#     def get_input_shape(self):
#         shape = torch.load(os.path.join(self.data[0][2],f'frame_{self.data[0][3]}.pt')).squeeze().shape
#         if self.with_voxel_norm:
#             shape = (2,) + shape
#         else:
#             shape = (1,) + shape
#         return shape
        

class S1200(BaseDataset):
    def __init__(
            self,
            root='/mnt/ssd/processed/S1200',
            img_cache=None,
            sequence_length=20,
            use_augmentations=False,
            contrastive=False,
            with_voxel_norm=False,
            dtype='float16',
        ):

        super(S1200, self).__init__(
            root=root,
            sequence_length=sequence_length,
            img_cache=img_cache,
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

        # Stella modified z-scoring part of the dataset2.py
        age_whole = self.meta_data_residual[['subject', 'age']].dropna(axis=0)
        cont_mean = age_whole['age'].mean()
        cont_std = age_whole['age'].std()

        for i, subject in enumerate(subject_list):
            try:
                age = torch.tensor(
                    (self.meta_data_residual[self.meta_data_residual["subject"] == int(subject)]["age"].values[0] - cont_mean)/cont_std
                )
            except Exception:
                # deal with discrepency that a few subjects don't have exact age, so we take the mean of the age range as the exact age proxy
                age = self.meta_data[self.meta_data["Subject"] == int(subject)]["Age"].values[0]
                age = np.array([float(x) for x in age.replace("+", "-").split("-")]).mean() # e.g. 22-25 -> 23.5
                age = torch.tensor(
                    (age-cont_mean)/cont_std
                )

        # for i, subject in enumerate(subject_list):
        #     try:
        #         age = torch.tensor(
        #             self.meta_data_residual[self.meta_data_residual["subject"] == int(subject)]["age"].values[0]
        #         )
        #     except Exception:
        #         # deal with discrepency that a few subjects don't have exact age, so we take the mean of the age range as the exact age proxy
        #         age = self.meta_data[self.meta_data["Subject"] == int(subject)]["Age"].values[0]
        #         age = torch.tensor([float(x) for x in age.replace("+", "-").split("-")]).mean() # e.g. 22-25 -> 23.5
                


            # age = torch.tensor(self.meta_data_residual[self.meta_data_residual["subject"] == int(subject)]["age"].values[0])
            sex = self.meta_data[self.meta_data["Subject"] == int(subject)]["Gender"].values[0]

            subject_path = os.path.join(img_root, subject)

            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - sequence_length + 1

            #####
            # Confounding variables
            conf_hcp_sub_path = '/lus/grand/projects/STlearn/HCP_confounding_variables/' + subject
            conf_hcp_sub_df = pd.read_csv(os.path.join(conf_hcp_sub_path, 'rfMRI_REST1_LR/Movement_Regressors.txt'), sep=r'\s+', header=None)
            conf_hcp_sub_df2 = conf_hcp_sub_df.iloc[20:]
            #####

            for start_frame in range(0, session_duration, sequence_length):
                data_tuple = (i, subject, subject_path, start_frame, sequence_length, age, sex, list(conf_hcp_sub_df2.iloc[start_frame,:])) #####
                data.append(data_tuple)
        return data

    def __getitem__(self, index):
        _, subject, subject_path, start_frame, sequence_length, age, sex, conf_list = self.data[index] #####
        age = self.label_dict[age] if isinstance(age, str) else age.float()

        y = self.load_sequence(subject_path, start_frame, sequence_length)
        
        background_value = y.flatten()[0]
        y = y.permute(0,4,1,2,3)
        y = torch.nn.functional.pad(y, (8, 7, 2, 1, 11, 10), value=background_value)
        y = y.permute(0,2,3,4,1)

        return {
            "fmri_sequence": y,
            "subject": subject,
            "sex": 1.0 if sex == 'M' else 0,
            "age": age,
            "TR": start_frame,
            "conf": conf_list, #####
        } 
        
class ABCD(BaseDataset):
    def __init__(
            self,
            root='/mnt/ssd/processed/ABCD',
            sequence_length=20,
            use_augmentations=False,
            contrastive=False,
            with_voxel_norm=False,
            dtype='float16',
            target = 'sex',
        ):
        self.dtype=dtype
        self.target = target
        super(ABCD, self).__init__(
            root=root,
            sequence_length=sequence_length,
            use_augmentations=use_augmentations,
            contrastive=contrastive,
            with_voxel_norm=with_voxel_norm,
            )

    def _set_data(self, root, sequence_length):
        data = []
        self.meta_data = pd.read_csv(os.path.join(root, "metadata", "ABCD_phenotype_total.csv"))
        img_root = os.path.join(root, 'img')
        subject_list = os.listdir(img_root)
        if self.target == 'sex': task_name = 'sex'
        elif self.target == 'age': task_name = 'age'
        elif self.target == 'int_total': task_name = 'nihtbx_totalcomp_uncorrected'
        elif self.target == 'int_fluid': task_name = 'nihtbx_fluidcomp_uncorrected'
        elif self.target == 'ASD': task_name = 'ASD_label'
        elif self.target == 'ADHD': task_name = 'ADHD_label'
        else: raise ValueError('downstream task not supported')

        # drop nan
        meta_task = self.meta_data[['subjectkey',task_name]].dropna()
        non_na_subjects = meta_task['subjectkey'].values
        subject_list = [subj for subj in subject_list if subj[4:] in non_na_subjects]
        
        if self.target == 'age':
            cont_mean = np.array(meta_task[task_name]).mean()
            cont_std = np.array(meta_task[task_name]).std()

        for i, subject in enumerate(subject_list):
            subject_name = subject[4:]

            #if subject_name in meta_task['subjectkey'].values:
            if self.target == 'age':
                target = (meta_task[meta_task["subjectkey"]==subject_name][task_name].values[0] - cont_mean)/cont_std
            else:
                target = meta_task[meta_task["subjectkey"]==subject_name][task_name].values[0]
            target = torch.tensor(target).type(torch.float) # Stella added this
            subject_path = os.path.join(img_root, subject)

            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - sequence_length + 1

            for start_frame in range(0, session_duration, sequence_length):
                data_tuple = (i, subject_name, subject_path, start_frame, sequence_length, target)
                data.append(data_tuple)
        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, target = self.data[index]
        #age = self.label_dict[age] if isinstance(age, str) else age.float()

        y = self.load_sequence(subject_path, start_frame, sequence_length)

        background_value = y.flatten()[0]
        y = y.permute(0,4,1,2,3)
        # ABCD image shape: 79, 97, 85
        y = torch.nn.functional.pad(y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:]
        y = y.permute(0,2,3,4,1)

        return {
            "fmri_sequence": y,
            "subject": subject_name,
            f"{self.target}": target,
            "TR": start_frame,
        } 


class ABCD_timeseries(BaseDataset):
    def __init__(
            self,
            root='/mnt/ssd/processed/ABCD',
            dtype='float16',
            use_augmentations=False,
            sequence_length=368,
            target = 'sex',
        ):
        self.dtype = dtype
        self.target = target
        super(ABCD_timeseries, self).__init__(
            root=root,
            sequence_length=sequence_length,
            use_augmentations=use_augmentations
            )


    def _set_data(self, root, sequence_length):
        data = []
        self.meta_data = pd.read_csv(os.path.join('/lus/grand/projects/STlearn/4D_fMRI_Transformer/data/metadata', "ABCD_phenotype_total.csv"))
        subject_list = os.listdir(root) # format : sub-NDARINVZRHTXMXD
        if self.target == 'sex': task_name = 'sex'
        elif self.target == 'age': task_name = 'age'
        elif self.target == 'int_total': task_name = 'nihtbx_totalcomp_uncorrected'
        elif self.target == 'int_fluid': task_name = 'nihtbx_fluidcomp_uncorrected'
        elif self.target == 'ASD': task_name = 'ASD_label'
        elif self.target == 'ADHD': task_name = 'ADHD_label'
        else: raise ValueError('downstream task not supported')

        # drop nan
        meta_task = self.meta_data[['subjectkey',task_name]].dropna() # format: NDARINV00BD7VDC
        non_na_subjects = meta_task['subjectkey'].values # format: NDARINV00BD7VDC
        subject_list = [subj for subj in subject_list if subj[4:] in non_na_subjects] # format: sub-NDARINV00BD7VDC
        if self.target == 'age':
            cont_mean = np.array(meta_task[task_name]).mean()
            cont_std = np.array(meta_task[task_name]).std()

        for i, subject in enumerate(subject_list):
            subject_name = subject[4:]
            #if subject_name in meta_task['subjectkey'].values:
            if self.target == 'age':
                target = (meta_task[meta_task["subjectkey"]==subject_name][task_name].values[0] - cont_mean)/cont_std
            else:
                target = meta_task[meta_task["subjectkey"]==subject_name][task_name].values[0]
            target = torch.tensor(target).type(torch.float) # Stella added this

            file_name = 'desikankilliany_'+subject+'.npy'   
            path_to_fMRIs = os.path.join(root, subject, file_name)

            data.append((i, subject, path_to_fMRIs, target))
        return data
            

    def __getitem__(self, index):
        _, subject, path_to_fMRIs, target = self.data[index]
        y = np.load(path_to_fMRIs)[20:].T # [84, 350 ~ 361]
        ts_length = y.shape[1]
        pad = 368-ts_length

        y = scipy.stats.zscore(y, axis=None) # (84, 350 ~ 361)
        y = torch.nn.functional.pad(torch.from_numpy(y), (pad//2, pad-pad//2), "constant", 0) # (84, 361)
        y = y.T.float() #.type(torch.DoubleTensor) # (361, 84)

        return {'fmri_sequence':y,'subject':subject, self.target:target}