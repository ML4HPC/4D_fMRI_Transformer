#fixed split
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # add parent path
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, Subset, RandomSampler
from .datasets2_conf import S1200, ABCD, ABCD_timeseries #####
from torch.utils.data.distributed import DistributedSampler
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .parser import str2bool
from utils import reproducibility
from multiprocessing import Manager

class fMRIDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.kwargs = kwargs
        self.seed = kwargs.get('seed') # 잘 들어감

        # generate splits folder
        split_dir_path = os.path.join(self.hparams.base_path,f'data/splits/{self.hparams.dataset_name}')
        os.makedirs(split_dir_path, exist_ok=True)
        if self.hparams.tff_split:
            self.split_file_path = os.path.join(split_dir_path, "TFF_split.txt")
        else:
            self.split_file_path = os.path.join(split_dir_path, f"split_fixed_{self.hparams.dataset_split_num}.txt")

        # pl.seed_everything(seed=self.hparams.data_seed)

    ##### data split generation #####
    def save_split(self, sets_dict):
        with open(self.split_file_path, "w+") as f:
            for name, subj_list in sets_dict.items():
                f.write(name + "\n")
                for subj_name in subj_list:
                    f.write(str(subj_name) + "\n")

    def get_dataset(self):
        if self.hparams.dataset_name == "S1200":
            return S1200
        elif self.hparams.dataset_name == "ABCD":
            return ABCD
        elif self.hparams.dataset_name == "ABCD_timeseries":
            return ABCD_timeseries
        else:
            raise NotImplementedError

    def convert_subject_list_to_idx_list(self, train_names, val_names, test_names, subj_list):
        subj_idx = np.array([str(x[1]) for x in subj_list])
        train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
        val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
        test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()
        return train_idx, val_idx, test_idx

    def determine_split_randomly(self, index_l):
        S = np.unique([x[1] for x in index_l]) #len(np.unique([x[1] for x in index_l]))
        S_train = int(len(S) * self.hparams.train_split)
        S_val = int(len(S) * self.hparams.val_split)
        S_train = np.random.choice(S, S_train, replace=False)
        remaining = np.setdiff1d(S, S_train) # np.setdiff1d(np.arange(S), S_train)
        S_val = np.random.choice(remaining, S_val, replace=False)
        S_test = np.setdiff1d(S, np.concatenate([S_train, S_val])) # np.setdiff1d(np.arange(S), np.concatenate([S_train, S_val]))
        train_idx, val_idx, test_idx = self.convert_subject_list_to_idx_list(S_train, S_val, S_test, self.subject_list)
        self.save_split({"train_subjects": S_train, "val_subjects": S_val, "test_subjects": S_test})
        return train_idx, val_idx, test_idx

    def load_split(self):
        subject_order = open(self.split_file_path, "r").readlines()
        subject_order = [x[:-1] for x in subject_order]
        train_index = np.argmax(["train" in line for line in subject_order])
        val_index = np.argmax(["val" in line for line in subject_order])
        test_index = np.argmax(["test" in line for line in subject_order])
        train_names = subject_order[train_index + 1 : val_index]
        val_names = subject_order[val_index + 1 : test_index]
        test_names = subject_order[test_index + 1 :]
        return train_names, val_names, test_names

    def prepare_data(self):
        # this function is only called at global rank==0
        return

    def setup(self, stage=None):
        #reproducibility(self.kwargs) # Stella added this (originially self.kwargs and it was running.. but why it doesn't work TT)
        # this function will be called at each devices
        Dataset = self.get_dataset()
        
        # shared cache
        # manager = Manager()
        # img_cache = manager.dict()

        if self.hparams.dataset_name == "S1200":
            params = {
                "root": self.hparams.image_path,
                "sequence_length": self.hparams.sequence_length,
                "with_voxel_norm": self.hparams.with_voxel_norm,
                # shared cache
                # "img_cache" : img_cache
            }
        elif self.hparams.dataset_name == "ABCD":
            params = {
                "root": self.hparams.image_path,
                "sequence_length": self.hparams.sequence_length,
                "with_voxel_norm": self.hparams.with_voxel_norm,
                "target": self.hparams.target
            }
        elif self.hparams.dataset_name == "ABCD_timeseries":
            params = {
                "root": self.hparams.image_path,
                "sequence_length": self.hparams.sequence_length,
                "target": self.hparams.target
            }
        dataset_w_aug = Dataset(**params, use_augmentations=True)
        dataset_wo_aug = Dataset(**params, use_augmentations=False)

        self.subject_list = dataset_w_aug.data
        self.subject_list = dataset_wo_aug.data
        if os.path.exists(self.split_file_path):
            print(f'loading split file : {self.split_file_path}')
            train_names, val_names, test_names = self.load_split()
            train_idx, val_idx, test_idx = self.convert_subject_list_to_idx_list(
                train_names, val_names, test_names, self.subject_list
            )
        else:
            #이걸 거치고 나면, train, val, test 가 0,0,0임.
            train_idx, val_idx, test_idx = self.determine_split_randomly(self.subject_list)
            # raise NameError('Use the split file')
            
            
        print("length of train_idx:", len(train_idx))  # 900984
        print("length of val_idx:", len(val_idx))  # 192473 -> 1000
        print("length of test_idx:", len(test_idx))  # 194774

        self.train_dataset = Subset(dataset_w_aug, train_idx)
        self.val_dataset = Subset(dataset_wo_aug, val_idx)
        self.test_dataset = Subset(dataset_wo_aug, test_idx)

        # DistributedSampler is internally called in pl.Trainer
        def get_params(train):
            return {
                "batch_size": self.hparams.batch_size,
                "num_workers": self.hparams.num_workers,
                "drop_last": True,
                "pin_memory": True,
                #"persistent_workers": train and (self.hparams.strategy == 'ddp'),
            }
        
        if self.hparams.distributed:
            train_sampler = DistributedSampler(self.train_dataset , shuffle=True)
            valid_sampler = DistributedSampler(self.val_dataset, shuffle=False)
            test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            train_sampler = RandomSampler(self.train_dataset)
            valid_sampler = None
            test_sampler = None
        
        self.train_loader = DataLoader(self.train_dataset, **get_params(train=True), sampler=train_sampler)
        self.val_loader = DataLoader(self.val_dataset, **get_params(train=False), sampler=valid_sampler)
        self.test_loader = DataLoader(self.test_dataset, **get_params(train=False), sampler=test_sampler)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_dataloader()

    @classmethod
    def add_data_specific_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=True, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("DataModule arguments")
        #group.add_argument("--data_seed", type=int, default=1, help='1,2,3')
        group.add_argument("--seed", type=int, default=27)
        group.add_argument("--dataset_split_num", type=int, default=1) # dataset split, choose from 1, 2, or 3
        group.add_argument("--dataset_name", type=str, choices=["S1200", "ABCD", "Dummy", "ABCD_timeseries"], default="S1200")
        group.add_argument('--target', type=str, default='sex', choices=['sex','age','ASD','ADHD','int_total','int_fluid'],help='fine_tune_task must be specified as follows -- {sex:classification, age:regression, ASD:classification, ADHD:classification, int_***:regression}')
        group.add_argument('--fine_tune_task',
                        default='binary_classification',
                        choices=['regression','binary_classification'],
                        help='fine tune model objective. choose binary_classification in case of a binary classification task')
        group.add_argument("--image_path", default="/pscratch/sd/s/stella/ABCD_TFF/MNI_to_TRs")
        group.add_argument("--train_split", default=0.7)
        group.add_argument("--val_split", default=0.15)
        group.add_argument("--num_subset", type=int, default=-1)
        

        # group.add_argument("--batch_size", type=int, default=4)
        group.add_argument("--sequence_length", type=int, default=20, choices = [20, 368])
        group.add_argument("--num_workers", type=int, default=8)

        # group.add_argument("--to_float16", type=str2bool, default=True)
        group.add_argument("--with_voxel_norm",action='store_true', default=False)
        group.add_argument("--use_augmentations", default=False, action='store_true')
        group.add_argument("--tff_split", action='store_true')

        # group.add_argument("--no_random_TR", default=False, action='store_true')
        # group.add_argument("--cuda", default=True)
        # group.add_argument("--base_path", default=os.getcwd())
        return parser
