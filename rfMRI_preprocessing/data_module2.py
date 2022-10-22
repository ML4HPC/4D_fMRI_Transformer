import os
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, Subset
from .data_preprocess_and_load.datasets2 import S1200
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .parser import str2bool

class fMRIDataModule2(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # generate splits folder
        split_dir_path = f'./data/splits/{self.hparams.dataset_name}'
        os.makedirs(split_dir_path, exist_ok=True)
        self.split_file_path = os.path.join(split_dir_path, f"seed_{self.hparams.data_seed}.txt")

        pl.seed_everything(seed=self.hparams.data_seed)

    def save_split(self, sets_dict):
        with open(self.split_file_path, "w+") as f:
            for name, subj_list in sets_dict.items():
                f.write(name + "\n")
                for subj_name in subj_list:
                    f.write(str(subj_name) + "\n")

    def get_dataset(self):
        if self.hparams.dataset_name == "S1200":
            return S1200
        else:
            raise NotImplementedError

    def convert_subject_list_to_idx_list(self, train_names, val_names, test_names, subj_list):
        subj_idx = np.array([str(x[0]) for x in subj_list])
        train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
        val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
        test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()
        return train_idx, val_idx, test_idx

    def determine_split_randomly(self, index_l):
        S = len(np.unique([x[0] for x in index_l]))
        S_train = int(S * self.hparams.train_split)
        S_val = int(S * self.hparams.val_split)
        S_train = np.random.choice(S, S_train, replace=False)
        remaining = np.setdiff1d(np.arange(S), S_train)
        S_val = np.random.choice(remaining, S_val, replace=False)
        S_test = np.setdiff1d(np.arange(S), np.concatenate([S_train, S_val]))
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
        # this function will be called at each devices
        Dataset = self.get_dataset()
        params = {
            "root": self.hparams.image_path,
            "sequence_length": self.hparams.sequence_length,
            "with_voxel_norm": self.hparams.with_voxel_norm
        }
        dataset_w_aug = Dataset(**params, use_augmentations=True)
        dataset_wo_aug = Dataset(**params, use_augmentations=False)

        self.subject_list = dataset_w_aug.data
        if os.path.exists(self.split_file_path):
            train_names, val_names, test_names = self.load_split()
            train_idx, val_idx, test_idx = self.convert_subject_list_to_idx_list(
                train_names, val_names, test_names, self.subject_list
            )
        else:
            train_idx, val_idx, test_idx = self.determine_split_randomly(self.subject_list)

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
                "persistent_workers": train and (self.hparams.strategy == 'ddp'),
                "shuffle": train
            }
        self.train_loader = DataLoader(self.train_dataset, **get_params(train=True))
        self.val_loader = DataLoader(self.val_dataset, **get_params(train=False))
        self.test_loader = DataLoader(self.test_dataset, **get_params(train=False))

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
        group.add_argument("--data_seed", type=int, default=1234)
        group.add_argument("--dataset_name", type=str, choices=["S1200", "ABCD", "Dummy"], default="ABCD")
        group.add_argument("--image_path", default="/pscratch/sd/s/stella/ABCD_TFF/MNI_to_TRs")
        group.add_argument("--train_split", default=0.7)
        group.add_argument("--val_split", default=0.15)
        group.add_argument("--num_subset", type=int, default=-1)

        group.add_argument("--batch_size", type=int, default=4)
        group.add_argument("--sequence_length", default=20)
        group.add_argument("--num_workers", type=int, default=8)

        group.add_argument("--to_float16", type=str2bool, default=False)
        group.add_argument("--with_voxel_norm", type=str2bool, default=False)
        group.add_argument("--use_augmentations", default=False, action='store_true')

        # group.add_argument("--no_random_TR", default=False, action='store_true')
        # group.add_argument("--cuda", default=True)
        # group.add_argument("--base_path", default=os.getcwd())
        return parser
