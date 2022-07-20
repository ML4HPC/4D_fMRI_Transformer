import numpy as np
import torch
from torch.utils.data import DataLoader,Subset, Dataset, RandomSampler

#DDP
from torch.utils.data.distributed import DistributedSampler
from data_preprocess_and_load.datasets import *
from utils import reproducibility
import os


# for data parallel
# torch.distributed.init_process_group(
#      backend='nccl', world_size=4, rank=int(os.environ["LOCAL_RANK"]), store=None)

class DataHandler():
    def __init__(self,test=False,**kwargs):
        self.test = test
        self.kwargs = kwargs
        self.dataset_name = kwargs.get('dataset_name')
        # self.num_val_samples = kwargs.get('num_val_samples')
        self.splits_folder = Path(kwargs.get('base_path')).joinpath('splits',self.dataset_name)
        self.splits_folder.mkdir(exist_ok=True)
        self.seed = kwargs.get('seed')
        self.current_split = self.splits_folder.joinpath('seed_{}.txt'.format(self.seed))
    
    #여기에 데이터셋 추가하면 됨.
    def get_dataset(self):
        if self.dataset_name == 'S1200':
            return rest_1200_3D
        elif self.dataset_name == 'ucla':
            return ucla
        else:
            raise NotImplementedError

    def current_split_exists(self):
        return self.current_split.exists() # 해당 폴더가 존재하는지 확인

    def create_dataloaders(self):
        reproducibility(**self.kwargs) #reproducibility를 위한 여러 설정을 위한 함수
        dataset = self.get_dataset() # self.dataset_name에 의해 결정
        train_loader = dataset(**self.kwargs) #이름은 loader이나 실제론 dataset class
        eval_loader = dataset(**self.kwargs)
        eval_loader.augment = None
        self.subject_list = train_loader.index_l
        #print('index_l:',self.subject_list)
        if self.current_split_exists():
            train_names, val_names, test_names = self.load_split()
            train_idx, val_idx, test_idx = self.convert_subject_list_to_idx_list(train_names,val_names,test_names,self.subject_list)
        else:
            train_idx,val_idx,test_idx = self.determine_split_randomly(self.subject_list,**self.kwargs)

        # train_idx = [train_idx[x] for x in torch.randperm(len(train_idx))[:1000]]
        # val_idx = [val_idx[x] for x in torch.randperm(len(val_idx))[:1000]]
        
        ## restrict to 1000 datasets
        #val_idx = list(np.random.choice(len(val_idx), self.num_val_samples, replace=False))
        
        #index를 통해 dataset의 일부를 가져오고싶을때 Subset 사용
        print('length of train_idx:', len(train_idx)) #900984
        print('length of val_idx:', len(val_idx)) #192473 -> 1000
        print('length of test_idx:', len(test_idx)) #194774
        
        train_loader = Subset(train_loader, train_idx)
        val_loader = Subset(eval_loader, val_idx)
        test_loader = Subset(eval_loader, test_idx)
        
        if self.kwargs.get('distributed'):
            train_sampler = DistributedSampler(train_loader , shuffle=True)
            valid_sampler = DistributedSampler(val_loader, shuffle=True)
            test_sampler = DistributedSampler(test_loader, shuffle=True)
        else:
            train_sampler = RandomSampler(train_loader)
            valid_sampler = RandomSampler(val_loader)
            test_sampler = RandomSampler(test_loader)
        
        
        
        ## Stella transformed this part ##
        training_generator = DataLoader(train_loader, **self.get_params(**self.kwargs),
                                       sampler=train_sampler)
        
        val_generator = DataLoader(val_loader, **self.get_params(eval=True,**self.kwargs),
                                  sampler=valid_sampler)
        
        test_generator = DataLoader(test_loader, **self.get_params(eval=True,**self.kwargs),
                                   sampler=test_sampler) if self.test else None
        
        
        return training_generator, val_generator, test_generator


    def get_params(self,eval=False,**kwargs):
        batch_size = kwargs.get('batch_size')
        workers = kwargs.get('workers')
        cuda = kwargs.get('cuda')
        if eval:
            workers = 0
        params = {'batch_size': batch_size,
                  #'shuffle': True,
                  'num_workers': workers,
                  'drop_last': True,
                  'pin_memory': True,  # True if cuda else False,
                  'persistent_workers': True if workers > 0 and cuda else False}
        return params

    def save_split(self,sets_dict):
        with open(self.current_split,'w+') as f:
            for name,subj_list in sets_dict.items():
                f.write(name + '\n')
                for subj_name in subj_list:
                    f.write(str(subj_name) + '\n')

    def convert_subject_list_to_idx_list(self,train_names,val_names,test_names,subj_list):
        subj_idx = np.array([str(x[0]) for x in subj_list])
        train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
        val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
        test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()
        return train_idx,val_idx,test_idx

    def determine_split_randomly(self,index_l,**kwargs):
        train_percent = kwargs.get('train_split')
        val_percent = kwargs.get('val_split')
        S = len(np.unique([x[0] for x in index_l]))
        S_train = int(S * train_percent)
        S_val = int(S * val_percent)
        S_train = np.random.choice(S, S_train, replace=False)
        remaining = np.setdiff1d(np.arange(S), S_train)
        S_val = np.random.choice(remaining,S_val, replace=False)
        S_test = np.setdiff1d(np.arange(S), np.concatenate([S_train,S_val]))
        train_idx,val_idx,test_idx = self.convert_subject_list_to_idx_list(S_train,S_val,S_test,self.subject_list)
        self.save_split({'train_subjects':S_train,'val_subjects':S_val,'test_subjects':S_test})
        return train_idx,val_idx,test_idx

    def load_split(self):
        subject_order = open(self.current_split, 'r').readlines()
        subject_order = [x[:-1] for x in subject_order]
        train_index = np.argmax(['train' in line for line in subject_order])
        val_index = np.argmax(['val' in line for line in subject_order])
        test_index = np.argmax(['test' in line for line in subject_order])
        train_names = subject_order[train_index + 1:val_index]
        val_names = subject_order[val_index+1:test_index]
        test_names = subject_order[test_index + 1:]
        return train_names,val_names,test_names
