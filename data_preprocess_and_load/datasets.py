#4D_fMRI_Transformer
import os
import torch
from torch.utils.data import Dataset, IterableDataset
# import augmentations #commented out because of cv errors
import pandas as pd
from pathlib import Path


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
    def register_args(self,**kwargs):
        #todo:decide if keep immedieate load or not
        self.device = None#torch.device('cuda') if kwargs.get('cuda') else torch.device('cpu')
        self.index_l = []
        self.norm = 'global_normalize'
        self.complementary = 'per_voxel_normalize'
        self.random_TR = kwargs.get('random_TR')
        self.target = kwargs.get('target')
        self.fine_tune_task = kwargs.get('fine_tune_task')
        
        self.set_augmentations(**kwargs)
        self.stride_factor = 1
        self.sequence_stride = 1 # 어느 정도 주기로 volume을 샘플링 할 것인가
        self.sequence_length = kwargs.get('sequence_length') # 몇 개의 volume을 사용할 것인가(STEP1:1,STEP2:20,STEP3:20 마다 다름)
        self.sample_duration = self.sequence_length * self.sequence_stride #샘플링하는 대상이 되는 구간의 길이 
        self.stride = max(round(self.stride_factor * self.sample_duration),1) # sequence lenghth 만큼씩 이동해서 rest_1200_3D의 init 파트의 for문에서 TR index를 불러옴
        self.TR_skips = range(0,self.sample_duration,self.sequence_stride)

    def get_input_shape(self):
        shape = torch.load(os.path.join(self.index_l[0][2],self.index_l[0][3] + '.pt')).squeeze().shape
        return shape

    def set_augmentations(self,**kwargs):
        if kwargs.get('augment_prob') > 0:
            self.augment = augmentations.brain_gaussian(**kwargs)
        else:
            self.augment = None

    def TR_string(self,filename_TR,x):
        #all datasets should have the TR mentioned in the format of 'some prefix _ number.pt'
        TR_num = [xx for xx in filename_TR.split('_') if xx.isdigit()][0]
        #assert len(filename_TR.split('_')) == 2
        filename = filename_TR.replace(TR_num,str(int(TR_num) + x)) + '.pt'
        # print('filename:',filename)
        return filename

    def determine_TR(self,TRs_path,TR):
        if self.random_TR: #no sliding window
            possible_TRs = len(os.listdir(TRs_path)) - self.sample_duration
            #TR = 'TR_' + str(torch.randint(0,possible_TRs,(1,)).item())
            TR = 'rfMRI_LR_TR_' + str(torch.randint(0,possible_TRs,(1,)).item())
        return TR

    def load_sequence(self, TRs_path, TR):
        # the logic of this function is that always the first channel corresponds to global norm and if there is a second channel it belongs to per voxel.
        TR = self.determine_TR(TRs_path,TR)
        #TR이 가능한 범위에서 뽑은 하나의 인덱스, 그리고 그로부터 TR_skips 만큼 떨어진 구간까지의 volume들을 하나의 시퀀스로 합쳐줌.
        y = torch.cat([torch.load(os.path.join(TRs_path, self.TR_string(TR, x)),map_location=self.device).unsqueeze(0) for x in self.TR_skips], dim=4)
        
        #만약 여러 normalization method를 거친 이미지들을 합치고 싶은 경우에 4번째(time) dimension을 기준으로 합쳐줌.
        if self.complementary is not None:
            y1 = torch.cat([torch.load(os.path.join(TRs_path, self.TR_string(TR, x)).replace(self.norm, self.complementary),map_location=self.device).unsqueeze(0)
                            for x in self.TR_skips], dim=4)
            y1[y1!=y1] = 0
            y = torch.cat([y, y1], dim=0)
            del y1
        return y

class rest_1200_3D(BaseDataset):
    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        #self.root = r'../TFF/'
        self.data_dir = kwargs.get('image_path')
        self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','HCP_1200_gender.csv'))
        self.meta_data_residual = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','HCP_1200_precise_age.csv'))
        # self.data_dir = os.path.join(self.root, 'MNI_to_TRs')
        self.subject_names = os.listdir(self.data_dir)
        self.label_dict = {'F': torch.tensor([0.0]), 'M': torch.tensor([1.0]), '22-25': torch.tensor([1.0, 0.0]),
                           '26-30': torch.tensor([1.0, 0.0]),
                           '31-35': torch.tensor([0.0, 1.0]), '36+': torch.tensor([0.0, 1.0])}  # torch.tensor([1])}
        self.subject_folders = []
        for i,subject in enumerate(os.listdir(self.data_dir)):
            try: 
                age = torch.tensor(self.meta_data_residual[self.meta_data_residual['subject']==int(subject)]['age'].values[0])
            except Exception:
                #deal with discrepency that a few subjects don't have exact age, so we take the mean of the age range as the exact age proxy
                age = self.meta_data[self.meta_data['Subject'] == int(subject)]['Age'].values[0]
                age = torch.tensor([float(x) for x in age.replace('+','-').split('-')]).mean()
            sex = self.meta_data[self.meta_data['Subject']==int(subject)]['Gender'].values[0]
            path_to_TRs = os.path.join(self.data_dir,subject,self.norm) # self.norm == global_normalize
            subject_duration = len(os.listdir(path_to_TRs)) #sequence length of the subject
            session_duration = subject_duration - self.sample_duration # 샘플링하는 길이만큼을 빼주어야 처음부터 sequence length - sample_duration 까지의 index를 샘플 가능. subsequence의 시작 index가 될 수 있는 인덱스들
            filename = os.listdir(path_to_TRs)[0]
            filename = filename[:filename.find('TR')+3]
            
            #이 부분이 결정적으로 샘플링하는 부분
            for k in range(0,session_duration,self.stride):
                self.index_l.append((i, subject, path_to_TRs,filename + str(k),session_duration, age , sex))

    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subj, subj_name, path_to_TRs, TR , session_duration, age, sex = self.index_l[index]
        age = self.label_dict[age] if isinstance(age,str) else age.float()
        y = self.load_sequence(path_to_TRs,TR)
        if self.augment is not None:
            y = self.augment(y)
        return {'fmri_sequence':y,'subject':subj,'sex':self.label_dict[sex],'age':age,'TR':int(TR.split('_')[3])} # {'fmri_sequence':y,'subject':subj,'subject_binary_classification':self.label_dict[sex],'subject_regression':age,'TR':int(TR.split('_')[3])}

class ABCD_3D(BaseDataset):
    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        #self.root = r'../TFF/'
        self.data_dir = kwargs.get('image_path')
        self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','ABCD_phenotype_total.csv'))
        self.subject_names = os.listdir(self.data_dir)
        self.subject_folders = []
        
        # ABCD 에서 target value가 결측값인 샘플 제거
        non_na = self.meta_data[['subjectkey',self.target]].dropna(axis=0)
        
        #voxel normalize가 덜 된 subject 제거
        mask=non_na['subjectkey'].isin(['NDARINVRTD32ZG1','NDARINVAAV56RVU','NDARINVRTDH8349','NDARINVAAPJB31X','NDARINVAAX7P792','NDARINVRTDZTY9C','NDARINVAAR0XGYL']+['NDARINV425E5RC6','NDARINVF9ZWE4J9','NDARINVTJJXH8K7','NDARINV7PB55MX2'])
        non_na = non_na[~mask]
        subjects = list(non_na['subjectkey']) 
        
        if self.fine_tune_task == 'regression':
            cont_mean = non_na[self.target].mean()
            cont_std = non_na[self.target].std()
        for i,subject in enumerate(os.listdir(self.data_dir)):
            if subject in subjects:
                # Normalization
                if self.fine_tune_task == 'regression':
                    target = torch.tensor((self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0] - cont_mean) / cont_std)
                    target = target.float()
                elif self.fine_tune_task == 'binary_classification':
                    target = torch.tensor(self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0]) 
                path_to_TRs = os.path.join(self.data_dir,subject,self.norm) # self.norm == global_normalize
                subject_duration = len(os.listdir(path_to_TRs)) #sequence length of the subject
                session_duration = subject_duration - self.sample_duration
                # 샘플링하는 길이만큼을 빼주어야 처음부터 sequence length - sample_duration 까지의 index를 샘플 가능. 
                if self.sample_duration > session_duration - 10:
                    continue
                
                filename = os.listdir(path_to_TRs)[0]
                filename = filename[:filename.find('TR')+3] 
            
                #이 부분이 결정적으로 샘플링하는 부분
                # for k in range(0,session_duration,self.stride):
                for k in range(10,session_duration,self.stride):
                    self.index_l.append((i, subject, path_to_TRs,filename + str(k),session_duration, target))
        # print('len(index_l):',len(self.index_l))
    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subj, subj_name, path_to_TRs, TR , session_duration, target = self.index_l[index]
        y = self.load_sequence(path_to_TRs,TR)
        if self.augment is not None:
            y = self.augment(y)
        return {'fmri_sequence':y,'subject':subj,'subject_name':subj_name,self.target:target,'TR':int(TR.split('_')[2])} #{'fmri_sequence':y,'subject':subj,'subject_binary_classification':sex,'subject_regression':age,'TR':int(TR.split('_')[2])}
    
    def determine_TR(self,TRs_path,TR):
        if self.random_TR: #no sliding window
            possible_TRs = len(os.listdir(TRs_path)) - self.sample_duration
            #TR = 'TR_' + str(torch.randint(0,possible_TRs,(1,)).item())
            #TR = 'rfMRI_TR_' + str(torch.randint(0,possible_TRs,(1,)).item())
            TR = 'rfMRI_TR_' + str(torch.randint(10,possible_TRs,(1,)).item())
        return TR

class DummyDataset(BaseDataset):
    # (74,95,80) : same as ABCD
    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        self.sequence_length = kwargs.get('sequence_length')
        self.total_samples = 1000
        self.y = torch.randn((self.total_samples, 2, 74, 95, 80, self.sequence_length))
        self.sex = torch.randint(0,2,(self.total_samples,))
        self.age = torch.randn(self.total_samples)
        self.TR = torch.randint(20,300,(self.total_samples,))
        for k in range(0,self.total_samples):
            self.index_l.append((k, 'subj'+ str(k), self.TR[k], self.age[k], self.sex[k]))
            
    def __len__(self):
        return self.total_samples

    def __getitem__(self,idx):
        seq_idx, subj, TR, age, sex = self.index_l[idx]
        y = self.y[seq_idx]
        return {'fmri_sequence':y,'subject':subj,'sex':sex,'age':age,'TR':TR}
            
    def get_input_shape(self):
        shape = (74, 95, 80)
        return shape
