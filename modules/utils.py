#from data_preprocess_and_load.datasets import * #####
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from datetime import datetime
from pytz import timezone
import os
import dill
import random
import builtins
import time

def weight_loader(args):
    model_weights_path = None
    try:
        if args.step == '0':
            task = 'transformer_baseline'
        if args.step == '1' :
            task = 'autoencoder_reconstruction'          
        elif args.step == '2':
            task = 'transformer_reconstruction'
            if os.path.exists(args.model_weights_path_phase1):
                model_weights_path = args.model_weights_path_phase1 
        elif args.step == '3':
            task = 'fine_tune_{}'.format(args.fine_tune_task)
            if os.path.exists(args.model_weights_path_phase2):
                model_weights_path = args.model_weights_path_phase2
        elif args.step == '4':
            task = 'test'
            #task = 'fine_tune_{}'.format(args.fine_tune_task)
            if os.path.exists(args.model_weights_path_phase3):
                model_weights_path = args.model_weights_path_phase3
    except:
            #if no weights were provided
            model_weights_path = None 

    
    # print(f'loading weight from {model_weights_path}')
    return model_weights_path, args.step, task

def sort_pth_files(files_Path):
        file_name_and_time_lst = []
        for f_name in os.listdir(files_Path):
            if f_name.endswith('.pth'):
                written_time = os.path.getctime(os.path.join(files_Path,f_name))
                file_name_and_time_lst.append((f_name, written_time))
        # 생성시간 역순으로 정렬 
        # first file is the latest file. 
        sorted_file_lst = sorted(file_name_and_time_lst, key=lambda x: x[1], reverse=True)

        return sorted_file_lst
    
def datestamp():
    time = datetime.now(timezone('Asia/Seoul')).strftime("%m_%d__%H_%M_%S")
    return time

def reproducibility(seed, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        cudnn.deterministic = True
        cudnn.benchmark = True # If true, use the best algorithm (the value can be changed by changing the algorithm)

def sort_args(phase, args):
    phase_specific_args = {}
    for name, value in args.items():
        if not 'phase' in name:
            phase_specific_args[name] = value
        elif 'phase' + phase in name:
            phase_specific_args[name.replace('_phase' + phase, '')] = value
    return phase_specific_args

def args_logger(args):
    args_to_pkl(args)
    args_to_text(args)


def args_to_pkl(args):
    with open(os.path.join(args.experiment_folder,'arguments_as_is.pkl'),'wb') as f:
        #f.write(vars(args))
        dill.dump(vars(args),f)

def args_to_text(args):
    with open(os.path.join(args.experiment_folder,'argument_documentation.txt'),'w+') as f:
        for name,arg in vars(args).items():
            f.write('{}: {}\n'.format(name,arg))
