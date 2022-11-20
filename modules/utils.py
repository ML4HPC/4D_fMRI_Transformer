#from data_preprocess_and_load.datasets import * #####
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from datetime import datetime
from pytz import timezone
import argparse
import os
import dill
import random
import builtins
import time

def _get_sync_file():    
        """Logic for naming sync file using slurm env variables"""
        if 'SCRATCH' in os.environ:
            sync_file_dir = '%s/pytorch-sync-files' % os.environ['SCRATCH'] # Perlmutter
        else:
            sync_file_dir = '%s/pytorch-sync-files' % '/lus/grand/projects/STlearn/'
            #raise Exception('there is no env variable SCRATCH. Please check sync_file dir')
        os.makedirs(sync_file_dir, exist_ok=True)

        #temporally add two lines below for torchrun
        if ('SLURM_JOB_ID' in os.environ) and ('SLURM_STEP_ID' in os.environ) :
            sync_file = 'file://%s/pytorch_sync.%s.%s' % (
            sync_file_dir, os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID'])
        else:
            sync_file = 'file://%s/pytorch_sync.%s.%s' % (
            sync_file_dir, '10004', '10003')
        return sync_file
    
    
def init_distributed(args):   
    
    # torchrun: sbatch script에서 WORLD_SIZE를 지정해준 경우 (노드 당 gpu * 노드의 수)
    if "WORLD_SIZE" in os.environ: # for torchrun
        args.world_size = int(os.environ["WORLD_SIZE"])
        #print('args.world_size:',args.world_size)
    elif 'SLURM_NTASKS' in os.environ: # for slurm scheduler
        args.world_size = int(os.environ['SLURM_NTASKS'])
    else:
        pass # torch.distributed.launch
        
    args.distributed = args.world_size > 1 # default: world_size = -1 
    ngpus_per_node = torch.cuda.device_count()
    
    if args.distributed:
        start_time = time.time()
        #args.local_rank = int(os.environ['LOCAL_RANK']) #stella added this line
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'RANK' in os.environ: # for torchrun
            args.rank = int(os.environ['RANK'])
            args.gpu = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        
        #print('args.rank:',args.rank)
        #print('args.gpu:',args.gpu)    
        if args.init_method == 'file':
            sync_file = _get_sync_file()
            #print('initializing DDP with sync file')
        elif args.init_method == 'env':
            sync_file = "env://"
            #os.environ['MASTER_PORT'] = '10025'
            #os.environ['MASTER_ADDR'] = '127.0.0.1'# os.environ['SLURM_JOB_NODELIST']
            #print(os.environ['MASTER_PORT'])
            #print(os.environ['MASTER_ADDR'])
            #print('initializing DDP with env variables')
        dist.init_process_group(backend=args.dist_backend, init_method=sync_file,
                            world_size=args.world_size, rank=args.rank)
        dist_init_time = time.time() - start_time
        #print(f'seconds taken for DDP initialization: {dist_init_time}')
    else:
        args.rank = 0
        args.gpu = 0

    # suppress printing if not on master gpu

    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
        
def weight_loader(args):
    model_weights_path = None
    try:
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

def reproducibility(**kwargs):
    seed = kwargs.get('seed')
    cuda = kwargs.get('cuda')
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = False #True
    cudnn.benchmark = True

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
