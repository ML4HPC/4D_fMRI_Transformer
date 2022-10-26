from modules.utils import *  #including 'init_distributed', 'weight_loader'
from modules.trainer import Trainer
import os
from pathlib import Path
import torch

from config import get_arguments

#DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn import DataParallel
import builtins

#AMP
from torch.cuda.amp import GradScaler, autocast

# ASP
#from apex.contrib.sparsity import ASP


#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"



def setup_folders(base_path): 
    os.makedirs(os.path.join(base_path,'experiments'),exist_ok=True) 
    os.makedirs(os.path.join(base_path,'runs'),exist_ok=True)
    #os.makedirs(os.path.join(base_path, 'splits'), exist_ok=True)
    return None

def run_phase(args,loaded_model_weights_path,phase_num,phase_name):
    """
    main process that runs each training phase
    :return path to model weights (pytorch file .pth) aquried by the current training phase
    """
    experiment_folder = '{}_{}_{}_{}'.format(args.dataset_name,phase_name,args.target,args.exp_name) # datestamp() #05_02_20_05_12: 5월 2일 20시 05분 12초
    #experiment_folder = 'S1200_autoencoder_reconstruction_07_27__09_42_36'
    experiment_folder = Path(os.path.join(args.base_path,'experiments',experiment_folder))
    os.makedirs(experiment_folder, exist_ok=True)
    setattr(args,'loaded_model_weights_path_phase' + phase_num,loaded_model_weights_path)
    args.experiment_folder = experiment_folder
    args.experiment_title = experiment_folder.name
    
    
    fine_tune_task = args.fine_tune_task
    print(f'saving the results at {args.experiment_folder}')
    args_logger(args)
    args = sort_args(phase_num, vars(args))
    S = ['train','val']
    trainer = Trainer(sets=S,**args)
    trainer.training()
    if phase_num == '3' and not fine_tune_task == 'regression':
        critical_metric = 'accuracy'
    else:
        critical_metric = 'loss'
    model_weights_path = os.path.join(trainer.writer.experiment_folder,trainer.writer.experiment_title + '_BEST_val_{}.pth'.format(critical_metric))
    
    return model_weights_path


def test(args,phase_num,model_weights_path):
    experiment_folder = '{}_{}_{}'.format(args.dataset_name, 'test_{}'.format(args.fine_tune_task), args.exp_name) #, datestamp())
    experiment_folder = Path(os.path.join(args.base_path,'tests', experiment_folder))
    os.makedirs(experiment_folder,exist_ok=True)
    setattr(args,'loaded_model_weights_path_phase' + phase_num, model_weights_path) # 이름이 이게 맞나?
    
    args.experiment_folder = experiment_folder
    args.experiment_title = experiment_folder.name
    args_logger(args)
    args = sort_args(args.step, vars(args))
    S = ['test']
    #trainer = Trainer(experiment_folder, '3', args, ['test'], model_weights_path)
    trainer = Trainer(sets=S,**args)
    trainer.testing()
    
    if not args.fine_tune_task == 'regression':
        critical_metric = 'accuracy'
    else:
        critical_metric = 'loss'
    model_weights_path = os.path.join(trainer.writer.experiment_folder,trainer.writer.experiment_title + '_BEST_test_{}.pth'.format(critical_metric))

''' 기존 함수
def test(args,model_weights_path):
    experiment_folder = '{}_{}_{}'.format(args.dataset_name, 'test_{}'.format(args.fine_tune_task), datestamp())
    experiment_folder = os.path.join(args.base_path,'tests', experiment_folder)
    os.makedirs(experiment_folder,exist_ok=True)
    trainer = Trainer(experiment_folder, '3', args, ['test'], model_weights_path)
    trainer.testing()
'''

if __name__ == '__main__':
    base_path = os.getcwd() 
    setup_folders(base_path) 
    args = get_arguments(base_path)

    # DDP initialization
    init_distributed(args)

    # load weights that you specified at the Argument
    model_weights_path, step, task = weight_loader(args)

    if step == '4' :
        print(f'starting testing')
        phase_num = '4'
        test(args, phase_num, model_weights_path) # have some problems here - I checked it! -Stella 
    else:
        print(f'starting phase{step}: {task}')
        run_phase(args,model_weights_path,step,task)
        print(f'finishing phase{step}: {task}')
