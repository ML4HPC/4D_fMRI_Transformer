from utils import *
from trainer import Trainer
import os
from pathlib import Path
import torch

#DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn import DataParallel
import builtins

#AMP
from torch.cuda.amp import GradScaler, autocast

# ASP
#from apex.contrib.sparsity import ASP


# for data parallel
# torch.distributed.init_process_group(
#      backend='nccl', world_size=4, rank=int(os.environ["LOCAL_RANK"]), store=None)

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

def get_arguments(base_path):
    """
    handle arguments from commandline.
    some other hyper parameters can only be changed manually (such as model architecture,dropout,etc)
    notice some arguments are global and take effect for the entire three phase training process, while others are determined per phase
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str,default="baseline") 
    parser.add_argument('--dataset_name', type=str, choices=['S1200','ABCD','Dummy'],default="S1200") 
    parser.add_argument('--image_path', default='./MNI_to_TRs') #perlmutetr: MNI_to_TRs, neuron: samples # /pscratch/sd/s/stella/ABCD_TFF/MNI_to_TRs for ABCD
    parser.add_argument('--base_path', default=base_path)
    parser.add_argument('--step', default='1', choices=['1','2','3'], help='which step you want to run')
    parser.add_argument('--target', type=str, default='sex', choices=['sex','age','ASD_label','ADHD_label','nihtbx_totalcomp_uncorrected','nihtbx_fluidcomp_uncorrected'],help='fine_tune_task must be specified as follows -- {sex:classification, age:regression, ASD_label:classification, ADHD_label:classification, nihtbx_***:regression}')
    parser.add_argument('--fine_tune_task',
                        default='binary_classification',
                        choices=['regression','binary_classification'],
                        help='fine tune model objective. choose binary_classification in case of a binary classification task')
    parser.add_argument('--seed', type=int, default=55555555)
    parser.add_argument('--num_val_samples', type=int, default=1000) #10000이 default. 변화 없음.
    parser.add_argument("--resume", action='store_true', help = 'if you add this option in the command line like --resume, args.resume would change to be True')
    parser.set_defaults(resume=False)
    
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--log_dir', type=str, default=os.path.join(base_path, 'runs')) #로그는 runs에 저장되는데..?
    parser.add_argument('--random_TR', action='store_false') #True면(인자를 넣어주지 않으면) 전체 sequence 로부터 random sampling(default). False면 (--random_TR 인자를 넣어주면) 0번째 TR부터 sliding window
    
    parser.add_argument('--intensity_factor', default=1)
    parser.add_argument('--perceptual_factor', default=1)
    parser.add_argument('--reconstruction_factor', default=1)
    parser.add_argument('--transformer_hidden_layers', default=2)
    parser.add_argument('--train_split', default=0.7)
    parser.add_argument('--val_split', default=0.15)
    parser.add_argument('--running_mean_size', default=5000)
    
    # DDP configs:
    parser.add_argument('--world_size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')

    # AMP configs:
    parser.add_argument('--amp', action='store_false')
    parser.add_argument('--gradient_clipping', action='store_true')
    #parser.add_argument('--opt_level', default='O1', type=str,
    #                    help='opt level of amp. O1 is recommended')
    
    # Gradient accumulation
    parser.add_argument("--accumulation_steps", default=1, type=int,required=False,help='mini batch size == accumulation_steps * args.train_batch_size')
    
    # Nsight profiling
    parser.add_argument("--profiling", action='store_true')
    
    
    ##phase 1
    parser.add_argument('--task_phase1', type=str, default='autoencoder_reconstruction')
    parser.add_argument('--batch_size_phase1', type=int, default=8, help='for DDP, each GPU processes batch_size_pahse1 samples') #이걸.. 잘게 쪼개볼까? 원래는 4였음.
    parser.add_argument('--validation_frequency_phase1', type=int, default=10000000) # 11 for test #original: 10000) #원래는 1000이었음 -> 약 7분 걸릴 예정.
    parser.add_argument('--nEpochs_phase1', type=int, default=20) #epoch는 10개인 걸로~
    parser.add_argument('--augment_prob_phase1', default=0)
    parser.add_argument('--optim_phase1', default='Adam')
    parser.add_argument('--weight_decay_phase1', default=1e-7)
    parser.add_argument('--lr_policy_phase1', default='step', choices=['step','SGDR'], help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase1', type=float, default=1e-3)
    parser.add_argument('--lr_gamma_phase1', type=float, default=0.97)
    parser.add_argument('--lr_step_phase1', type=int, default=500)
    parser.add_argument('--lr_warmup_phase1', type=int, default=500)
    parser.add_argument('--sequence_length_phase1', default=1)
    parser.add_argument('--workers_phase1', default=4)

    ##phase 2
    parser.add_argument('--task_phase2', type=str, default='transformer_reconstruction')
    parser.add_argument('--batch_size_phase2', type=int, default=4) #원래는 1이었음
    parser.add_argument('--validation_frequency_phase2', type=int, default=10000000) # 11 for test original: 10000) #원래는 500이었음
    parser.add_argument('--optim_phase2', default='Adam')
    parser.add_argument('--nEpochs_phase2', type=int, default=20)
    parser.add_argument('--augment_prob_phase2', default=0)
    parser.add_argument('--weight_decay_phase2', default=1e-7)
    parser.add_argument('--lr_policy_phase2', default='step', choices=['step','SGDR'], help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase2', type=float, default=1e-3)
    parser.add_argument('--lr_gamma_phase2', type=float, default=0.97)
    parser.add_argument('--lr_step_phase2', type=int, default=1000)
    parser.add_argument('--lr_warmup_phase2', type=int, default=500)
    parser.add_argument('--sequence_length_phase2', default=20)
    parser.add_argument('--workers_phase2', default=4)
    parser.add_argument('--model_weights_path_phase1', default=None)

    ##phase 3
    parser.add_argument('--task_phase3', type=str, default='fine_tune')
    parser.add_argument('--batch_size_phase3', type=int, default=4) #원래는 3이었음
    parser.add_argument('--validation_frequency_phase3', type=int, default=10000) # 11 for test # original: 10000) #원래는 200이었음
    parser.add_argument('--nEpochs_phase3', type=int, default=20)
    parser.add_argument('--augment_prob_phase3', default=0)
    parser.add_argument('--optim_phase3', default='Adam')
    parser.add_argument('--weight_decay_phase3', default=1e-2)
    parser.add_argument('--lr_policy_phase3', default='step', choices=['step','SGDR'], help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase3', type=float, default=1e-4)
    parser.add_argument('--lr_gamma_phase3', type=float, default=0.9)
    parser.add_argument('--lr_step_phase3', type=int, default=1500)
    parser.add_argument('--lr_warmup_phase3', type=int, default=100)
    parser.add_argument('--sequence_length_phase3', default=20)
    parser.add_argument('--workers_phase3', default=4)
    parser.add_argument('--model_weights_path_phase2', default=None)
    
    ##phase 4 (test)
    parser.add_argument('--model_weights_path_phase3', default=None)
    
    
    args = parser.parse_args()
    return args

def setup_folders(): #cuda_num):
    #cuda_num = str(cuda_num)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    base_path = os.getcwd() # 스크립트를 돌린 위치가 base_path가 됨 - 이 python file이 있는 곳 (지금은 TFF 디렉토리)
    os.makedirs(os.path.join(base_path,'experiments'),exist_ok=True) #여기서^^ 여기서 4개씩 막 만들어지는구나?
    os.makedirs(os.path.join(base_path,'runs'),exist_ok=True)
    os.makedirs(os.path.join(base_path, 'splits'), exist_ok=True)
    return base_path

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
    # run해서 model_weights_path를 뽑아내는 형태. 근데 왜 BEST_val_loss.pth 같은 게 안 나오지..? 아.. 이거 다 끝나야 저장되는거구나..
    # epoch 9의 BEST val.pth를 뽑아다가 써야 함.

def test(args,model_weights_path):
    experiment_folder = '{}_{}_{}'.format(args.dataset_name, 'test_{}'.format(args.fine_tune_task), datestamp())
    experiment_folder = os.path.join(args.base_path,'tests', experiment_folder)
    os.makedirs(experiment_folder,exist_ok=True)
    trainer = Trainer(experiment_folder, '3', args, ['test'], model_weights_path)
    trainer.testing()

def _get_sync_file():    
        """Logic for naming sync file using slurm env variables"""
        sync_file_dir = '%s/pytorch-sync-files' % os.environ['SCRATCH']
        os.makedirs(sync_file_dir, exist_ok=True)

        #temporally add two lines below for torch.distributed.launcher
        try:
            sync_file = 'file://%s/pytorch_sync.%s.%s' % (
            sync_file_dir, os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID'])
        except KeyError as k:
            sync_file = 'file://%s/pytorch_sync.%s.%s' % (
            sync_file_dir, '12345', '12345')
        return sync_file

def weight_loader(args):
    model_weights_path = None
    try:
        if args.step == '1' :
            task = 'autoencoder_reconstruction'          
        elif args.step == '2':
            task = 'tranformer_reconstruction'
            if os.path.exists(args.model_weights_path_phase1):
                model_weights_path = args.model_weights_path_phase1 
        elif args.step == '3':
            task = 'fine_tune_{}'.format(args.fine_tune_task)
            if os.path.exists(args.model_weights_path_phase2):
                model_weights_path = args.model_weights_path_phase2
        elif args.step == 'test':
            task = None
            if os.path.exists(args.model_weights_path_phase3):
                model_weights_path = args.model_weights_path_phase3
    except:
            #if no weights were provided
            model_weights_path = None 

    
    # print(f'loading weight from {model_weights_path}')
    return model_weights_path, args.step, task
    
    

    
def main(base_path):
    args = get_arguments(base_path)
    
    ### DDP         
    # sbatch script에서 WORLD_SIZE를 지정해준 경우 (노드 당 gpu * 노드의 수)
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    # 혹은 슬럼에서 자동으로 ntasks per node * nodes 로 구해줌
    elif 'SLURM_NTASKS' in os.environ:
        args.world_size = int(os.environ['SLURM_NTASKS'])
        
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()
    
    if args.distributed:
        #args.local_rank = int(os.environ['LOCAL_RANK']) #stella added this line
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        sync_file = _get_sync_file()
        dist.init_process_group(backend=args.dist_backend, init_method=sync_file,
                            world_size=args.world_size, rank=args.rank)
    else:
        args.rank = 0
        args.gpu = 0

    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    # load weights that you specified at the Argument
    model_weights_path, step, task = weight_loader(args)
    
    if step == 'test' :
        print(f'starting testing')
        test(args, model_weights_path) # have some problems here
    else:
        print(f'starting phase{step}: {task}')
        run_phase(args,model_weights_path,step,task)
        print(f'finishing phase{step}: {task}')
        
if __name__ == '__main__':
    base_path = setup_folders() #cuda_num=0) #현재는 gpu 0번만 쓰는 상태 -> i see - > gpu 0, 1, 2, 3으로 바꿨음.
    main(base_path)

