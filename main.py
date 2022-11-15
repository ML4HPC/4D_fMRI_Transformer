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

import optuna 
from copy import deepcopy
import dill
import logging
import sys

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
    experiment_folder = '{}_{}_{}_{}'.format(args.dataset_name,phase_name,args.target,args.exp_name)
    experiment_folder = Path(os.path.join(args.base_path,'experiments',experiment_folder))
    os.makedirs(experiment_folder, exist_ok=True)
    setattr(args,'loaded_model_weights_path_phase' + phase_num,loaded_model_weights_path)
    args.experiment_folder = experiment_folder
    args.experiment_title = experiment_folder.name
    
    fine_tune_task = args.fine_tune_task
    print(f'saving the results at {args.experiment_folder}')
    
    # save hyperparameters
    args_logger(args)

    # make args to dict. + detach phase numbers from args
    kwargs = sort_args(phase_num, vars(args))

    S = ['train','val']

    if kwargs.get('use_optuna') == True:
        # referred to these links
        # https://python-bloggers.com/2022/08/hyperparameter-tuning-a-transformer-with-optuna/

        LR_MIN = 1e-5
        LR_CEIL = 1e-2
        WD_MIN = 4e-5
        WD_CEIL = 0.01
        
        #TF_HL = [4,8,16]
        #TF_AH = [4,8,16]
        #SL = [8,16,20,32]
        Validation_Frequency = 69
        # HCP : int(44700(# of samples) / (int(kwargs.get('batch_size')) * args.world_size)) 
        # 69 for batch size 16 and world size 40
        # same as iteration
        NUM_EPOCHS = 3 # each trial undergo 3 epochs
        is_classification = kwargs.get('fine_tune_task') == 'binary_classification'

        def objective(single_trial: optuna.Trial): 
            # https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_simple.py
            trial = optuna.integration.pytorch_distributed.TorchDistributedTrial(single_trial, device=torch.device(args.gpu))


            trial_kwargs = deepcopy(kwargs)
            # validate the performance per 500 iteration
            trial_kwargs['optim'] = 'Adam'
            trial_kwargs['validation_frequency'] = Validation_Frequency 
            trial_kwargs['lr_init'] = trial.suggest_loguniform('lr_init', low=LR_MIN, high=LR_CEIL)
            trial_kwargs['weight_decay'] = trial.suggest_loguniform('weight_decay', low=WD_MIN, high=WD_CEIL)
            #trial_kwargs['transformer_hidden_layers'] = trial.suggest_categorical('transformer_hidden_layers', choices= TF_HL)
            #trial_kwargs['transformer_num_attention_heads'] = trial.suggest_categorical('transformer_num_attention_heads', choices=TF_AH)
            #trial_kwargs['sequence_length'] = trial.suggest_categorical('sequence_length', choices=SL)
            trial_kwargs['nEpochs'] = NUM_EPOCHS
            trial_kwargs['trial'] = trial

            
            trainer = Trainer(sets=S,**trial_kwargs)

            # classification
            best_val_AUROC, best_val_loss = trainer.training()

            return best_val_AUROC if is_classification else best_val_loss


        #----------------------------------------------------------------------------------------------------
        #                    CREATE OPTUNA STUDY
        #----------------------------------------------------------------------------------------------------

        print('Triggering Optuna study')
        NUM_TRIALS = args.num_trials
        study_name = args.exp_name
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        if args.rank == 0:
            print('study_name:',study_name)
            storage=optuna.storages.RDBStorage(
            url="sqlite:///{}.db".format(study_name),
            engine_kwargs={ "connect_args": {"timeout": 10}},
            skip_compatibility_check=True
            )
            study = optuna.create_study(study_name=study_name, sampler=optuna.samplers.RandomSampler(), pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=5, interval_steps=1) ,storage=storage, load_if_exists=True, direction='maximize' if is_classification else 'minimize') 
            study.optimize(objective, n_trials=NUM_TRIALS)  
        else:
            for _ in range(NUM_TRIALS):
                try:
                    objective(None)
                except optuna.TrialPruned:
                    pass
        
        # with DDP, each process (ranks) undergo 'NUM_TRIALS' trails
        # so, total NUM_TRIALS * world_size would be run (20 * 40 = 800)

        if args.rank == 0:
            assert study is not None
            pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            print("Study statistics: ")
            print("Number of finished trials: ", len(study.trials))
            print("Number of pruned trials: ", len(pruned_trials))
            print("Number of complete trials: ", len(complete_trials))

            print('Finding study best parameters')
            print("Best trial:")
            trial = study.best_trial
            print("  Value: ", trial.value)
            
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
                print('replace hyperparameter with best hyperparameters')
                kwargs[key] = value

            #kwargs to pkl
            with open(os.path.join(args.experiment_folder,'best_arguments.pkl'),'wb') as f:
                dill.dump(kwargs,f)
            
            #kwargs to txt
            with open(os.path.join(args.experiment_folder,'best_argument_documentation.txt'),'w+') as f:
                for name,arg in kwargs.items():
                    f.write('{}: {}\n'.format(name,arg))

    else:
        trainer = Trainer(sets=S,**kwargs)
        trainer.training()

        # sort the pth files at the target directory. (the latest pth file comes first.)
        pths = sort_pth_files(self.experiment_folder)

        if len(pths) > 0 : 
            return pths[0][0] # the most recent checkpoints (= the best model checkpoints)
        else:
            return None


def test(args,phase_num,model_weights_path):
    experiment_folder = '{}_{}_{}'.format(args.dataset_name, 'test_{}'.format(args.fine_tune_task), args.exp_name) 
    experiment_folder = Path(os.path.join(args.base_path,'tests', experiment_folder))
    os.makedirs(experiment_folder,exist_ok=True)
    setattr(args,'loaded_model_weights_path_phase' + phase_num, model_weights_path) 
    
    args.experiment_folder = experiment_folder
    args.experiment_title = experiment_folder.name

    fine_tune_task = args.fine_tune_task
    # save hyperparameters
    args_logger(args)
    # make args to dict. + detach phase numbers from args
    kwargs = sort_args(args.step, vars(args))
    S = ['test']
    trainer = Trainer(sets=S,**kwargs)
    trainer.testing()


if __name__ == '__main__':
    base_path = os.getcwd() 
    print(base_path)
    setup_folders(base_path) 
    args = get_arguments(base_path)

    # DDP initialization
    init_distributed(args)

    # load weights that you specified at the Argument
    model_weights_path, step, task = weight_loader(args)

    if step == '4' : # test_only
        print(f'starting testing')
        test(args, step, model_weights_path) 
        print(f'finishing testing')
    else:
        print(f'starting phase{step}: {task}')
        model_weights_path = run_phase(args,model_weights_path,step,task)
        print(f'finishing phase{step}: {task}')

        # after finishing step3(classification/regression), run test.
        if step == '3':
            print(f'starting testing')
            test(args, '4', model_weights_path) 
            print(f'finishing testing')
