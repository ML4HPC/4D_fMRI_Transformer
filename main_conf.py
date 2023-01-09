from modules.utils import *  #including 'init_distributed', 'weight_loader'
from modules.trainer_conf import Trainer #####
import os
from pathlib import Path
import torch

from config_conf import get_arguments #####

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

#import wandb

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
    
    print(f'saving the results at {args.experiment_folder}')
    
    # save hyperparameters
    args_logger(args)

    # make args to dict. + detach phase numbers from args
    kwargs = sort_args(phase_num, vars(args))

    #wandb
    #wandb.init(project=args.exp_name)
    #wandb.config = kwargs

    S = ['train','val']

    if kwargs.get('use_optuna') == True:
        # referred to these links
        # https://python-bloggers.com/2022/08/hyperparameter-tuning-a-transformer-with-optuna/
        if kwargs.get('hyp_lr_init'):
            LR_MIN = kwargs.get('hyp_lr_init_min') #1e-6
            LR_CEIL = kwargs.get('hyp_lr_init_ceil') #1e-3
        if kwargs.get('hyp_weight_decay'):
            WD_MIN = kwargs.get('hyp_weight_decay_min') #1e-5
            WD_CEIL = kwargs.get('hyp_weight_decay_ceil') #1e-2
        
        if kwargs.get('hyp_transformer_hidden_layers'):
            TF_HL_small = kwargs.get('hyp_transformer_hidden_layers_range_small') #8
            TF_HL_big = kwargs.get('hyp_transformer_hidden_layers_range_big') #16
            TF_HL = [TF_HL_small, TF_HL_big]
        
        if kwargs.get('hyp_transformer_num_attention_heads'):
            TF_AH_small = kwargs.get('hyp_transformer_num_attention_heads_range_small') #8
            TF_AH_big = kwargs.get('hyp_transformer_num_attention_heads_range_big') #16
            TF_AH = [TF_AH_small, TF_AH_big]

        if kwargs.get('hyp_seq_len'):    
            SL_small = kwargs.get('hyp_seq_len_range_small') #10
            SL_big = kwargs.get('hyp_seq_len_range_big') #20
            SL = [SL_small, SL_big]

        if kwargs.get('hyp_dropout'):    
            DO_small = kwargs.get('hyp_dropout_range_small') #0.1
            DO_big = kwargs.get('hyp_dropout_range_big') #0.8
            DO = [DO_small, DO_big]
        #Validation_Frequency = 69
        # 69 for batch size 16 and world size 40
        # same as iteration
        NUM_EPOCHS = kwargs.get('opt_num_epochs') # each trial undergo 'opt_num_epochs' epochs
        is_classification = kwargs.get('fine_tune_task') == 'binary_classification'

        def objective(single_trial: optuna.Trial): 
            # https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_simple.py
            device = torch.device(int(os.environ["LOCAL_RANK"]))
            trial = optuna.integration.pytorch_distributed.TorchDistributedTrial(single_trial, device=device)

            # The code below should be changed for hyperparameter tuning
            trial_kwargs = deepcopy(kwargs)
            trial_kwargs['lr_step'] = 500
            if kwargs.get('hyp_batch_size'):
                trial_kwargs['batch_size'] = trial.suggest_int("batch_size",low=4, high=16, step=4)
            if kwargs.get('hyp_lr_init'):
                trial_kwargs['lr_init'] = trial.suggest_float("lr_init",low=LR_MIN, high=LR_CEIL, log=True)
            if kwargs.get('hyp_lr_gamma'):
                trial_kwargs['lr_gamma'] = trial.suggest_float("lr_gamma",low=0.1, high=0.9)
            if kwargs.get('hyp_weight_decay'):
                trial_kwargs['weight_decay'] = trial.suggest_float('weight_decay', low=WD_MIN, high=WD_CEIL, log=True)

            # model related
            if kwargs.get('hyp_transformer_hidden_layers'):
                trial_kwargs['transformer_hidden_layers'] = trial.suggest_categorical('transformer_hidden_layers', choices=TF_HL)
            if kwargs.get('hyp_transformer_num_attention_heads'):
                trial_kwargs['transformer_num_attention_heads'] = trial.suggest_categorical('transformer_num_attention_heads', choices=TF_AH)
            if kwargs.get('hyp_seq_len'):
                trial_kwargs['sequence_length'] = trial.suggest_categorical('sequence_length', choices=SL)
            if kwargs.get('hyp_dropout'):
                trial_kwargs['transformer_dropout_rate'] = trial.suggest_float('transformer_dropout_rate', low = 0.1, high=0.8, step=0.1)
            trial_kwargs['nEpochs'] = NUM_EPOCHS
            trial_kwargs['trial'] = trial
            trainer = Trainer(sets=S,**trial_kwargs)

            # classification
            best_val_AUROC, best_val_loss = trainer.training()

            return best_val_AUROC if is_classification else best_val_loss


        #----------------------------------------------------------------------------------------------------
        #                    CREATE OPTUNA STUDY
        #----------------------------------------------------------------------------------------------------

        study_name = args.exp_name
        print('NUM_TRIALS:', args.num_trials)
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        if args.rank == 0:
            print('Triggering Optuna study')
            print('study_name:',study_name)
            # storage=optuna.storages.RDBStorage(
            # url='postgresql://junbeom_admin:DBcase6974!@nerscdb03.nersc.gov/junbeom', #"sqlite:///{}.db".format(study_name),
            # skip_compatibility_check=True
            # )
            storage=optuna.storages.RDBStorage(
            url="sqlite:///{}.db".format(study_name),
            engine_kwargs={ "connect_args": {"timeout": 10}},
            skip_compatibility_check=True
            )
            # Default is TPESampler 
            study = optuna.create_study(study_name=study_name, pruner = optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=args.n_warmup_steps, interval_steps=args.interval_steps) ,storage=storage, load_if_exists=True, direction='maximize' if is_classification else 'minimize') 
            study.optimize(objective, n_trials=args.num_trials)  
        else:
            for _ in range(args.num_trials):
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
                if key == 'learning_rate':
                    kwargs['lr_init'] = value
                elif key == 'gamma':
                    kwargs['lr_gamma'] = value
                else:
                    kwargs[key] = value

            #kwargs to pkl
            with open(os.path.join(args.experiment_folder,'best_arguments.pkl'),'wb') as f:
                dill.dump(kwargs,f)
            
            #kwargs to txt
            with open(os.path.join(args.experiment_folder,'best_argument_documentation.txt'),'w+') as f:
                for name,arg in kwargs.items():
                    f.write('{}: {}\n'.format(name,arg))

    else:
        if kwargs.get('use_best_params_from_optuna') == True:  # args.use_optuna should be False 
            print('use_best_params_from_optuna')
            study_name = args.exp_name
            storage=optuna.storages.RDBStorage(
                    url="sqlite:///{}.db".format(study_name),
                    engine_kwargs={ "connect_args": {"timeout": 10}},
                    skip_compatibility_check=True
                    )
            is_classification = kwargs.get('fine_tune_task') == 'binary_classification'
            study = optuna.create_study(study_name=study_name, pruner = optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=args.n_warmup_steps, interval_steps=args.interval_steps) ,storage=storage, load_if_exists=True, direction='maximize' if is_classification else 'minimize')
            for key,value in study.best_params.items():
                if key == 'learning_rate':
                    print(f"replacing the value of learning_rate : from {kwargs['lr_init']} to {value}")
                    kwargs['lr_init'] = value
                elif key == 'gamma':
                    print(f"replacing the value of gamma : from {kwargs['lr_gamma']} to {value}")
                    kwargs['lr_gamma'] = value
                else:
                    print(f'replacing the value of {key} : from {kwargs[key]} to {value}')
                    kwargs[key] = value
                
            kwargs['lr_step'] = 500 # fix the hyperparameter for fixed periods.

        trainer = Trainer(sets=S,**kwargs)
        trainer.training()
 
        return None


        


def test(args,phase_num,model_weights_path):
    experiment_folder = '{}_{}_{}'.format(args.dataset_name, 'test_{}'.format(args.fine_tune_task), args.exp_name) 
    experiment_folder = Path(os.path.join(args.base_path, 'tests', experiment_folder))
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
        _ = run_phase(args,model_weights_path,step,task)
        print(f'finishing phase{step}: {task}')

        # sort the pth files at the target directory. (the latest pth file comes first.)
        pths = sort_pth_files(args.experiment_folder)

        if len(pths) > 0 : 
            model_weights_path = pths[0][0] # the most recent checkpoints (= the best model checkpoints)
        else:
            model_weights_path = None

        # after finishing step3(classification/regression), run test.
        if step == '3' and (not args.use_optuna):
            print(f'starting testing')
            test(args, '4', model_weights_path) 
            print(f'finishing testing')
