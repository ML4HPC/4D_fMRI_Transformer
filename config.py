from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from modules.data_preprocess_and_load.data_module3 import fMRIDataModule

def get_arguments(base_path):
    """
    handle arguments from commandline.
    some other hyper parameters can only be changed manually (such as model architecture,dropout,etc)
    notice some arguments are global and take effect for the entire three phase training process, while others are determined per phase
    """
    parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_name', type=str,default="baseline") 
    parser.add_argument('--base_path', default=base_path)
    parser.add_argument('--step', default='1', choices=['1','2','3','4'], help='which step you want to run')
    
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--log_dir', type=str, default=os.path.join(base_path, 'runs'))
    
    
    # parser.add_argument('--random_TR', action='store_false') #True면(인자를 넣어주지 않으면) 전체 sequence 로부터 random sampling(default). False면 (--random_TR 인자를 넣어주면) 0번째 TR부터 sliding window
    
    # loss-related
    parser.add_argument('--intensity_factor', default=1)
    parser.add_argument('--perceptual_factor', default=1)
    parser.add_argument('--which_perceptual', default='vgg', choices=['vgg','densenet3d'])
    parser.add_argument('--reconstruction_factor', default=1)
    
    # model related
    parser.add_argument('--transformer_hidden_layers', type=int,default=16)
    parser.add_argument('--transformer_num_attention_heads',type=int, default=16)
    parser.add_argument('--transformer_emb_size',type=int ,default=2640)
    parser.add_argument('--running_mean_size', default=5000)
    
    # DDP configs:
    parser.add_argument('--world_size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--init_method', default='file', type=str, choices=['file','env'], help='DDP init method')
    

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
    parser.add_argument('--optim_phase1', default='AdamW')
    parser.add_argument('--weight_decay_phase1', default=1e-7)
    parser.add_argument('--lr_policy_phase1', default='step', choices=['step','SGDR'], help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase1', type=float, default=1e-3)
    parser.add_argument('--lr_gamma_phase1', type=float, default=0.97)
    parser.add_argument('--lr_step_phase1', type=int, default=500)
    parser.add_argument('--lr_warmup_phase1', type=int, default=500)

    ##phase 2
    parser.add_argument('--task_phase2', type=str, default='transformer_reconstruction')
    parser.add_argument('--batch_size_phase2', type=int, default=4) #원래는 1이었음
    parser.add_argument('--validation_frequency_phase2', type=int, default=10000000) # 11 for test original: 10000) #원래는 500이었음
    parser.add_argument('--optim_phase2', default='Adam')
    parser.add_argument('--nEpochs_phase2', type=int, default=20)
    parser.add_argument('--augment_prob_phase2', default=0)
    parser.add_argument('--weight_decay_phase2', default=1e-7)
    parser.add_argument('--lr_policy_phase2', default='step', choices=['step','SGDR'], help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase2', type=float, default=1e-4)
    parser.add_argument('--lr_gamma_phase2', type=float, default=0.97)
    parser.add_argument('--lr_step_phase2', type=int, default=1000)
    parser.add_argument('--lr_warmup_phase2', type=int, default=500)
    parser.add_argument('--model_weights_path_phase1', default=None)
    parser.add_argument('--use_cont_loss', default=False)
    parser.add_argument('--use_mask_loss', default=False)

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
    parser.add_argument('--model_weights_path_phase2', default=None)
    
    ##phase 4 (test)
    parser.add_argument('--task_phase4', type=str, default='test')
    parser.add_argument('--model_weights_path_phase3', default=None)
    parser.add_argument('--batch_size_phase4', type=int, default=4)
    parser.add_argument('--nEpochs_phase4', type=int, default=20)
    parser.add_argument('--augment_prob_phase4', default=0)
    parser.add_argument('--optim_phase4', default='Adam')
    parser.add_argument('--weight_decay_phase4', default=1e-2)
    parser.add_argument('--lr_policy_phase4', default='step', choices=['step','SGDR'], help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase4', type=float, default=1e-4)
    parser.add_argument('--lr_gamma_phase4', type=float, default=0.9)
    parser.add_argument('--lr_step_phase4', type=int, default=1500)
    parser.add_argument('--lr_warmup_phase4', type=int, default=100)
    
    temp_args, _ = parser.parse_known_args()
    
    # Set dataset-specific Arguments
    Dataset = fMRIDataModule
    
    parser = Dataset.add_data_specific_args(parser)
    args = parser.parse_args()
    return args