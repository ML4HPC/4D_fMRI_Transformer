
from .loss_writer import Writer
from .learning_rate import LrHandler
#from data_preprocess_and_load.dataloaders import DataHandler
from .data_preprocess_and_load.data_module3 import fMRIDataModule
import torch
import warnings
import numpy as np
from tqdm import tqdm
from .model import Encoder_Transformer_Decoder,Encoder_Transformer_finetune,AutoEncoder,MobileNet_v2_Transformer_finetune, MobileNet_v3_Transformer_finetune
from .losses import get_intense_voxels
import time
import pathlib
import os

#DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn import DataParallel
import builtins

#torch AMP
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

import optuna

#wandb
import wandb

# ASP
#from apex.contrib.sparsity import ASP

#from apex.optimizers import FusedAdam


class Trainer():
    """
    main class to handle training, validation and testing.
    note: the order of commands in the constructor is necessary
    """
    def __init__(self,sets,**kwargs):
        
        self.register_args(**kwargs)
        
        self.eval_iter = 0
        self.batch_index = None
        self.best_loss = 100000
        #self.best_accuracy = 0
        self.best_AUROC = 0
        self.best_ACC = 0
        self.st_epoch = 1
        self.recent_pth = None
        self.state_dict = None
        
        # self.train_loader, self.val_loader, self.test_loader = DataHandler(**kwargs).create_dataloaders()
        # torch lightening Dataloader
        dm = fMRIDataModule(**kwargs)
        dm.setup()
        dm.prepare_data()
        self.train_loader = dm.train_dataloader()
        self.val_loader = dm.val_dataloader()
        self.test_loader = dm.test_dataloader()
        
        #located here to give 'number of batches' to the LR scheduler.
        self.lr_handler = LrHandler(self.train_loader, **kwargs)
        
        self.create_model() # model on cpu
        if not self.use_optuna and not self.train_from_scratch:
            self.load_model_checkpoint()
        self.set_model_device() # set DDP or DP after loading checkpoint at CPUs

        #wandb
        if self.rank == 0:
            os.environ["WANDB_API_KEY"] = kwargs.get('wandb_key')
            os.environ["WANDB_MODE"] = kwargs.get('wandb_mode')
            wandb.init(project='4D fMRI Transformers',entity='fsb',reinit=True, name=self.experiment_title, config=kwargs)
            wandb.watch(self.model,log='all',log_freq=10)
        
        self.create_optimizer()
        self.lr_handler.set_schedule(self.optimizer)
        self.scaler = GradScaler() 
        
        if not self.use_optuna and not self.train_from_scratch:    
            self.load_optim_checkpoint()

        self.writer = Writer(sets,**kwargs)
        self.sets = sets
        
        self.nan_list = []

        for name, loss_dict in self.writer.losses.items():
            if loss_dict['is_active']:
                print('using {} loss'.format(name))
                setattr(self, name + '_loss_func', loss_dict['criterion'])
    
    def _sort_pth_files(self, files_Path):
        file_name_and_time_lst = []
        for f_name in os.listdir(files_Path):
            if f_name.endswith('.pth'):
                written_time = os.path.getctime(os.path.join(files_Path,f_name))
                file_name_and_time_lst.append((f_name, written_time))
        # 생성시간 역순으로 정렬 
        sorted_file_lst = sorted(file_name_and_time_lst, key=lambda x: x[1], reverse=True)

        return sorted_file_lst
    
    def load_model_checkpoint(self):
        pths = self._sort_pth_files(self.experiment_folder)
        if len(pths) > 0 : # if there are any checkpoints from which we can resume the training. 
            self.recent_pth = pths[0][0] # the most recent checkpoints
            print(f'loading checkpoint from {os.path.join(self.experiment_folder,self.recent_pth)}')
            self.state_dict = torch.load(os.path.join(self.experiment_folder,self.recent_pth),map_location='cpu') #, map_location=self.device
            self.model.load_partial_state_dict(self.state_dict['model_state_dict'],load_cls_embedding=False)
            self.model.loaded_model_weights_path = os.path.join(self.experiment_folder,self.recent_pth)

        elif self.loaded_model_weights_path: # if there are weights from previous phase
            self.recent_pth = None
            self.state_dict = torch.load(self.loaded_model_weights_path,map_location='cpu') #, map_location=self.device
            self.model.load_partial_state_dict(self.state_dict['model_state_dict'],load_cls_embedding=False)
            self.model.loaded_model_weights_path = self.loaded_model_weights_path
            
        else:
            self.recent_pth = None
            self.state_dict = None
            print('There are no checkpoints or weights from previous steps')
            
    def load_optim_checkpoint(self):
        if self.recent_pth and self.state_dict: # if there are any checkpoints
            self.optimizer.load_state_dict(self.state_dict['optimizer_state_dict'])
            self.lr_handler.schedule.load_state_dict(self.state_dict['schedule_state_dict'])
            self.optimizer.param_groups[0]['lr'] = self.state_dict['lr']
            self.scaler.load_state_dict(self.state_dict['amp_state'])
            self.st_epoch = int(self.state_dict['epoch']) + 1
            self.best_loss = self.state_dict['loss_value']
            text = 'Training start from epoch {} and learning rate {}.'.format(self.st_epoch, self.optimizer.param_groups[0]['lr'])
            if 'ACC' in self.state_dict:
                text += 'validation ACC - {}'.format(self.state_dict['ACC'])
            print('Training start from epoch {} and learning rate {}.'.format(self.st_epoch, self.optimizer.param_groups[0]['lr']))
            
        elif self.state_dict:  # if there are weights from previous phase
            text = 'loaded model weights:\nmodel location - {}\nlast learning rate - {}\nvalidation loss - {}\n'.format(
                self.loaded_model_weights_path, self.state_dict['lr'],self.state_dict['loss_value'])
            if 'ACC' in self.state_dict:
                text += 'validation ACC - {}'.format(self.state_dict['ACC'])
            print(text)
        else:
            pass
            
            
    def create_optimizer(self):
        lr = self.lr_handler.base_lr
        params = self.model.parameters()
        weight_decay = self.kwargs.get('weight_decay')
        #self.optimizer = FusedAdam(params, lr=lr, weight_decay=weight_decay)
        optim = self.kwargs.get('optim')
        self.optimizer = getattr(torch.optim,optim)(params, lr=lr, weight_decay=weight_decay)  #torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        
        # attach optimizer to cuda device.
        # for state in self.optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.cuda(self.gpu)

    def create_model(self):
        dim = next(iter(self.train_loader))["fmri_sequence"].shape[2:5] # channel, w, h, d
        #print('task is:', self.task.lower()) transformer_reconstruction
        if self.task.lower() in ['fine_tune', 'test']:
            if self.block_type == 'MobileNet_v2':
                self.model = MobileNet_v2_Transformer_finetune(dim,**self.kwargs)
            elif self.block_type == 'MobileNet_v3':
                self.model = MobileNet_v3_Transformer_finetune(dim,**self.kwargs)
            elif self.block_type == 'green':
                self.model = Encoder_Transformer_finetune(dim,**self.kwargs)
        elif self.task.lower() == 'autoencoder_reconstruction':
            self.model = AutoEncoder(dim,**self.kwargs)
        elif self.task.lower() == 'transformer_reconstruction':
            self.model = Encoder_Transformer_Decoder(dim,**self.kwargs)
        
    def set_model_device(self):
        if self.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if self.gpu is not None:
                #print('id of gpu is:', self.gpu)
                self.device = torch.device('cuda:{}'.format(self.gpu))
                torch.cuda.set_device(self.gpu)
                self.model.cuda(self.gpu)
                if self.task.lower() == 'autoencoder_reconstruction': # having unused parameter for 
                    self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu], broadcast_buffers=False) 
                else: # having unused parameter (classifier token)
                    self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu], broadcast_buffers=False, find_unused_parameters=True) 
                net_without_ddp = self.model.module
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.cuda()
                if self.task.lower() == 'autoencoder_reconstruction':
                    self.model = torch.nn.parallel.DistributedDataParallel(self.model) 
                else: # having unused parameter (classifier token)
                    self.model = torch.nn.parallel.DistributedDataParallel(self.model,find_unused_parameters=True) 
                net_without_ddp = self.model.module
            
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = DataParallel(self.model).to(self.device) if torch.cuda.is_available() else self.model

    def training(self):
         
        if self.profiling == True:
            self.nEpochs = 1
        for epoch in range(self.st_epoch,self.nEpochs + 1): 
            start = time.time()
            self.train_epoch(epoch)
            self.eval_epoch('val')

            # aggregate and print losses and metrics
            print('______epoch summary {}/{}_____\n'.format(epoch,self.nEpochs))
            self.writer.loss_summary(lr=self.optimizer.param_groups[0]['lr'])
            self.writer.accuracy_summary(mid_epoch=True)
            self.writer.save_history_to_csv()

            #wandb
            if self.rank == 0:
                self.writer.register_wandb(epoch, lr=self.optimizer.param_groups[0]['lr'])

            if self.use_optuna:
                val_ACC = self.get_last_ACC()
                if val_ACC > self.best_ACC:
                    self.best_ACC = val_ACC
                self.trial.report(val_ACC, step=epoch-1)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            else:
                self.save_checkpoint_(epoch, len(self.train_loader), self.scaler) 
                    
            # else:
            #     dist.barrier()
            end = time.time()
            
            print(f'time taken to perform {epoch}: {end-start:.2f}')
        
        return self.best_AUROC, self.best_loss #validation AUROC
            
 
    def train_epoch(self,epoch):       
        if self.distributed:
            self.train_loader.sampler.set_epoch(epoch)
        self.train()

        times = []
        for batch_idx, input_dict in enumerate(tqdm(self.train_loader,position=0,leave=True)): 
            ### training ###
            #start_time = time.time()
            torch.cuda.nvtx.range_push("training steps")
            self.writer.total_train_steps += 1
            self.optimizer.zero_grad()
            if self.amp:
                torch.cuda.nvtx.range_push("forward pass")
                with autocast():
                    loss_dict, loss = self.forward_pass(input_dict)
                torch.cuda.nvtx.range_pop()
                loss = loss / self.accumulation_steps # gradient accumulation
                torch.cuda.nvtx.range_push("backward pass")
                self.scaler.scale(loss).backward()
                torch.cuda.nvtx.range_pop()
                
                if  (batch_idx + 1) % self.accumulation_steps == 0: # gradient accumulation
                    # gradient clipping 
                    if self.gradient_clipping:
                        torch.cuda.nvtx.range_push("unscale")
                        self.scaler.unscale_(self.optimizer)
                        torch.cuda.nvtx.range_pop()
                        torch.cuda.nvtx.range_push("gradient_clipping")
                        print('executing gradient clipping')
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, error_if_nonfinite=False)
                        torch.cuda.nvtx.range_pop()
                    
                    torch.cuda.nvtx.range_push("optimize")
                    self.scaler.step(self.optimizer)
                    torch.cuda.nvtx.range_pop()
                    scale = self.scaler.get_scale()
                    self.scaler.update()
                    skip_lr_sched = (scale > self.scaler.get_scale())
                if not skip_lr_sched:
                    self.lr_handler.schedule_check_and_update(self.optimizer) 
            else:
                torch.cuda.nvtx.range_push("forward pass")
                loss_dict, loss = self.forward_pass(input_dict)
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("backward pass")
                loss.backward()
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("optimize")
                self.optimizer.step()
                torch.cuda.nvtx.range_pop()
                self.lr_handler.schedule_check_and_update(self.optimizer)

            # register train_losses to loss_writer  
            self.writer.write_losses(loss_dict, set='train')
            torch.cuda.nvtx.range_pop()
            
            #for profiling, early stopping
            if self.profiling == True:
                if batch_idx == 10 : 
                    break
            '''
            use batch-validation only for Optuna tuning 
            if self.use_optuna:
                if (self.eval_iter + 1) % self.validation_frequency == 0:
                    step = ((batch_idx + 1) // self.validation_frequency)-1 # start from 0
                    print(f'optuna: evaluating at epoch {epoch} batch {batch_idx}')
                    if (not self.distributed) or self.rank == 0 :
                        ### validation ##
                        self.eval_epoch('val')
                        self.writer.loss_summary(lr=self.optimizer.param_groups[0]['lr'])
                        self.writer.accuracy_summary(mid_epoch=True)
                        self.writer.experiment_title = self.writer.experiment_title
                        self.writer.save_history_to_csv()
                        if not self.use_optuna:
                            self.save_checkpoint_(epoch, batch_idx, self.scaler) # validation마다 checkpoint 저장               
                        # val_loss = self.get_last_loss()
                        val_AUROC = self.get_last_AUROC()

                        if val_AUROC > self.best_AUROC:
                            self.best_AUROC = val_AUROC
                        
                        # report current performances
                        self.trial.report(val_AUROC, step=step)

                        if self.trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
            
                        self.train()
                    # else:
                    #     dist.barrier()
            self.eval_iter += 1
            '''
                
    def eval_epoch(self,set):
        loader = self.val_loader if set == 'val' else self.test_loader
        self.eval(set)
        with torch.no_grad():
            for batch_idx, input_dict in enumerate(tqdm(loader, position=0, leave=True)):
                with autocast():
                    loss_dict, _ = self.forward_pass(input_dict)
                    
                # register val/test losses to loss writer
                self.writer.write_losses(loss_dict, set=set)
                if self.profiling == True:
                    if batch_idx == 10 : 
                        break
        
    def forward_pass(self,input_dict):
        '''
        shape of input dict is : torch.Size([batch, ch, w, h, d, t])
        '''
        input_dict = {k:(v.to(self.gpu) if (self.cuda and torch.is_tensor(v)) else v) for k,v in input_dict.items()}
        if self.with_voxel_norm:
            with torch.no_grad():
                mean = input_dict['fmri_sequence'][:,:,:,:,:,-2:-1]
                std = input_dict['fmri_sequence'][:,:,:,:,:,-1:]
                fmri = input_dict['fmri_sequence'][:,:,:,:,:,:-2]

                background_value = torch.min(fmri.flatten(start_dim=1), dim=1, keepdim=False)[0]
                background_mask = fmri == background_value[:, None, None, None, None, None]
                vnorm_fmri = torch.zeros_like(fmri,device=self.gpu)

                vnorm_fmri[~background_mask] = ((fmri - mean) / (std + 1e-8))[~background_mask]
                vnorm_fmri.add_(background_mask * background_value[:, None, None, None, None, None]) # inplace operation
                
                input_dict['fmri_sequence'] = torch.cat([fmri, vnorm_fmri], dim=1)
        output_dict = self.model(input_dict['fmri_sequence']) 
        torch.cuda.nvtx.range_push("aggregate_losses")
        loss_dict, loss = self.aggregate_losses(input_dict, output_dict)
        torch.cuda.nvtx.range_pop()
        if self.task in ['fine_tune', 'test']:
            self.compute_accuracy(input_dict, output_dict)
        return loss_dict, loss


    def aggregate_losses(self,input_dict,output_dict):
        final_loss_dict = {}
        final_loss_value = 0
        for loss_name, current_loss_dict in self.writer.losses.items():
            if current_loss_dict['is_active']:
                loss_func = getattr(self, 'compute_' + loss_name)
                torch.cuda.nvtx.range_push(f"{loss_name}")
                current_loss_value = loss_func(input_dict,output_dict)
                torch.cuda.nvtx.range_pop()
                """
                if current_loss_value.isnan().sum() > 0:
                    warnings.warn('found nans in computation')
                    print('at {} loss'.format(loss_name))
                    self.nan_list+=np.array(input_dict['subject_name'])[(output_dict['reconstructed_fmri_sequence'].reshape(output_dict['reconstructed_fmri_sequence'].shape[0],-1).isnan().sum(axis=1).detach().cpu().numpy() > 0)].tolist()
                    print('current_nan_list:',set(self.nan_list))
                """
                lamda = current_loss_dict['factor']
                factored_loss = current_loss_value * lamda
                final_loss_dict[loss_name] = factored_loss.item()
                final_loss_value += factored_loss
        
        # make total loss -> it is registered into loss writer
        final_loss_dict['total'] = final_loss_value.item()
        return final_loss_dict, final_loss_value
        
    def testing(self):
        if (not self.distributed) or self.rank == 0 :
            self.eval_epoch('test')
            self.writer.loss_summary(lr=0)
            self.writer.accuracy_summary(mid_epoch=False)
            for metric_name in dir(self.writer):
                if ('history' not in metric_name) or ( metric_name == 'save_history_to_csv') :
                    continue
                print(metric_name)
                metric_score = getattr(self.writer,metric_name)[-1]
                print('final test score - {} = {}'.format(metric_name,metric_score))
    
    def train(self):
        self.mode = 'train'
        self.model = self.model.train()
        
    def eval(self,set):
        self.mode = set
        self.model = self.model.eval()

    def get_last_loss(self):
        if self.kwargs.get('fine_tune_task') == 'regression': #self.model.task
            return self.writer.val_MAE[-1]
        else:
            return self.writer.total_val_loss_history[-1]

    def get_last_AUROC(self):
        if hasattr(self.writer,'val_AUROC'):
            return self.writer.val_AUROC[-1]
        else:
            return None

    def get_last_ACC(self):
        if hasattr(self.writer,'val_Balanced_Accuracy'):
            return self.writer.val_Balanced_Accuracy[-1]
        else:
            return None

    def save_checkpoint_(self, epoch, batch_idx, scaler):

        loss = self.get_last_loss()
        #accuracy = self.get_last_AUROC()
        ACC = self.get_last_ACC()
        AUROC = self.get_last_AUROC()

        title = str(self.writer.experiment_title) + '_epoch_' + str(int(epoch)) + '_batch_index_'+ str(batch_idx)
        directory = self.writer.experiment_folder

        # Create directory to save to
        if not os.path.exists(directory):
            os.makedirs(directory)
        if self.amp:
            amp_state = scaler.state_dict()

        # Build checkpoint dict to save.
        ckpt_dict = {
            'model_state_dict':self.model.module.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict() if self.optimizer is not None else None,
            'epoch':epoch,
            'loss_value':loss,
            'amp_state': amp_state}
        
        # if AUROC is not None:
        if ACC is not None:
            ckpt_dict['ACC'] = ACC
        if self.lr_handler.schedule is not None:
            ckpt_dict['schedule_state_dict'] = self.lr_handler.schedule.state_dict()
            ckpt_dict['lr'] = self.optimizer.param_groups[0]['lr']
            print(f"current_lr:{self.optimizer.param_groups[0]['lr']}")
        if hasattr(self,'loaded_model_weights_path'):
            ckpt_dict['loaded_model_weights_path'] = self.loaded_model_weights_path
        
        # Save checkpoint per one epoch 
        # commented out by JB
        # name = "{}.pth".format(core_name) 
        # torch.save(ckpt_dict, os.path.join(directory, name))
        
        core_name = title
        # best loss나 best accuracy를 가진 모델만 저장하는 코드
        # classification
        #if accuracy is not None and self.best_accuracy < accuracy:
        if ACC is not None and self.best_ACC < ACC:
            #self.best_accuracy = accuracy
            self.best_ACC = ACC
            #name = "{}_BEST_val_accuracy.pth".format(core_name)
            name = "{}_BEST_val_ACC.pth".format(core_name)
            torch.save(ckpt_dict, os.path.join(directory, name))
            print(f'updating best saved model with ACC:{ACC}')
            #print(f'updating best saved model with accuracy:{accuracy}')

        # regression
        elif ACC is None and self.best_loss > loss:
            self.best_loss = loss
            name = "{}_BEST_val_loss.pth".format(core_name)
            torch.save(ckpt_dict, os.path.join(directory, name))
            print(f'updating best saved model with loss: {loss}')


    def compute_reconstruction(self,input_dict,output_dict):
        fmri_sequence = input_dict['fmri_sequence'][:,0].unsqueeze(1)
        reconstruction_loss = self.reconstruction_loss_func(output_dict['reconstructed_fmri_sequence'],fmri_sequence)
        return reconstruction_loss

    def compute_intensity(self,input_dict,output_dict):
        per_voxel = input_dict['fmri_sequence'][:,1,:,:,:,:]
        torch.cuda.nvtx.range_push("get_intensity_voxels")
        voxels = get_intense_voxels(per_voxel, output_dict['reconstructed_fmri_sequence'].shape, self.gpu) 
        torch.cuda.nvtx.range_pop()
        output_intense = output_dict['reconstructed_fmri_sequence'] * voxels  #[voxels]
        truth_intense = input_dict['fmri_sequence'][:,0] * voxels.squeeze(1) #[voxels.squeeze(1)]
        torch.cuda.nvtx.range_push("self.intensity_loss_func")
        intensity_loss = self.intensity_loss_func(output_intense.squeeze(), truth_intense.squeeze())
        torch.cuda.nvtx.range_pop()
        return intensity_loss

    def compute_perceptual(self,input_dict,output_dict):
        fmri_sequence = input_dict['fmri_sequence'][:,0].unsqueeze(1)
        perceptual_loss = self.perceptual_loss_func(output_dict['reconstructed_fmri_sequence'],fmri_sequence)
        return perceptual_loss
    
    def compute_contrastive(self,input_dict,output_dict):
        # fmri_sequence = input_dict['fmri_sequence'][:,0].unsqueeze(1)
        # print('shape of fmri_sequence is:', fmri_sequence.shape) [batch, channel, width, height, depth, T] [2, 1, 75, 93, 81, 20]
        contrastive_loss = self.contrastive_loss_func(output_dict['transformer_output_sequence'])
        return contrastive_loss
    
    def compute_mask(self, input_dict, output_dict):
        '''
        shape of output of fmri_sequence is: torch.Size([4, 20, 2640])
        '''
        mask_loss = self.mask_loss_func(output_dict['transformer_input_sequence'], output_dict['mask_list'], output_dict['transformer_output_sequence_for_mask_learning'])
        return mask_loss

    def compute_binary_classification(self,input_dict,output_dict):
        binary_loss = self.binary_classification_loss_func(output_dict['binary_classification'].squeeze(), input_dict[self.target].squeeze().float())
        #self.binary_classification_loss_func(output_dict['binary_classification'].squeeze(), input_dict['subject_binary_classification'].squeeze())
        return binary_loss

    def compute_regression(self,input_dict,output_dict):
        gender_loss = self.regression_loss_func(output_dict['regression'].squeeze(),input_dict[self.target].squeeze()) #self.regression_loss_func(output_dict['regression'].squeeze(),input_dict['subject_regression'].squeeze())
        return gender_loss

    def compute_accuracy(self,input_dict,output_dict):
        task = self.kwargs.get('fine_tune_task') #self.model.task
        out = output_dict[task].detach().clone().cpu()
        score = out.squeeze() if out.shape[0] > 1 else out
        labels = input_dict[self.target].clone().cpu() # input_dict['subject_' + task].clone().cpu()
        subjects = input_dict['subject'] #.clone().cpu()
        for i, subj in enumerate(subjects): # subjects : subjects in the batch
            subject = str(subj) #.item())
            if subject not in self.writer.subject_accuracy:
                self.writer.subject_accuracy[subject] = {'score': score[i].unsqueeze(0), 'mode': self.mode, 'truth': labels[i],'count': 1}
            else:
                # score[i].unsqueeze(0) denotes a logit for the sequence
                self.writer.subject_accuracy[subject]['score'] = torch.cat([self.writer.subject_accuracy[subject]['score'], score[i].unsqueeze(0)], dim=0)
                self.writer.subject_accuracy[subject]['count'] += 1

    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs
