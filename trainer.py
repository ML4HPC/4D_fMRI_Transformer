
from loss_writer import Writer
from learning_rate import LrHandler
from data_preprocess_and_load.dataloaders import DataHandler
import torch
import warnings
import numpy as np
from tqdm import tqdm
from model import Encoder_Transformer_Decoder,Encoder_Transformer_finetune,AutoEncoder
from losses import get_intense_voxels
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

# ASP
#from apex.contrib.sparsity import ASP

class Trainer():
    """
    main class to handle training, validation and testing.
    note: the order of commands in the constructor is necessary
    """
    def __init__(self,sets,**kwargs):
        
        self.register_args(**kwargs)
        self.lr_handler = LrHandler(**kwargs)
        
        self.train_loader, self.val_loader, _ = DataHandler(**kwargs).create_dataloaders()
        self.create_model()
        
        # load weights from the previous step
        self.load_model_from_previous_phase(load_cls_embedding=False)
        
        self.create_optimizer()
        self.lr_handler.set_schedule(self.optimizer)
        self.scaler = GradScaler() 
        
        self.eval_iter = 0
        self.batch_index = None
        self.best_loss = 100000
        self.best_accuracy = 0
        self.st_epoch = 0
        
        # Load weights from the recent pth (args.resume = True)
        if self.resume == True:
            self.checkpoint_load()
        
        #ASP.prune_trained_model(self.model, self.optimizer)

        self.writer = Writer(sets,**kwargs) #여기서 이미 writer class를 불러옴.
        self.sets = sets

        for name, loss_dict in self.writer.losses.items():
            if loss_dict['is_active']:
                print('using {} loss'.format(name))
                setattr(self, name + '_loss_func', loss_dict['criterion'])
    
    def checkpoint_load(self):
        pths = self.find_pth(self.experiment_folder)
        if len(pths) == 0:
            pass
        else:
            recent_pth = pths[0][0] # 가장 최근에 생성된 pth
            print(f'loading checkpoint from {os.path.join(self.experiment_folder,recent_pth)}')
            state_dict = torch.load(os.path.join(self.experiment_folder,recent_pth)) 
            self.model.module.load_partial_state_dict(state_dict['model_state_dict'],load_cls_embedding=False)
            self.model.module.loaded_model_weights_path = os.path.join(self.experiment_folder,recent_pth)
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.lr_handler.schedule.load_state_dict(state_dict['schedule_state_dict'])
            self.optimizer.param_groups[0]['lr'] = state_dict['lr']
            self.scaler.load_state_dict(state_dict['amp_state'])
            self.st_epoch = int(state_dict['epoch'] + 1)
            self.best_loss = state_dict['loss_value']
            print('Training start from epoch {} and learning rate {}.'.format(self.st_epoch, self.optimizer.param_groups[0]['lr']))
    
    def find_pth(self, files_Path):
        file_name_and_time_lst = []
        for f_name in os.listdir(files_Path):
            if f_name.endswith('.pth'):
                written_time = os.path.getctime(os.path.join(files_Path,f_name))
                file_name_and_time_lst.append((f_name, written_time))
        # 생성시간 역순으로 정렬 
        sorted_file_lst = sorted(file_name_and_time_lst, key=lambda x: x[1], reverse=True)

        return sorted_file_lst
    
    def load_model_from_previous_phase(self,load_cls_embedding):
        if self.loaded_model_weights_path is not None: #after autoencoder
            state_dict = torch.load(self.loaded_model_weights_path)
            self.lr_handler.set_lr(state_dict['lr'])
            self.model.module.load_partial_state_dict(state_dict['model_state_dict'],load_cls_embedding=False)
            self.model.module.loaded_model_weights_path = self.loaded_model_weights_path
            text = 'loaded model weights:\nmodel location - {}\nlast learning rate - {}\nvalidation loss - {}\n'.format(
                self.loaded_model_weights_path, state_dict['lr'],state_dict['loss_value'])
            if 'accuracy' in state_dict:
                text += 'validation accuracy - {}'.format(state_dict['accuracy'])
            print(text)    
            
    def create_optimizer(self):
        lr = self.lr_handler.base_lr
        params = self.model.parameters()
        weight_decay = self.kwargs.get('weight_decay')
        optim = self.kwargs.get('optim')
        self.optimizer = getattr(torch.optim,optim)(params, lr=lr, weight_decay=weight_decay)  #torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def create_model(self):
        dim = self.train_loader.dataset.dataset.get_input_shape()
        if self.task.lower() == 'fine_tune':
            self.model = Encoder_Transformer_finetune(dim,**self.kwargs)
        elif self.task.lower() == 'autoencoder_reconstruction':
            self.model = AutoEncoder(dim,**self.kwargs)
        elif self.task.lower() == 'transformer_reconstruction':
            self.model = Encoder_Transformer_Decoder(dim,**self.kwargs)
            
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
                self.device = torch.device("cuda" if self.cuda else "cpu")
                self.model.cuda()
                if self.task.lower() == 'autoencoder_reconstruction':
                    self.model = torch.nn.parallel.DistributedDataParallel(self.model) 
                else: # having unused parameter (classifier token)
                    self.model = torch.nn.parallel.DistributedDataParallel(self.model,find_unused_parameters=True) 
                model_without_ddp = self.model.module
        else:
            self.device = torch.device("cuda" if self.cuda else "cpu")
            self.model = DataParallel(self.model).to(self.device)        
            
        torch.backends.cudnn.benchmark = True   
        


    def training(self):
        if self.profiling == True:
            self.nEpochs = 1
        for epoch in range(self.st_epoch,self.nEpochs): 
            start = time.time()
            self.train_epoch(epoch)
            self.eval_epoch('val')
            print('______epoch summary {}/{}_____\n'.format(epoch+1,self.nEpochs)) 
            self.writer.loss_summary(lr=self.lr_handler.schedule.get_last_lr()[0])
            self.writer.accuracy_summary(mid_epoch=False)
            self.writer.save_history_to_csv()
            self.save_checkpoint_(epoch, len(self.train_loader), self.scaler) 
            end = time.time()
            
            print(f'time taken to perform {epoch}: {end-start:.2f}')
            
 
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
                    torch.cuda.nvtx.range_push("optimize")
                    self.scaler.step(self.optimizer)
                    torch.cuda.nvtx.range_pop()
                    scale = self.scaler.get_scale()
                    self.scaler.update()
                    skip_lr_sched = (scale > self.scaler.get_scale())
                
            else:
                loss_dict, loss = self.forward_pass(input_dict)
                loss.backward()
                self.optimizer.step()
                
                
            if not skip_lr_sched:
                self.lr_handler.schedule_check_and_update()
            self.writer.write_losses(loss_dict, set='train')
            
            #end_time = time.time()
            #print(f'times taken to execute step {batch_idx}: {end_time-start_time}')
            #times.append(end_time - start_time)
            torch.cuda.nvtx.range_pop()
            
            
            
            #for profiling, early stopping
            if self.profiling == True:
                if batch_idx == 10 : 
                    break

            if (batch_idx + 1) % self.validation_frequency == 0:
                print('batch index is:', batch_idx)

                ### validation ##
                self.eval_epoch('val')
                self.writer.loss_summary(lr=self.lr_handler.schedule.get_last_lr()[0])
                self.writer.accuracy_summary(mid_epoch=True)
                self.writer.experiment_title = self.writer.experiment_title
                self.writer.save_history_to_csv()
                
                self.save_checkpoint_(epoch, batch_idx, self.scaler) # validation마다 checkpoint 저장               
                self.train()
                
    def eval_epoch(self,set):
        loader = self.val_loader if set == 'val' else self.test_loader 
        self.eval(set)
        with torch.no_grad():
            #times = [] 
            for batch_idx, input_dict in enumerate(tqdm(loader, position=0, leave=True)):
                # start_time = time.time()
                with autocast():
                    loss_dict, _ = self.forward_pass(input_dict)
                # end_time = time.time()
                # print('times taken to execute step {0}: {1}'.format(batch_idx,end_time-start_time))
                #times.append(end_time-start_time)
                self.writer.write_losses(loss_dict, set=set)
                if self.profiling == True:
                    if batch_idx == 10 : 
                        break
        #print('time spent for validation:',np.mean(times)) 
        
    def forward_pass(self,input_dict):
        input_dict = {k:(v.to(self.gpu) if self.cuda else v) for k,v in input_dict.items()}
        #print('shape of input dict is :', input_dict['fmri_sequence'].size())
        output_dict = self.model(input_dict['fmri_sequence'])
        torch.cuda.nvtx.range_push("aggregate_losses")
        loss_dict, loss = self.aggregate_losses(input_dict, output_dict)
        torch.cuda.nvtx.range_pop()
        if self.task == 'fine_tune':
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
                if current_loss_value.isnan().sum() > 0:
                    warnings.warn('found nans in computation')
                    print('at {} loss'.format(loss_name))
                lamda = current_loss_dict['factor']
                factored_loss = current_loss_value * lamda
                final_loss_dict[loss_name] = factored_loss.item()
                final_loss_value += factored_loss
        final_loss_dict['total'] = final_loss_value.item()
        return final_loss_dict, final_loss_value
        
    def testing(self):
        self.eval_epoch('test')
        self.writer.loss_summary(lr=0)
        self.writer.accuracy_summary(mid_epoch=False)
        for metric_name in dir(self.writer):
            if 'history' not in metric_name:
                continue
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

    def get_last_accuracy(self):
        if hasattr(self.writer,'val_AUROC'):
            return self.writer.val_AUROC[-1]
        else:
            return None

    def save_checkpoint_(self, epoch, batch_idx, scaler):
        partial_epoch = epoch # + (batch_idx / len(self.train_loader))
        
        print('in save_checkpoint_ function, epoch is:', partial_epoch)
        loss = self.get_last_loss()
        accuracy = self.get_last_accuracy()
        title = str(self.writer.experiment_title) + '_epoch_' + str(int(epoch)) + '_batch_index_'+ str(batch_idx) # 이 함수 안에서만 쓰도록 함~
        self.save_checkpoint(
            self.writer.experiment_folder, title, partial_epoch, loss ,accuracy, scaler, self.optimizer ,schedule=self.lr_handler.schedule) #experiments에 저장
        
    # helper function of the save_checkpoint_ (don't need to merge them)
    def save_checkpoint(self, directory, title, epoch, loss, accuracy, scaler, optimizer=None,schedule=None):
        # Create directory to save to
        if not os.path.exists(directory):
            os.makedirs(directory)
        if self.amp:
            amp_state = scaler.state_dict()

        # Build checkpoint dict to save.
        ckpt_dict = {
            'model_state_dict':self.model.module.state_dict(),
            'optimizer_state_dict':optimizer.state_dict() if optimizer is not None else None,
            'epoch':epoch,
            'loss_value':loss,
            'amp_state': amp_state}
        if accuracy is not None:
            ckpt_dict['accuracy'] = accuracy
        if schedule is not None:
            ckpt_dict['schedule_state_dict'] = schedule.state_dict()
            ckpt_dict['lr'] = schedule.get_last_lr()[0]
        # 수상한 줄... 은 별 거 없고 이 모델의 path를 받아와서 저장하는 것. 그러면 transformer는 ae의 path를 가지고 있겠군 
        if hasattr(self,'loaded_model_weights_path'):
            ckpt_dict['loaded_model_weights_path'] = self.loaded_model_weights_path
        
        # Save checkpoint per one epoch 
        # commented out by JB
        # name = "{}.pth".format(core_name) 
        # torch.save(ckpt_dict, os.path.join(directory, name))
        
        core_name = title
        # best loss나 best accuracy를 가진 모델만 저장하는 코드
        if self.best_loss > loss:
            self.best_loss = loss
            name = "{}_BEST_val_loss.pth".format(core_name)
            torch.save(ckpt_dict, os.path.join(directory, name))
            print('updating best saved model...')
        if accuracy is not None and self.best_accuracy < accuracy:
            self.best_accuracy = accuracy
            name = "{}_BEST_val_accuracy.pth".format(core_name)
            torch.save(ckpt_dict, os.path.join(directory, name))
            print('updating best saved model...')


    def compute_reconstruction(self,input_dict,output_dict):
        fmri_sequence = input_dict['fmri_sequence'][:,0].unsqueeze(1)
        reconstruction_loss = self.reconstruction_loss_func(output_dict['reconstructed_fmri_sequence'],fmri_sequence)
        return reconstruction_loss

    def compute_intensity(self,input_dict,output_dict):
        per_voxel = input_dict['fmri_sequence'][:,1,:,:,:,:]
        torch.cuda.nvtx.range_push("get_intensity_voxels")
        voxels = get_intense_voxels(per_voxel, output_dict['reconstructed_fmri_sequence'].shape, self.gpu) 
        torch.cuda.nvtx.range_pop()
        output_intense = output_dict['reconstructed_fmri_sequence'][voxels]
        truth_intense = input_dict['fmri_sequence'][:,0][voxels.squeeze(1)]
        torch.cuda.nvtx.range_push("self.intensity_loss_func")
        intensity_loss = self.intensity_loss_func(output_intense.squeeze(), truth_intense)
        torch.cuda.nvtx.range_pop()
        return intensity_loss

    def compute_perceptual(self,input_dict,output_dict):
        fmri_sequence = input_dict['fmri_sequence'][:,0].unsqueeze(1)
        perceptual_loss = self.perceptual_loss_func(output_dict['reconstructed_fmri_sequence'],fmri_sequence)
        return perceptual_loss

    def compute_binary_classification(self,input_dict,output_dict):
        binary_loss = self.binary_classification_loss_func(output_dict['binary_classification'].squeeze(), input_dict[self.target].squeeze())
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
        subjects = input_dict['subject'].clone().cpu()
        for i, subj in enumerate(subjects):
            subject = str(subj.item())
            if subject not in self.writer.subject_accuracy:
                self.writer.subject_accuracy[subject] = {'score': score[i].unsqueeze(0), 'mode': self.mode, 'truth': labels[i],'count': 1}
            else:
                self.writer.subject_accuracy[subject]['score'] = torch.cat([self.writer.subject_accuracy[subject]['score'], score[i].unsqueeze(0)], dim=0)
                self.writer.subject_accuracy[subject]['count'] += 1

    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs
