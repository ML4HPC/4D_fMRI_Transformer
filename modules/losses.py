import sys
sys.path.insert(1, './CNN_hardparameter_sharing/models')

import torch
import torch.nn as nn
from torchvision import models
from .densenet3d import *
import numpy as np


def get_intense_voxels(yy,shape,gpu):
    
    '''
    # previous code
    y = yy.clone().cuda(gpu)
    voxels = torch.empty(shape,device=gpu)
    low_quantile, high_quantile, = (0.9,0.99)
    for batch in range(y.shape[0]):
        for TR in range(y.shape[-1]):
            yy = y[batch, :, :, :, TR]
            background = yy[0, 0, 0]
            yy[yy <= background] = 0
            yy = abs(yy)
            voxels[batch, :, :, :, :, TR] = (yy > torch.quantile(yy[yy > 0], low_quantile)).unsqueeze(0)
    return voxels.view(shape)>0
    '''
    
    y1 = yy.clone()
    b, h, w, d, t = y1.shape
    
    y1 = y1.permute(0,4,1,2,3).contiguous().view(b*t, h*w*d)
    y1[y1<=y1[:,0:1]]=0
    y1 = abs(y1)
    
    low_quantile, high_quantile = (0.9,0.99)
    
    to_quantile = 1 - ((y1>y1[:,0:1]).sum(dim=1) / y1.shape[1] * (1-low_quantile))
    
    voxels = (y1 > torch.quantile(y1.float(), to_quantile.float(), dim=1).diag().unsqueeze(1))
    
    xx1 = voxels.view(b,t,h,d,w).permute(0,2,3,4,1).view(shape)>0
    
    return xx1
    
    
## Stella added this module
class DenseNet3D(nn.Module):
    def __init__(self):
        super(DenseNet3D, self).__init__()
        model = densenet3D121()
        #dn3_weight = torch.load('./CNN_hardparameter_sharing/densenet3D121_UKB_age_180097.pth')
        dn3_weight = torch.load('./CNN_hardparameter_sharing/UKB_sex_densenet3D121_6cbde7.pth')
        #dn3_weight.popitem() # remove 'classifiers.0.0.bias'
        #dn3_weight.popitem() # remove 'classifiers.0.0.weight'
        model.load_state_dict(dn3_weight)
        features = model.features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        
        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # shape of x in DenseNet3D: torch.Size([20, 1, 75, 93, 81]) - batch*T, channel, width, height, depth
        ## c.f. ([648, 3, 75, 93]) in VGG
        
        rnd = np.random.randint(1, x.shape[0]) # for batch 1
        x = x[rnd:rnd+1, :, :, :, :]
        # shape of x is : torch.Size([1, 1, 75, 93, 81])
        
        # expected : 5-dimensional input for 5-dimensional weight [64, 1, 7, 7, 7]
        # get : 5-dimensional input of size [1, 1, 75, 93, 81]
        
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        #shape of h_relu_1_2: torch.Size([1, 64, 38, 24, 21]) #초반
        #shape of h_relu_2_2: torch.Size([1, 1024, 9, 6, 5]) #후반

        out = (h_relu_1_2, h_relu_2_2)
        
        return out
        

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # shape of x in Vgg16: torch.Size([648, 3, 75, 93])
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        #h_relu_3_3 = h
        out = (h_relu_1_2, h_relu_2_2)
        #shape of out of h_relu_1_2 : torch.Size([648, 64, 75, 93])
        #shape of out of h_relu_2_2 : torch.Size([648, 128, 37, 46])

        return out


# Stella added this module
# at reconstructed tensor level
# 성능 별로면 mask loss 참조해서 (4, 20, 2640) 짜리로 ㄱㄱ

# contrastive loss at transformer output level
class Cont_Loss(nn.Module):
    def __init__(self,**kwargs):
        super(Cont_Loss, self).__init__()
        #task = kwargs.get('task')
        self.cont_loss = None

    def forward(self, input):
        #print('input of the contrastive loss is:', input.shape)
        margin = 60000
        #input.shape - [batch, T, embedding] [4, 20, 2640]
        _, seq_len, _ = input.shape
        loss = 0
        for a in range(seq_len):
            for b in range(seq_len):
                if a>b:
                    input_1 = input[:, a:a+1, :]
                    input_2 = input[:, b:b+1, :]
                    squared_distance = torch.sum(torch.square((input_1 - input_2)))
                    #print('squared distance is:', squared_distance)
                    if a-b == 1:
                        label = 0
                    else:
                        label = 1
                    loss_function = label*squared_distance + (1 - label)*(max (0, (margin - squared_distance)))
                    #print('{} and {} cont loss is: {}'.format(a, b, loss_function)) # a랑 b가 가까우면 loss function이 작아야 함.
                    loss+=loss_function
        self.cont_loss = loss/(seq_len*(seq_len-1)*1000) #np.sum(loss_function)/len(input_1) #just for scaling
        return self.cont_loss

# contrastive loss at reconstructecd tensor level
'''
class Cont_Loss(nn.Module):
    def __init__(self,**kwargs):
        super(Cont_Loss, self).__init__()
        #task = kwargs.get('task')
        self.cont_loss = None

    def forward(self, input):
        margin = 60000
        #input.shape - [batch, channel, width, height, depth, T] [2, 1, 75, 93, 81, 20]
        _, _, _, _, _, seq_len = input.shape
        loss = 0
        for a in range(seq_len):
            for b in range(seq_len):
                if a>b:
                    input_1 = input[:, :, :, :, :, a]
                    input_2 = input[:, :, :, :, :, b]
                    squared_distance = torch.sum(torch.square((input_1 - input_2)))
                    #print('squared distance is:', squared_distance)
                    if a-b == 1:
                        label = 0
                    else:
                        label = 1
                    loss_function = label*squared_distance + (1 - label)*(max (0, (margin - squared_distance)))
                    #print('{} and {} cont loss is: {}'.format(a, b, loss_function)) # a랑 b가 가까우면 loss function이 작아야 함.
                    loss+=loss_function
        self.cont_loss = loss/(seq_len*(seq_len-1)*1000) #np.sum(loss_function)/len(input_1) #just for scaling
        #print('cont loss is:', self.cont_loss)
        return self.cont_loss
'''

class Mask_Loss(nn.Module):
    def __init__(self,**kwargs):
        super(Mask_Loss, self).__init__()
        self.mask_loss = 0.0
        
    def forward(self, input, mask_list, target):
        margin = 15000
        '''
        shape of input & target in compute mask is: torch.Size([4, 20, 2640])
        input : transformer에 넣기 전 encoded된 벡터
        mask_list : masking한 index (0 ~ 19 사이 임의의 정수 3개) : torch.Size([4, 3])
        '''
        seq_len = input.shape[1]
        batch_size = input.shape[0]
        masked_index_size = mask_list.shape[1] # 3
        #shape of mask_list is: torch.Size([1, 3])..? 왜..? 아.... multinode... -> when slurm
        #shape of mask_list is: torch.Size([4, 3])..? -> when interactive node
        
        whole_loss = 0
        
        # batch size is divided by 4 for slurm script (이거 어떻게 조건 줘야 할 지 모르겠음 - Stella)
        for j in range(batch_size):
            loss_per_batch = 0
            for k in range(masked_index_size):
                idx_masked_vox = mask_list[j][k]
                #print('index of masked voxel is:',idx_masked_vox)
                ## 복원된 voxel
                reh = target[j, idx_masked_vox:idx_masked_vox+1, :]

                ## contrastive loss
                loss = 0
                for i in range(seq_len):
                    input_frame = input[j, i:i+1, :]
                    if abs(idx_masked_vox-i) <= 1:
                        #print('nearby voxel index is:', i)
                        label = 1
                    else:
                        label = 0

                    squared_distance = torch.sum(torch.square((reh - input_frame)))
                    #print('distance is {0} from {1}'.format(squared_distance, i))
                    loss_function = label*squared_distance + (1 - label)*(max (0, (margin - squared_distance)))
                    #print('loss fucntion is {0} from {1}'.format(loss_function, i))
                    loss_function/=(seq_len*(seq_len-1))

                    loss+=loss_function  # batch j에서 (j = 0, 1, 2, 3) masked index k에서의 loss
                '''loss 계산 완료!'''
                
                loss_per_batch+=loss
            loss_per_batch/=masked_index_size
            #print('loss for batch {0} is {1}'.format(j, loss_per_batch))
            
        whole_loss+=loss_per_batch
        
        self.mask_loss = whole_loss/(batch_size*100) #np.sum(loss_function)/len(input_1) #just for scaling
        
        print('mask loss is:', self.mask_loss)
        
        
        return self.mask_loss 
        

        
        
class Percept_Loss(nn.Module):
    def __init__(self,**kwargs):
        super(Percept_Loss, self).__init__()
        print('notice: changed layers in perceptual back to old version')
        task = kwargs.get('task')
        if task == 'test':
            pass
        elif task == 'autoencoder_reconstruction':
            self.memory_constraint = 0.25
        elif task == 'transformer_reconstruction':
            self.memory_constraint = 0.1
        if 'reconstruction' in task:
            self.which_model = kwargs.get('which_perceptual')
            if self.which_model == 'vgg':
                self.vgg = Vgg16().to(memory_format=torch.channels_last)
                if kwargs.get('cuda'):
                    self.vgg.cuda(kwargs.get('gpu'))
                self.loss = nn.MSELoss()
            elif self.which_model == 'densenet3d':
                self.densenet3d = DenseNet3D() #.to(memory_format=torch.channels_last)
                if kwargs.get('cuda'):
                    self.densenet3d.cuda(kwargs.get('gpu'))
                self.loss = nn.MSELoss()


    def forward(self, input, target):
        assert input.shape == target.shape, 'input and target should have identical dimension'
        assert len(input.shape) == 6
        batch, channel, width, height, depth, T = input.shape
        if self.which_model == 'vgg':
            num_slices = batch * T * depth
            represent = torch.randperm(num_slices)[:int(num_slices * self.memory_constraint)]
            input = input.permute(0, 5, 1, 4, 2, 3).reshape(num_slices, 1, width, height)
            target = target.permute(0, 5, 1, 4, 2, 3).reshape(num_slices, 1, width, height)
            input = input[represent, :, :, :].repeat(1,3,1,1)

            # Convert from NCHW to NHWC to accellerate
            input = input.contiguous(memory_format=torch.channels_last)
            target = target[represent, :, :, :].repeat(1,3,1,1)
            input = self.vgg(input)
            target = self.vgg(target)
            loss = 0
            for i,j in zip(input,target):
                loss += self.loss(i,j)
        elif self.which_model == 'densenet3d':
            ## don't cut 3D to 2D! just get whole 3D data here ##
            num_slices = batch * T
            represent = torch.randperm(num_slices)[:int(num_slices * self.memory_constraint)]
            # permute -> batch, T, channel, width, height, depth
            
            input = input.permute(0, 5, 1, 2, 3, 4).reshape(num_slices, 1, width, height, depth)
            #input = input.contiguous(memory_format=torch.channels_last)
            
            target = target.permute(0, 5, 1, 2, 3, 4).reshape(num_slices, 1, width, height, depth)
            
            # compute loss
            input = self.densenet3d(input)
            target = self.densenet3d(target)
            loss = 0
            for i,j in zip(input,target):
                loss += self.loss(i,j)
        return loss
