import torch
import torch.nn as nn
from torchvision import models

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
    
    voxels = (y1 > torch.quantile(y1, to_quantile, dim=1).diag().unsqueeze(1))
    
    xx1 = voxels.view(b,t,h,d,w).permute(0,2,3,4,1).view(shape)>0
    
    return xx1
    
    

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
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        #h_relu_3_3 = h
        out = (h_relu_1_2, h_relu_2_2)
        return out


# Stella added this module
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
                    #print('{} and {} cont loss is: {}'.format(a, b, loss_function)) # a랑 b가 가까우면 loss function이 작아야 함. 아주 나이스! 
                    loss+=loss_function
        self.cont_loss = loss/(seq_len*(seq_len-1)*1000) #np.sum(loss_function)/len(input_1) #just for scaling
        print('cont loss is:', self.cont_loss)
        return self.cont_loss
        
        
        
class Percept_Loss(nn.Module):
    def __init__(self,**kwargs):
        super(Percept_Loss, self).__init__()
        print('notice: changed layers in perceptual back to old version')
        task = kwargs.get('task')
        if task == 'autoencoder_reconstruction':
            self.memory_constraint = 0.25
        elif task == 'transformer_reconstruction':
            self.memory_constraint = 0.1
        if 'reconstruction' in task:
            self.vgg = Vgg16().to(memory_format=torch.channels_last)
            if kwargs.get('cuda'):
                self.vgg.cuda(kwargs.get('gpu'))
            self.loss = nn.MSELoss()

    def forward(self, input, target):
        assert input.shape == target.shape, 'input and target should have identical dimension'
        assert len(input.shape) == 6
        batch, channel, width, height, depth, T = input.shape
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
        return loss
