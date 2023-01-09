import os
from abc import ABC, abstractmethod
import torch
from transformers import BertConfig,BertPreTrainedModel, BertModel
from datetime import datetime
import torch.nn as nn
from .nvidia_blocks import *
import random

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    def determine_shapes(self,encoder,dim):
        def get_shape(module,input,output):
            module.input_shape = tuple(input[0].shape[-3:])
            module.output_shape = tuple(output[0].shape[-3:])
        hook1 = encoder.down_block1.register_forward_hook(get_shape)
        hook2 = encoder.down_block3.register_forward_hook(get_shape)
        
        
        #input_shape = (1,2,) + dim 
        input_shape = (1,) + dim  #batch,norms,H,W,D,time
        x = torch.ones((input_shape))
        with torch.no_grad():
            encoder(x)
            del x
        self.shapes = {'dim_0':encoder.down_block1.input_shape,
                       'dim_1':encoder.down_block1.output_shape,
                       'dim_2':encoder.down_block3.input_shape,
                       'dim_3':encoder.down_block3.output_shape}
        hook1.remove()
        hook2.remove()

    def register_vars(self,**kwargs):
        intermediate_vec = kwargs.get('transformer_emb_size')
        transformer_dropout = kwargs.get('transformer_dropout_rate')
        # Dropout rates for each layer
        if kwargs.get('task') == 'fine_tune':
            self.dropout_rates = {'input': 0, 'green': 0.35,'mobilev3': 0.35,'Up_green': 0,'transformer':transformer_dropout}
        else:
            self.dropout_rates = {'input': 0, 'green': 0.2, 'mobilev3': 0.35,'Up_green': 0.2,'transformer':transformer_dropout}

        self.BertConfig = BertConfig(hidden_size=kwargs.get('transformer_emb_size'), vocab_size=1,
                                     num_hidden_layers=kwargs.get('transformer_hidden_layers'),
                                     num_attention_heads=kwargs.get('transformer_num_attention_heads'), max_position_embeddings=kwargs.get('sequence_length')+1,
                                     hidden_dropout_prob=self.dropout_rates['transformer'])#, torchscript=True)
        # max_position_embeddings : The maximum sequence length that this model might
        #                           ever be used with. Typically set this to something large just in case
        #                           (e.g., 512 or 1024 or 2048).

        self.label_num = 1
        if kwargs.get('with_voxel_norm'):
            self.inChannels = 2
        else:
            self.inChannels = 1
        self.outChannels = 1
        self.model_depth = 4
        self.intermediate_vec = intermediate_vec
        self.use_cuda = kwargs.get('gpu') #'cuda'
        self.shapes = kwargs.get('shapes')

    def load_partial_state_dict(self, state_dict,load_cls_embedding):
        print('loading parameters onto new model...')
        own_state = self.state_dict()
        loaded = {name:False for name in own_state.keys()}
        for name, param in state_dict.items():
            if name not in own_state:
                print('notice: {} is not part of new model and was not loaded.'.format(name))
                continue
            elif 'cls_embedding' in name and not load_cls_embedding:
                continue
            elif 'position' in name and param.shape != own_state[name].shape:
                print('debug line above')
                continue
            param = param.data
            own_state[name].copy_(param)
            loaded[name] = True
        for name,was_loaded in loaded.items():
            if not was_loaded:
                print('notice: named parameter - {} is randomly initialized'.format(name))

class Encoder(BaseModel):
    def __init__(self,**kwargs):
        super(Encoder, self).__init__()
        self.register_vars(**kwargs)
        self.down_block1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(self.inChannels, self.model_depth, kernel_size=3, stride=1, padding=1)),
            ('sp_drop0', nn.Dropout3d(self.dropout_rates['input'])),
            ('green0', GreenBlock(self.model_depth, self.model_depth, self.dropout_rates['green'])),
            ('downsize_0', nn.Conv3d(self.model_depth, self.model_depth * 2, kernel_size=3, stride=2, padding=1))]))
        self.down_block2 = nn.Sequential(OrderedDict([
            ('green10', GreenBlock(self.model_depth * 2, self.model_depth * 2, self.dropout_rates['green'])),
            ('green11', GreenBlock(self.model_depth * 2, self.model_depth * 2, self.dropout_rates['green'])),
            ('downsize_1', nn.Conv3d(self.model_depth * 2, self.model_depth * 4, kernel_size=3, stride=2, padding=1))]))
        self.down_block3 = nn.Sequential(OrderedDict([
            ('green20', GreenBlock(self.model_depth * 4, self.model_depth * 4, self.dropout_rates['green'])),
            ('green21', GreenBlock(self.model_depth * 4, self.model_depth * 4, self.dropout_rates['green'])),
            ('downsize_2', nn.Conv3d(self.model_depth * 4, self.model_depth * 8, kernel_size=3, stride=2, padding=1))]))
        self.final_block = nn.Sequential(OrderedDict([
            ('green30', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green'])),
            ('green31', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green'])),
            ('green32', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green'])),
            ('green33', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green']))]))

    def forward(self,x):
        torch.cuda.nvtx.range_push("down_block1")
        x = self.down_block1(x)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("down_block2")
        x = self.down_block2(x)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("down_block3")
        x = self.down_block3(x)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("final_block")
        x = self.final_block(x)
        torch.cuda.nvtx.range_pop()
        return x


class BottleNeck_in(BaseModel):
    def __init__(self,**kwargs):
        super(BottleNeck_in, self).__init__()
        self.register_vars(**kwargs)
        self.reduce_dimension = nn.Sequential(OrderedDict([
            ('group_normR', nn.GroupNorm(num_channels=self.model_depth * 8, num_groups=8)),
            # ('norm0', nn.BatchNorm3d(model_depth * 8)),
            ('reluR0', nn.LeakyReLU(inplace=True)),
            ('convR0', nn.Conv3d(self.model_depth * 8, self.model_depth // 2, kernel_size=(3, 3, 3), stride=1, padding=1)),
        ]))
        flat_factor = tuple_prod(self.shapes['dim_3'])
        self.flatten = nn.Flatten()
        if (flat_factor * self.model_depth // 2) == self.intermediate_vec:
            self.into_bert = nn.Identity()
            print('flattened vec identical to intermediate vector...\ndroppping fully conneceted bottleneck...')
        else:
            self.into_bert = nn.Linear(in_features=(self.model_depth // 2) * flat_factor, out_features=self.intermediate_vec)

    def forward(self, inputs):
        x = self.reduce_dimension(inputs)
        x = self.flatten(x)
        x = self.into_bert(x)

        return x


class BottleNeck_out(BaseModel):
    def __init__(self,**kwargs):
        super(BottleNeck_out, self).__init__()
        self.register_vars(**kwargs)
        flat_factor = tuple_prod(self.shapes['dim_3'])
        minicube_shape = (self.model_depth // 2,) + self.shapes['dim_3']
        self.out_of_bert = nn.Linear(in_features=self.intermediate_vec, out_features=(self.model_depth // 2) * flat_factor)
        self.expand_dimension = nn.Sequential(OrderedDict([
            ('unflatten', nn.Unflatten(1, minicube_shape)),
            ('group_normR', nn.GroupNorm(num_channels=self.model_depth // 2, num_groups=2)),
            # ('norm0', nn.BatchNorm3d(model_depth * 8)),
            ('reluR0', nn.LeakyReLU(inplace=True)),
            ('convR0', nn.Conv3d(self.model_depth // 2, self.model_depth * 8, kernel_size=(3, 3, 3), stride=1, padding=1)),
        ]))

    def forward(self, x):
        x = self.out_of_bert(x)
        return self.expand_dimension(x)

class Decoder(BaseModel):
    def __init__(self,**kwargs):
        super(Decoder, self).__init__()
        self.register_vars(**kwargs)
        # determine_shapes에서 등록한 사이즈대로 Upsample을 실행
        self.decode_block = nn.Sequential(OrderedDict([
            ('upgreen0', UpGreenBlock(self.model_depth * 8, self.model_depth * 4, self.shapes['dim_2'], self.dropout_rates['Up_green'])),
            ('upgreen1', UpGreenBlock(self.model_depth * 4, self.model_depth * 2, self.shapes['dim_1'], self.dropout_rates['Up_green'])),
            ('upgreen2', UpGreenBlock(self.model_depth * 2, self.model_depth, self.shapes['dim_0'], self.dropout_rates['Up_green'])),
            ('blue_block', nn.Conv3d(self.model_depth, self.model_depth, kernel_size=3, stride=1, padding=1)),
            ('output_block', nn.Conv3d(in_channels=self.model_depth, out_channels=self.outChannels, kernel_size=1, stride=1))
        ]))

    def forward(self, x):
        x = self.decode_block(x)
        return x


class AutoEncoder(BaseModel):
    def __init__(self,dim,**kwargs):
        super(AutoEncoder, self).__init__()
        # ENCODING
        self.task = 'autoencoder_reconstruction'
        self.encoder = Encoder(**kwargs).to(memory_format=torch.channels_last_3d)
        self.determine_shapes(self.encoder,dim)
        kwargs['shapes'] = self.shapes
        # BottleNeck into bert
        self.into_bert = BottleNeck_in(**kwargs).to(memory_format=torch.channels_last_3d)

        # BottleNeck out of bert
        self.from_bert = BottleNeck_out(**kwargs).to(memory_format=torch.channels_last_3d)

        # DECODER
        self.decoder = Decoder(**kwargs).to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        # if x.isnan().any():
        #torch._assert(x.isnan().any(), 'nans in data!')
            # print('nans in data!')
        batch_size, Channels_in, W, H, D, T = x.shape
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(batch_size * T, Channels_in, W, H, D)
        # changed from NCHDW to NHWDC format for accellerating
        x = x.contiguous(memory_format=torch.channels_last_3d)
        torch.cuda.nvtx.range_push("encoder")
        encoded  = self.encoder(x)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("into_bert")
        encoded = self.into_bert(encoded)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("from_bert")  
        encoded = self.from_bert(encoded)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("decoder")
        reconstructed_image = self.decoder(encoded)
        torch.cuda.nvtx.range_pop()
        _, Channels_out, W, H, D = reconstructed_image.shape
        torch.cuda.nvtx.range_push("reshaping")
        reconstructed_image = reconstructed_image.reshape(batch_size, T, Channels_out, W, H, D).permute(0, 2, 3, 4, 5, 1)
        torch.cuda.nvtx.range_pop()
        return {'reconstructed_fmri_sequence': reconstructed_image}

class Transformer_Block(BertPreTrainedModel, BaseModel):
    def __init__(self,config,**kwargs):
        super(Transformer_Block, self).__init__(config)
        self.register_vars(**kwargs)
        self.cls_pooling = True
        self.bert = BertModel(self.BertConfig, add_pooling_layer=self.cls_pooling)
        #self.bert = torch.jit.trace(self.bert, torch.ones((1, 20))) #, self.BertConfig.hidden_size)))
        #torch._C._jit_set_autocast_mode(True)
        self.init_weights()
        self.register_buffer('cls_id', (torch.ones((1, 1, self.BertConfig.hidden_size)) * 0.5),persistent=False)
        #self.cls_id = torch.ones(1, 1, self.BertConfig.hidden_size, device=kwargs.get('gpu'), requires_grad=False) * 0.5 # nn.Parameter(torch.ones(1, 1, self.BertConfig.hidden_size) * 0.5)
        self.cls_embedding = nn.Sequential(nn.Linear(self.BertConfig.hidden_size, self.BertConfig.hidden_size), nn.LeakyReLU())
        
    def concatenate_cls(self, x):
        '''
        shape of x in transformer block: torch.Size([1, 20, 2640])
        '''
        # torch.cuda.nvtx.range_push("register_buffer")
        #self.register_buffer('cls_id', (torch.ones((x.size()[0], 1, self.BertConfig.hidden_size),device=x.get_device()) * 0.5),persistent=False)
        #cls_token = self.cls_embedding(self.cls_id)
        # torch.cuda.nvtx.range_pop()
        cls_token = self.cls_embedding(self.cls_id.expand(x.size()[0], -1, -1))
        
        # print('Size of cls_token: ', cls_token.size())
        # print('Size of cls_id: ', self.cls_id.size())
        # print('Size of x: ', x.size())
        return torch.cat([cls_token, x], dim=1)


    def forward(self, x ):
        inputs_embeds = self.concatenate_cls(x=x)
        outputs = self.bert(input_ids=None,
                            attention_mask=None,
                            token_type_ids=None,
                            position_ids=None,
                            head_mask=None,
                            inputs_embeds=inputs_embeds, #give our embeddings
                            encoder_hidden_states=None,
                            encoder_attention_mask=None,
                            output_attentions=None,
                            output_hidden_states=None,
                            return_dict=self.BertConfig.use_return_dict
                            )

        sequence_output = outputs[0][:, 1:, :]
        pooled_cls = outputs[1]

        return {'sequence': sequence_output, 'cls': pooled_cls}


class Encoder_Transformer_Decoder(BaseModel):
    def __init__(self, dim,**kwargs):
        super(Encoder_Transformer_Decoder, self).__init__()
        self.task = 'transformer_reconstruction'
        self.register_vars(**kwargs)
        # ENCODING
        self.encoder = Encoder(**kwargs).to(memory_format=torch.channels_last_3d)
        self.determine_shapes(self.encoder,dim)
        kwargs['shapes'] = self.shapes

        # changed from NCHDW to NHWDC format for accellerating

        # BottleNeck into bert
        self.into_bert = BottleNeck_in(**kwargs).to(memory_format=torch.channels_last_3d)

        # transformer
        self.transformer = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)

        # BottleNeck out of bert
        self.from_bert = BottleNeck_out(**kwargs).to(memory_format=torch.channels_last_3d)

        # DECODER
        self.decoder = Decoder(**kwargs).to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        batch_size, inChannels, W, H, D, T = x.shape
        
        #print('shape of x:', x.size()) 
        
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(batch_size * T, inChannels, W, H, D)
        # changed from NCHDW to NHWDC format for accellerating
        x = x.contiguous(memory_format=torch.channels_last_3d) 
        torch.cuda.nvtx.range_push("Encoder") 
        encoded = self.encoder(x)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("into_bert")
        encoded = self.into_bert(encoded)
        torch.cuda.nvtx.range_pop()
        encoded = encoded.reshape(batch_size, T, -1)
        torch.cuda.nvtx.range_push("transformer")
        transformer_dict = self.transformer(encoded)
        transformer_sequence = transformer_dict['sequence']
        torch.cuda.nvtx.range_pop()
        out = transformer_dict['sequence'].reshape(batch_size * T, -1)
        torch.cuda.nvtx.range_push("from_bert")
        out = self.from_bert(out)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("decoder")
        reconstructed_image = self.decoder(out)
        torch.cuda.nvtx.range_pop()
        reconstructed_image = reconstructed_image.reshape(batch_size, T, self.outChannels, W, H, D).permute(0, 2, 3, 4, 5, 1)
        
        # for masked loss
        ## step 1 masking
        encoded_copy = encoded # copy
        seq_len = encoded_copy.shape[1] # 20
        ratio = 0.15
        mask_list = []
        
        ### not overlapping
        for i in range(int(seq_len * ratio)):  
            x = random.randint(0, seq_len-1)
            while x in mask_list:
                x = random.randint(0, seq_len-1)
            mask_list.append(x)
        
        for x in mask_list:
            encoded_copy[:, x:x+1, :] = torch.zeros(encoded_copy.shape[0], 1, encoded_copy.shape[2])
        
        device = encoded.get_device() if encoded.get_device() >= 0 else 'cpu'
        mask_list = torch.tensor(mask_list).reshape(1, -1).to(device)
        #print('mask list of model.py is:', mask_list)
        
        ## step 2 rehabiliation of masked input with transformer
        transformer_dict_for_mask = self.transformer(encoded_copy)
        transformer_sequence_for_mask = transformer_dict_for_mask['sequence'] # torch.Size([1, 20, 2640])        
        transformer_sequence_for_mask = transformer_sequence_for_mask.reshape(batch_size, T, -1)
        
        return {'reconstructed_fmri_sequence': reconstructed_image,
                'transformer_input_sequence' : encoded,
                'mask_list' : mask_list,
                'transformer_output_sequence_for_mask_learning' : transformer_sequence_for_mask,
                'transformer_output_sequence': transformer_sequence}
                
        # Stella modified this (added transformer_sequence)


class Encoder_Transformer_finetune(BaseModel):
    def __init__(self,dim,**kwargs):
        super(Encoder_Transformer_finetune, self).__init__()
        self.task = kwargs.get('fine_tune_task')
        self.return_value = kwargs.get('return_value')
        self.register_vars(**kwargs)
        # ENCODING
        self.encoder = Encoder(**kwargs).to(memory_format=torch.channels_last_3d)
        self.determine_shapes(self.encoder, dim)
        kwargs['shapes'] = self.shapes
        # BottleNeck into bert
        self.into_bert = BottleNeck_in(**kwargs).to(memory_format=torch.channels_last_3d)

        # transformer
        self.transformer = Transformer_Block(self.BertConfig,**kwargs).to(memory_format=torch.channels_last_3d)
        # finetune classifier
        #if kwargs.get('fine_tune_task') == 'regression':
        #    self.final_activation_func = nn.LeakyReLU()
        #elif kwargs.get('fine_tune_task') == 'binary_classification':
        #    self.final_activation_func = nn.Sigmoid()
        #    self.label_num = 1
        #self.regression_head = nn.Sequential(nn.Linear(self.BertConfig.hidden_size, self.label_num),self.final_activation_func).to(memory_format=torch.channels_last_3d)
        self.regression_head = nn.Sequential(nn.Linear(self.BertConfig.hidden_size, self.label_num)).to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        batch_size, inChannels, W, H, D, T = x.shape
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(batch_size * T, inChannels, W, H, D)
        # Stella will modify this part for stacking
        # changed from NCHDW to NHWDC format for accellerating
        x = x.contiguous(memory_format=torch.channels_last_3d)
        encoded = self.encoder(x)
        encoded = self.into_bert(encoded)
        encoded = encoded.reshape(batch_size, T, -1)
        torch.cuda.nvtx.range_push("transformers")
        transformer_dict = self.transformer(encoded)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("regression head")
        CLS = transformer_dict['cls']
        prediction = self.regression_head(CLS)
        torch.cuda.nvtx.range_pop()
        if self.return_value == True:
            return prediction
        else:
            return {self.task:prediction}
        


class MobileNet_v2_Transformer_finetune(BaseModel):
    def __init__(self,dim,**kwargs):
        super(MobileNet_v2_Transformer_finetune, self).__init__()
        self.task = kwargs.get('fine_tune_task')
        self.register_vars(**kwargs)

        last_channel = self.intermediate_vec
        
        # ENCODING
        self.encoder = MobileNetV2(last_channel).to(memory_format=torch.channels_last_3d)
        #self.determine_shapes(self.encoder, dim)
        #kwargs['shapes'] = self.shapes
        # BottleNeck into bert
        #self.into_bert = BottleNeck_in(**kwargs).to(memory_format=torch.channels_last_3d)

        # transformer
        self.transformer = Transformer_Block(self.BertConfig,**kwargs).to(memory_format=torch.channels_last_3d)
        # finetune classifier
        #if kwargs.get('fine_tune_task') == 'regression':
        #    self.final_activation_func = nn.LeakyReLU()
        #elif kwargs.get('fine_tune_task') == 'binary_classification':
        #    self.final_activation_func = nn.Sigmoid()
        #    self.label_num = 1
        #self.regression_head = nn.Sequential(nn.Linear(self.BertConfig.hidden_size, self.label_num),self.final_activation_func).to(memory_format=torch.channels_last_3d)
        self.regression_head = nn.Sequential(nn.Linear(self.BertConfig.hidden_size, self.label_num)).to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        batch_size, inChannels, W, H, D, T = x.shape
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(batch_size * T, inChannels, W, H, D)

        # changed from NCHDW to NHWDC format for accellerating
        x = x.contiguous(memory_format=torch.channels_last_3d)
        encoded = self.encoder(x)
        #encoded = self.into_bert(encoded)
        encoded = encoded.reshape(batch_size, T, -1)
        torch.cuda.nvtx.range_push("transformers")
        transformer_dict = self.transformer(encoded)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("regression head")
        CLS = transformer_dict['cls']
        prediction = self.regression_head(CLS)
        torch.cuda.nvtx.range_pop()
        return {self.task:prediction}

class MobileNetV3(BaseModel):
    def __init__(self, **kwargs):
        super(MobileNetV3, self).__init__()
        self.register_vars(**kwargs)

        input_size=96
        input_channel = 16
        mode='small'
        width_mult=1.0
        last_channel = self.intermediate_vec
        norm_layer = nn.BatchNorm3d
        nlin_layer = nn.ReLU

        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]

        elif mode == 'mini':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 8,  8,  True,  'RE', 2],
                [3, 36,  12,  False, 'RE', 2],
                [3, 44,  12,  False, 'RE', 1],
                [5, 48,  20,  True,  'HS', 2],
                # [3, 16,  16,  True,  'RE', 2],
                # [3, 72,  24,  False, 'RE', 2],
                # [3, 88,  24,  False, 'RE', 1],
                # [5, 96,  40,  True,  'HS', 2],
                # [5, 240, 40,  True,  'HS', 1],
                # [5, 240, 40,  True,  'HS', 1],
                # [5, 120, 48,  True,  'HS', 1],
                # [5, 144, 48,  True,  'HS', 1],
                # [5, 288, 96,  True,  'HS', 2],
                # [5, 576, 96,  True,  'HS', 1],
                # [5, 576, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, 2, norm_layer=norm_layer, nlin_layer=Hswish)]

        # Dropout
        self.features.append(nn.Dropout3d(self.dropout_rates['input']))
        #self.classifier = []

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            # Dropout after bottleneck blocks
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl, drop_rate=self.dropout_rates['mobilev3'], norm_layer=norm_layer, nlin_layer=nlin_layer))
            input_channel = output_channel

        # building last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1x1_bn(input_channel, last_conv, norm_layer=norm_layer,nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool3d(1))
            self.features.append(nn.Conv3d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(conv_1x1x1_bn(input_channel, last_conv, norm_layer=norm_layer, nlin_layer=Hswish))
            # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
            self.features.append(nn.AdaptiveAvgPool3d(1))
            self.features.append(nn.Conv3d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'mini':
            last_conv = make_divisible(48 * width_mult)
            self.features.append(conv_1x1x1_bn(input_channel, last_conv, norm_layer=norm_layer, nlin_layer=Hswish))
            # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
            self.features.append(nn.AdaptiveAvgPool3d(1))
            self.features.append(nn.Conv3d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))


        else:
            raise NotImplementedError

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=dropout),    # refer to paper section 6
        #     nn.Linear(last_channel, n_class),
        # )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = x.mean(3).mean(2)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class MobileNet_v3_Transformer_finetune(BaseModel):
    def __init__(self,dim,**kwargs):
        super(MobileNet_v3_Transformer_finetune, self).__init__()
        self.task = kwargs.get('fine_tune_task')
        self.register_vars(**kwargs)
        
        # ENCODING
        self.encoder = MobileNetV3(**kwargs).to(memory_format=torch.channels_last_3d)
        #self.determine_shapes(self.encoder, dim)
        #kwargs['shapes'] = self.shapes
        # BottleNeck into bert
        #self.into_bert = BottleNeck_in(**kwargs).to(memory_format=torch.channels_last_3d)

        # transformer
        self.transformer = Transformer_Block(self.BertConfig,**kwargs).to(memory_format=torch.channels_last_3d)
        # finetune classifier
        #if kwargs.get('fine_tune_task') == 'regression':
        #    self.final_activation_func = nn.LeakyReLU()
        #elif kwargs.get('fine_tune_task') == 'binary_classification':
        #    self.final_activation_func = nn.Sigmoid()
        #    self.label_num = 1
        #self.regression_head = nn.Sequential(nn.Linear(self.BertConfig.hidden_size, self.label_num),self.final_activation_func).to(memory_format=torch.channels_last_3d)
        self.regression_head = nn.Sequential(nn.Linear(self.BertConfig.hidden_size, self.label_num)).to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        batch_size, inChannels, W, H, D, T = x.shape
        x = x.permute(0, 5, 1, 2, 3, 4).reshape(batch_size * T, inChannels, W, H, D)

        # changed from NCHDW to NHWDC format for accellerating
        x = x.contiguous(memory_format=torch.channels_last_3d)
        encoded = self.encoder(x)
        #encoded = self.into_bert(encoded)
        encoded = encoded.reshape(batch_size, T, -1)
        torch.cuda.nvtx.range_push("transformers")
        transformer_dict = self.transformer(encoded)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("regression head")
        CLS = transformer_dict['cls']
        prediction = self.regression_head(CLS)
        torch.cuda.nvtx.range_pop()
        return {self.task:prediction}

## baseline model - Stella
class Transformer_Net(BaseModel):
    def __init__(self, **kwargs):
        super(Transformer_Net, self).__init__()
        self.task = kwargs.get('fine_tune_task')
        self.register_vars(**kwargs)
        # transformer
        self.transformer = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
        if self.task == 'regression':
            self.final_activation_func = nn.LeakyReLU()
        elif self.task == 'binary_classification':
            self.final_activation_func = nn.Sigmoid()
            self.label_num = 1
        self.regression_head = nn.Linear(self.intermediate_vec, self.label_num) #.to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        
        batch_size, T, ROIs = x.shape # (batch size, 368, 84)

        torch.cuda.nvtx.range_push("transformer")

        
        transformer_dict = self.transformer(x)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("reshape")
        
        '''
        size of out seq is: torch.Size([1, 361, 84])
        size of out cls is: torch.Size([1, 84])
        size of prediction is: torch.Size([1, 1])
        '''
        out_seq = transformer_dict['sequence']
        out_cls = transformer_dict['cls']
        prediction = self.regression_head(out_cls)
        return {'reconstructed_fmri_sequence': out_seq, 'embedding_per_ROIs': out_cls, self.task:prediction}