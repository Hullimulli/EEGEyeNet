import torch
from torch import nn
import math
import logging
from DL_Models.torch_models.CNN.CNNMultiTask import CNN1DTBlock
from DL_Models.torch_models.CNN.CNNMultiTask import SEBlock,SelfAttentionBlock

class EncodingLayer(nn.Module):
    def __init__(self,n_rep,in_channels,out_channels,kernel_size=(3,3),pool_factor=(2,2),
                 dilation=(1,1),padding=(1,1),using_bn=False):
        super(EncodingLayer,self).__init__()

        self.n_rep=n_rep
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_factor = pool_factor
        self.padding = padding
        self.using_bn = using_bn
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv2ds_first = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                       kernel_size=self.kernel_size,stride =1,
                                       dilation = self.dilation,
                                       padding='same',padding_mode='zeros')
        self.conv2ds_others = nn.ModuleList([nn.Conv2d(in_channels=out_channels,out_channels=out_channels,
                                                       kernel_size=self.kernel_size,stride =1,
                                                       dilation=self.dilation,
                                                       padding='same',padding_mode='zeros') for _ in range(self.n_rep-1)])
        self.leakyrules = nn.ModuleList([nn.LeakyReLU() for _ in range(self.n_rep)])
        self.maxpool = nn.MaxPool2d(kernel_size=self.kernel_size,
                                    stride=self.pool_factor,
                                    dilation=self.dilation,
                                    padding= self.padding, #(int(self.pool_factor/2),int(self.pool_factor/2)),
                                    ceil_mode=False)
        self.batchnorm = nn.ModuleList([nn.BatchNorm2d(num_features=out_channels) for _ in range(self.n_rep)])

    def forward(self,x):
        for i in range(self.n_rep):
            if i == 0:
                x = self.conv2ds_first(x)
            else:
                x = self.conv2ds_others[i-1](x)

            if self.using_bn == True:
                x = self.batchnorm[i](x)

            x = self.leakyrules[i](x)

        out = self.maxpool(x)
        return out



class DecodingLayer(nn.Module):
    def __init__(self,n_rep,in_channels,out_channels,desired_outshape,kernel_size=(3,3),
                 pool_factor=(2,2),dilation=(1,1),padding=(1,1),using_bn=False,last_decoding_layer=False):
        super(DecodingLayer,self).__init__()

        self.n_rep = n_rep
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_factor = pool_factor
        self.desired_outshape = desired_outshape
        self.using_bn = using_bn
        self.last_decoding_layer = last_decoding_layer
        self.dilation = dilation


        self.kernel_size = kernel_size
        self.stride = self.pool_factor
        self.padding = padding

        self.convtranspose2ds_first = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                                         kernel_size=self.kernel_size,
                                                         stride=self.stride,
                                                         dilation=self.dilation,
                                                         padding=self.padding, padding_mode='zeros',)
        self.convtranspose2ds_others = nn.ModuleList(
            [nn.ConvTranspose2d(in_channels=out_channels,out_channels=out_channels,
                                kernel_size=self.kernel_size,
                                stride=(1,1),
                                dilation=1,
                                padding=self.padding, padding_mode='zeros',) for _ in range(n_rep - 1)])
        self.leakyrules = nn.ModuleList([nn.LeakyReLU() for _ in range(self.n_rep)])
        self.batchnorm = nn.ModuleList([nn.BatchNorm2d(num_features=out_channels) for _ in range(self.n_rep)])

    def forward(self,x):
        for i in range(self.n_rep):
            if i == 0:
                x = self.convtranspose2ds_first(x,output_size=self.desired_outshape)
            else:
                x = self.convtranspose2ds_others[i-1](x,output_size=self.desired_outshape)

            if self.using_bn == True:
                x = self.batchnorm[i](x)

            if self.last_decoding_layer==True and i == self.n_rep-1:
                x = x
            else:
                x = self.leakyrules[i](x)

        return x


class PredictionLayer(nn.Module):
    def __init__(self,input_shape,output_shape,hiden_layers_depth,dropout_rate=0.5):
        super(PredictionLayer,self).__init__()

        self.hiden_layers_depth= hiden_layers_depth
        self.output_shape= output_shape
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate


        self.flatten = nn.Flatten(start_dim=1) # leave the batchsize

        self.in_features = 1
        for i in range(len(input_shape)):
            self.in_features *= input_shape[i]

        self.dropout_1 = nn.Dropout(p=self.dropout_rate)
        self.fc_1 = nn.Linear(in_features=self.in_features,out_features=self.hiden_layers_depth)
        self.rule = nn.ReLU()
        self.dropout_2 = nn.Dropout(p=self.dropout_rate)
        self.fc_2 = nn.Linear(in_features=self.hiden_layers_depth,out_features=self.output_shape)


    def forward(self,x):
        out = self.flatten(x)
        out = self.dropout_1(out)
        out = self.fc_1(out)
        out = self.rule(out)
        out = self.dropout_2(out)
        out = self.fc_2(out)
        return out




class Autoencoder(nn.Module):
    def __init__(self,input_shape,output_shape,n_rep,path,using_bn = False,saveModel_suffix='',
                 use_self_attention=False,use_SEB=False):
        super().__init__()

        self.channel_parameters =[1,32,64,128,256,256]
        self.n_layers = len(self.channel_parameters)-1
        self.last_decoding_layer = [True if i==self.n_layers-1 else False for i in range(self.n_layers) ]
        self.kernel_size = (3,3)
        self.pool_factor = (2,2)
        self.dilation = (1,1) # (2,2)
        self.padding = ((self.kernel_size[0]-1)//2,(self.kernel_size[1]-1)//2)
        self.input_shape = input_shape # (n_electrodes,n_timepoints)
        self.output_shape = output_shape

        self.using_bn = using_bn
        self.path = path
        self.saveModel_suffix = saveModel_suffix

        self.hiden_layers_depth = 32
        self.n_rep = n_rep

        self.use_SEB = use_SEB
        self.nb_features = 500
        self.pre_conv_kernel = 32
        self.pre_conv_depth=2

        self.ori_shape = self.get_ori_shape()
        self.reverse_shape = self.ori_shape[::-1]
        self.input_shape_for_pl = (self.ori_shape[-1][0], self.ori_shape[-1][1], self.channel_parameters[-1])
        # self.input_shape_for_pl = (self.ori_shape[0][0], self.ori_shape[0][1], self.channel_parameters[0])

        logging.info(f"-----PARAMETERS FOR AUTOENCODER-----")
        logging.info(f"\tSave Model Suffix      : {self.saveModel_suffix}")
        logging.info(f"\tInput Shape            : {self.input_shape}")
        logging.info(f"\tOutput Shape           : {self.output_shape}")
        logging.info(f"\tChannel Parameters     : {self.channel_parameters}")

        logging.info(f"\tKernel Size            : {self.kernel_size}")
        logging.info(f"\tPool Factor            : {self.pool_factor}")
        logging.info(f"\tDilation               : {self.dilation}")
        logging.info(f"\tPadding                : {self.padding}")

        logging.info(f"\tUsing BN               : {self.using_bn}")
        logging.info(f"\tHidden Layer Depth     : {self.hiden_layers_depth}")
        logging.info(f"\tNumber of Conv. Layer in each Block    : {self.n_rep}")
        logging.info(f"\tInput Shape for Prediction Layer       : {self.input_shape_for_pl}")

        logging.info(f"\tUsing SE Block         : {self.use_SEB}")
        logging.info(f"\tNumber of Features     : {self.nb_features}")
        logging.info(f"\tKernel Size for Pre-conv Layer     : {self.pre_conv_kernel}")
        logging.info(f"\tDepth for Pre-conv Layer           : {self.pre_conv_depth}")



        self.encoder = nn.ModuleList([EncodingLayer(n_rep=self.n_rep,in_channels=self.channel_parameters[i],
                                                    out_channels=self.channel_parameters[i+1],
                                                    kernel_size=self.kernel_size,
                                                    pool_factor=self.pool_factor,
                                                    dilation=self.dilation,
                                                    padding=self.padding,
                                                    using_bn=self.using_bn) for i in range(self.n_layers)])


        self.decoder = nn.ModuleList([DecodingLayer(n_rep=self.n_rep,in_channels=self.channel_parameters[self.n_layers-i],
                                                    out_channels=self.channel_parameters[self.n_layers-i-1],
                                                    kernel_size=self.kernel_size,
                                                    pool_factor=self.pool_factor,
                                                    dilation=self.dilation,
                                                    padding=self.padding,
                                                    desired_outshape=self.reverse_shape[i+1],
                                                    using_bn=self.using_bn,
                                                    last_decoding_layer=self.last_decoding_layer[i]) for i in range(self.n_layers)])

        self.prediction_layer = PredictionLayer(input_shape=self.input_shape_for_pl,output_shape=self.output_shape,
                                                hiden_layers_depth=self.hiden_layers_depth)
        if self.use_SEB:
            self.pre_conv_block = nn.ModuleList([CNN1DTBlock(in_channels=self.input_shape[1] if i == 0 else self.nb_features,
                                                             out_channels=self.nb_features,
                                                             kernel_size=self.pre_conv_kernel) for i in range(self.pre_conv_depth)])
            self.se_layer = SEBlock(nb_channels=self.input_shape[0],nb_features=self.nb_features)


    def forward(self, x):
        if self.use_SEB:
            for i in range(self.pre_conv_depth):
                x = self.pre_conv_block[i](x)
            x, scale = self.se_layer(x)
        else:
            scale = None

        x = torch.unsqueeze(x, dim=-1)  # (batch_size, timepoints, num_electrodes, 1)
        x = torch.permute(x, (0, 3, 2, 1))  # (batch_size, 1, num_electrodes, timepoints)
        for i in range(self.n_layers):
            x = self.encoder[i](x)

        encoded = x

        for i in range(self.n_layers):
            x = self.decoder[i](x)
        decoded = x # (batch_size, 1, num_electrodes, timepoints)
        decoded = torch.squeeze(decoded, dim=1)
        decoded = torch.permute(decoded, (0, 2, 1))
        prediction = self.prediction_layer(encoded)
        # prediction = self.prediction_layer(decoded)
        return prediction, decoded, scale


    def get_ori_shape(self):
        ori_shape = []
        if self.use_SEB:
            current_shape = (self.input_shape[0],self.nb_features)
            ori_shape.append(current_shape)
        else:
            ori_shape.append(self.input_shape)
            current_shape = self.input_shape

        for i in range(self.n_layers):
            # only influence by MaxPool2d layer
            # H_out = [(H_in +2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1]
            # W_out = [(W_in +2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1]

            new_shape = (math.floor((current_shape[0]+2*self.padding[0] - self.dilation[0]*(self.kernel_size[0]-1)-1)/self.pool_factor[0] + 1),
                         math.floor((current_shape[1]+2*self.padding[1] - self.dilation[1]*(self.kernel_size[1]-1)-1)/self.pool_factor[1] + 1))
            ori_shape.append(new_shape)
            current_shape = new_shape
        return ori_shape


    def save(self):
        ckpt_dir = self.path + 'Autoencoder' + \
            '_nb_{}_{}'.format(0,self.saveModel_suffix) + '.pth'
        torch.save(self.state_dict(), ckpt_dir)
        logging.info(f"Saved new best model (on validation data) to ckpt_dir")


