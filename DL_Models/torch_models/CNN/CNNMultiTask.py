import torch.nn as nn
import torch
import logging


class SelfAttentionBlock(nn.Module):
    def __init__(self,nb_features):
        super().__init__()

        self.nb_features = nb_features

        self.linear_k = nn.Linear(self.nb_features,self.nb_features)
        self.linear_v = nn.Linear(self.nb_features, self.nb_features)
        self.linear_q = nn.Linear(self.nb_features, self.nb_features)


    def forward(self,x): # shape of x: (batch_size, nb_features, nb_electrodes)
        x = torch.permute(x,(0,2,1)) # (batch_size, nb_electrodes, nb_features)

        k = self.linear_k(x)
        v = self.linear_v(x)
        q = self.linear_q(x)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask=None)
        out = scaled_attention
        out = torch.permute(out, (0, 2, 1))  #  (batch_size, nb_features, nb_electrodes)
        return out,attention_weights

    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask):
        # attn_dim: num_joints for spatial and seq_len for temporal
        '''
        The scaled dot product attention mechanism introduced in the Transformer
        :param q: the query vectors matrix (..., attn_dim, d_model/num_heads)
        :param k: the key vector matrix (..., attn_dim, d_model/num_heads)
        :param v: the value vector matrix (..., attn_dim, d_model/num_heads)
        :param mask: a mask for attention
        :return: the updated encoding and the attention weights matrix
        '''
        kt = k.transpose(-1, -2)
        matmul_qk = torch.matmul(q, kt)  # (..., num_heads, attn_dim, attn_dim)

        # scale matmul_qk
        dk = torch.tensor(k.shape[-1], dtype=torch.int32)  # tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # normalized on the last axis (seq_len_k) so that the scores add up to 1.
        attention_weights = nn.functional.softmax(scaled_attention_logits,
                                                  dim=-1)  # (..., num_heads, attn_dim, attn_dim)

        attention_weights = torch.sum(attention_weights,dim=1)
        # get it from both dimensions
        # attention_weights = torch.sum(attention_weights, dim=1)+ torch.sum(attention_weights, dim=2)
        attention_weights = torch.sigmoid(attention_weights)
        attention_weights = torch.unsqueeze(attention_weights,dim=-1)

        # output = torch.matmul(attention_weights, v)  # (..., num_heads, attn_dim, depth)
        output = v * attention_weights
        attention_weights = torch.squeeze(attention_weights,dim=-1)
        return output, attention_weights


class SEBlock(nn.Module):
    def __init__(self,nb_channels,nb_features,reduction_ratio=0.5):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.nb_features = nb_features
        self.nb_channels = nb_channels # number of electrodes in our cases
        self.middle_layer_param = int(self.nb_channels*self.reduction_ratio)

        self.global_pooling = nn.AvgPool1d(kernel_size=self.nb_features)
        self.fc_1 = nn.Linear(in_features=self.nb_channels,out_features=self.middle_layer_param)
        self.relu  = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=self.middle_layer_param,out_features=self.nb_channels)

    def forward(self,x):
        # shape of x: (batch_size, nb_features, nb_electrodes)
        permuted_x = torch.permute(x,(0,2,1)) # (batch_size, nb_electrodes, nb_features)
        out = self.global_pooling(permuted_x) # (batch_size, nb_electrodes, 1)
        out = torch.permute(out,(0,2,1)) # (batch_size, 1, nb_electrodes)

        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)

        scale = torch.sigmoid(out)  #(batch_size, 1, nb_electrodes)
        out = x * scale  # (batch_size, nb_features, nb_electrodes)

        scale = torch.squeeze(scale,dim=1) #(batch_size, nb_electrodes)

        return out, scale


class CNN2DBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=(32,32)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # 2d

        self.pool_kernel_size = 9
        self.pool_dilation = 1
        self.pool_padding = int((self.pool_kernel_size - 1) // 2)

        self.conv = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=self.kernel_size,
                                   padding='same')
        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.leakyrule = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=self.pool_kernel_size, dilation=self.pool_dilation,
                                    padding=self.pool_padding, stride=1)


    def forward(self,x):
        out = self.conv(x)
        out = self.leakyrule(out)
        out = self.bn(out)
        out = self.maxpool(out)

        return out

class ShortCut2D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=(1,1)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              padding='same')
        self.bn = nn.BatchNorm2d(num_features=self.out_channels)


    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        return out


class CNN1DTSBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=(32,16)):
        super().__init__()
        self.in_channels = in_channels # (timepoints, num_electrodes)
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # 1d

        self.t_kernel = self.kernel_size[0]
        self.s_kernel = self.kernel_size[1]

        self.pool_kernel_size = 9
        self.pool_dilation = 1
        self.pool_padding = int((self.pool_kernel_size - 1) // 2)

        self.conv_time = nn.Conv1d(in_channels=self.in_channels[0],
                              out_channels=self.out_channels,
                                   kernel_size=self.t_kernel,
                                   padding='same')
        self.conv_spatial = nn.Conv1d(in_channels=self.in_channels[1],
                                   out_channels=self.out_channels,
                                   kernel_size=self.s_kernel,
                                   padding='same')

        self.leakyrule = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=self.pool_kernel_size,
                                         dilation=self.pool_dilation,
                                         padding=self.pool_padding, stride=1)
        self.bn = nn.BatchNorm2d(1)




    def forward(self,x):
        # x : (batch_size, timepoints, num_electrodes)
        out = self.conv_time(x) # (batch_size,out_c, num_electrodes)
        out = torch.permute(out,(0,2,1)) # (batch_size, num_electrodes, out_c)
        out = self.conv_spatial(out)  # (batch_size, out_c, out_c)
        out = torch.permute(out, (0, 2, 1))

        out = self.leakyrule(out)

        out = torch.unsqueeze(out,dim=1) # (batch_size, 1, out_c, out_c)
        out = self.bn(out)
        out = self.maxpool(out) # (batch_size, 1, out_c, out_c)


        out = torch.squeeze(out, dim=1) # (batch_size, out_c, out_c)

        return out


class ShortCut1DTS(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1):
        super().__init__()
        self.in_channels = in_channels # (time,num_elec)
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv_time = nn.Conv1d(in_channels=self.in_channels[0],out_channels=self.out_channels,
                              kernel_size=self.kernel_size,padding='same')
        self.conv_spatial = nn.Conv1d(in_channels=self.in_channels[1], out_channels=self.out_channels,
                                   kernel_size=self.kernel_size, padding='same')
        self.bn = nn.BatchNorm2d(1)


    def forward(self,x):
        out = self.conv_time(x) # (batch_size,out_c, num_electrodes)
        out = torch.permute(out, (0, 2, 1))  # (batch_size, num_electrodes, out_c)
        out = self.conv_spatial(out)  # (batch_size, out_c, out_c)
        out = torch.permute(out, (0, 2, 1))
        out = torch.unsqueeze(out, dim=1)  # (batch_size, 1, out_c, out_c)
        out = self.bn(out)
        out = torch.squeeze(out, dim=1)  # (batch_size, out_c, out_c)
        return out


class CNN1DTBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size #1d

        # self.pool_kernel_size = self.kernel_size if self.kernel_size%2 == 1 else self.kernel_size-1
        self.pool_kernel_size = 9
        self.pool_dilation = 1
        self.pool_padding = int((self.pool_kernel_size-1) //2)

        self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels,
                              kernel_size=self.kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(num_features = self.out_channels)
        self.leakyrule = nn.LeakyReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=self.pool_kernel_size,dilation=self.pool_dilation,
                                    padding=self.pool_padding,stride=1)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.leakyrule(out)
        out = self.maxpool(out)

        return out

class ShortCut1DT(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(in_channels=self.in_channels,out_channels=self.out_channels,
                              kernel_size=self.kernel_size,padding='same')
        self.bn = nn.BatchNorm1d(num_features = self.out_channels)


    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        return out

class Classifier(nn.Module):
    def __init__(self,input_features,hidden_para,num_class):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=input_features,out_features=hidden_para)
        self.bn = nn.BatchNorm1d(num_features=hidden_para)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(in_features=hidden_para,out_features=num_class)
        self.softmax = nn.Softmax()

    def forward(self,x):
        out = self.linear_1(x)
        if out.shape[0] != 1:
            out = self.bn(out)
        out = self.relu(out)
        out = self.linear_2(out)
        # out = self.softmax(out)

        return out



class CNNMultiTask(nn.Module):
    def __init__(self,input_shape,output_shape,depth,mode,path,use_residual=True,
                 nb_features = 64 ,kernel_size=(64,64), saveModel_suffix='',multitask=False,
                 use_SEB = False, use_self_attention=False):
        super().__init__()


        self.input_shape = input_shape # (n_electrodes,n_timepoints)
        self.nb_electrodes = self.input_shape[0]
        self.nb_timepoints = self.input_shape[1]
        self.output_shape = output_shape

        self.depth = depth # number of ConvBlock
        if depth % 3 != 0:
            logging.info('!!! WARNING: depth is not a multiple of 3 !!!')
        self.nb_features = nb_features
        self.kernel_size = kernel_size
        self.use_residual = use_residual
        self.use_SEB = use_SEB
        self.use_self_attention = use_self_attention


        self.mode = mode # mode = '1DT', '2D' or '1DST'
        self.path = path
        self.saveModel_suffix = saveModel_suffix
        self.multitask = multitask




        logging.info(f"-----PARAMETERS FOR CNNMULTITASK-----")
        logging.info(f"\tMode                   : {self.mode}")
        logging.info(f"\tSave Model Suffix      : {self.saveModel_suffix}")
        logging.info(f"\tInput Shape            : {self.input_shape}")
        logging.info(f"\tOutput Shape           : {self.output_shape}")

        logging.info(f"\tDepth                  : {self.depth}")
        logging.info(f"\tKernel Size            : {self.kernel_size}")
        logging.info(f"\tNumber of Features (Middle layer channel)      : {self.nb_features}")
        logging.info(f"\tUse Residual           : {self.use_residual}")
        logging.info(f"\tMultitask              : {self.multitask}")
        logging.info(f"\tUse SE Block           : {self.use_SEB}")
        logging.info(f"\tUse Self Attention     : {self.use_self_attention}")


        if self.mode == '1DT':
            self.conv_blocks = nn.ModuleList([CNN1DTBlock(in_channels=self.nb_timepoints if i == 0 else self.nb_features,
                                                         out_channels=self.nb_features,
                                                         kernel_size=self.kernel_size[0]) for i in range(self.depth)])
            self.shortcuts = nn.ModuleList([ShortCut1DT(in_channels=self.nb_timepoints if i == 0 else self.nb_features,
                                                     out_channels=self.nb_features) for i in
                                            range(int(self.depth / 3))])
            self.output_layer = nn.Linear(in_features=self.nb_features * self.nb_electrodes,
                                          out_features=self.output_shape)
            self.classifier = Classifier(input_features=self.nb_features * self.nb_electrodes,
                                        hidden_para=128,num_class=72)
            if self.use_SEB:
                self.se_layer = SEBlock(nb_channels=self.nb_electrodes,nb_features=self.nb_features)
            if self.use_self_attention:
                self.self_attention_layer = SelfAttentionBlock(nb_features=self.nb_features)

            self.gap_layer = nn.AvgPool1d(kernel_size=2, stride=1)

        elif self.mode == '1DTS':
            self.conv_blocks = nn.ModuleList([CNN1DTSBlock(in_channels=(self.nb_timepoints,self.nb_electrodes) if i ==0 else (self.nb_features,self.nb_features),
                                                           out_channels=self.nb_features,
                                                           kernel_size=(32,16)) for i in range(self.depth)])
            self.shortcuts = nn.ModuleList([ShortCut1DTS(in_channels=(self.nb_timepoints,self.nb_electrodes) if i ==0 else (self.nb_features,self.nb_features),
                                                         out_channels=self.nb_features) for i in range(int(self.depth / 3))])
            self.output_layer = nn.Linear(in_features=self.nb_features * self.nb_features,
                                          out_features=self.output_shape)
            self.classifier = Classifier(input_features=self.nb_features * self.nb_features,
                                         hidden_para=128, num_class=72)
            if self.use_SEB:
                self.se_layer = SEBlock(nb_channels=self.nb_features,nb_features=self.nb_features)
            if self.use_self_attention:
                self.self_attention_layer = SelfAttentionBlock(nb_features=self.nb_features)


        elif self.mode == '2D':
            self.conv_blocks = nn.ModuleList([CNN2DBlock(
                in_channels=1 if i == 0 else self.nb_features,
                out_channels=self.nb_features,
                kernel_size=(32, 16)) for i in range(self.depth)])
            self.shortcuts = nn.ModuleList([ShortCut2D(
                in_channels=1 if i == 0 else self.nb_features,
                out_channels=self.nb_features) for i in range(int(self.depth / 3))])
            self.output_layer = nn.Linear(in_features=self.nb_features * self.nb_timepoints * self.nb_electrodes,
                                          out_features=self.output_shape)
        else:
            raise Exception('Not implemented.')



    def forward(self,x):
        # x : (batch_size, timepoints, num_electrodes)
        current_batch_size = x.shape[0]
        if self.mode == '2D':
            x = torch.unsqueeze(x,dim=1)
        # x = torch.permute(x,(0,2,1)) # x : (batch_size,num_electrodes,timepoints)
        input_res = x  # set for the residual shortcut connection
        # Stack the modules and residual connection
        shortcut_cnt = 0


        for d in range(self.depth):
            x = self.conv_blocks[d](x)
            if self.use_residual and d % 3 == 2:
                res = self.shortcuts[shortcut_cnt](input_res)
                shortcut_cnt += 1
                if self.use_SEB:
                    x, _  = self.se_layer(x)
                x = torch.add(x, res)
                x = nn.functional.relu(x)
                input_res = x
        # x: (batch_size, num_electrodes, nb_features)
        # x = self.gap_layer(x)
        if self.use_self_attention:
            x, _ = self.self_attention_layer(x)
        x = x.reshape(current_batch_size, -1)
        output = self.output_layer(x)  # Defined in BaseNet
        if self.multitask:
            id = self.classifier(x)
            return output, id
        return output

    def save(self):
        ckpt_dir = self.path + 'CNNMultiTask' + \
            '_nb_{}_{}'.format(0,self.saveModel_suffix) + '.pth'
        torch.save(self.state_dict(), ckpt_dir)
        logging.info(f"Saved new best model (on validation data) to ckpt_dir")

    def predict(self, x):
        # x : (batch_size, timepoints, num_electrodes)
        scale = None
        current_batch_size = x.shape[0]
        if self.mode == '2D':
            x = torch.unsqueeze(x, dim=1)
        # x = torch.permute(x,(0,2,1)) # x : (batch_size,num_electrodes,timepoints)
        input_res = x  # set for the residual shortcut connection
        # Stack the modules and residual connection
        shortcut_cnt = 0

        for d in range(self.depth):
            x = self.conv_blocks[d](x)
            if self.use_residual and d % 3 == 2:
                res = self.shortcuts[shortcut_cnt](input_res)
                shortcut_cnt += 1
                if self.use_SEB:
                    x, scale = self.se_layer(x)
                x = torch.add(x, res)
                x = nn.functional.relu(x)
                input_res = x
        # x: (batch_size, num_electrodes, nb_features)
        if self.use_self_attention:
            x, scale = self.self_attention_layer(x)
        x = x.reshape(current_batch_size, -1)
        output = self.output_layer(x)  # Defined in BaseNet
        if self.multitask:
            id = self.classifier(x)
            return output, id
        return output, scale


if __name__ == '__main__':
    model = CNNMultiTask(input_shape=(129,500),output_shape=2,depth=12,mode='1DT',path= '',use_residual=True,
                 nb_features = 64 ,kernel_size=(64,64), saveModel_suffix='',multitask=False,
                 use_SEB = False, use_self_attention=False)
    print(model)