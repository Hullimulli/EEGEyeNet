import math
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import logging
from DL_Models.torch_models.CNN.CNNMultiTask import SEBlock,SelfAttentionBlock,CNN1DTBlock
from DL_Models.torch_models.GCN.utils import zero_softmax,normalized_adj_m

import torch.nn.functional as F
import scipy.io as sio
import numpy as np
from config import config
import matplotlib.pyplot as plt
import random

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


    def forward(self,x,in_ = None):
        # if in_ is not None:
        #     x = self.clean_x(x,in_)
        out = self.flatten(x)
        out = self.dropout_1(out)
        out = self.fc_1(out)
        out = self.rule(out)
        out = self.dropout_2(out)
        out = self.fc_2(out)
        return out

    def clean_x(self,x,in_):
        # remove the electrodes that we do not want
        if isinstance(in_, list):
            bs = len(in_)
            for i in range(bs):
                x[i,in_[i],:] = 0
            return x

        elif len(in_.shape) == 2:  # in_: (bs, number_dropout_elec)
            bs, k = in_.shape[0],in_.shape[1]
            for i in range(bs):
                x[i,in_[i],:] = 0
            return x

        else:
            raise Exception('Not implemented with in_ shape {} in Prediction layer'.format(in_.shape))
            return None


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, init=True ,init_domain=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init = init
        self.init_domain = init_domain
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        node_n = 129
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        # self.att = Parameter(self.get_original_adj())
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.init:
            if self.init_domain:# initialize with domain knowledge
                self.att = Parameter(self.get_original_adj())
            else:
                self.att.data.uniform_(-stdv, stdv)
        else:
            self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.matmul(input, self.weight)
        if self.init:
            dropout_elec = adj # naming issue!!
            if dropout_elec is not None:
                mask = self.make_mask(dropout_elec) #(bs)
                bs = len(dropout_elec)
                att = torch.stack([self.att for _ in range(bs)]) # (bs,node_n,node_n)
                att = att.mul(mask) # # (bs,node_n,node_n)
                output = torch.bmm(att, support)
            else:
                att = self.att
                output = torch.matmul(att, support)


            # output = torch.matmul(self.att, support)
        else:
            # output = torch.matmul(self.att, support)
            output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def make_mask(self,droupout_elec):
        nb_electrodes=129
        bs = len(droupout_elec)
        mask = torch.ones(bs,nb_electrodes,nb_electrodes)
        for i in range(bs):
            mask[i,droupout_elec[i],:] = 0.0
            mask[i, :,droupout_elec[i]] = 0.0
            # mask[i] = mask[i].fill_diagonal_(1.0)
        mask = mask.float()

        if torch.cuda.is_available():
            mask = mask.cuda()
        return mask

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    def get_original_adj(self):
        # adj = np.eye(N = self.nb_electrodes)
        if torch.cuda.is_available():
            dir = config['data_dir'] + 'elec_pos/'
            # dir = '/cluster/work/hilliges/niweng/deepeye/data/elec_pos/'
        else:
            dir = 'D:\\ninavv\\master\\thesis\\data\\elec_pos\\'
        electrodePositions = sio.loadmat(dir + "lay129_head.mat")['lay129_head']['pos'][0][0][3:132]

        from scipy.spatial import distance_matrix
        dis_matrix = distance_matrix(x=electrodePositions,y=electrodePositions)
        longest_distance = np.max(dis_matrix)

        threshold_1 = 0.125 * longest_distance
        # threshold_2 = 0.875 * longest_distance

        adj = torch.tensor((dis_matrix < threshold_1)*1).float()
        # adj_2 = torch.tensor((dis_matrix > threshold_2)*1).float()
        # adj = adj - adj_2

        # mean_ = np.mean(dis_matrix)
        # adj = torch.tensor(mean_-dis_matrix).float()
        if torch.cuda.is_available():
            adj = adj.cuda()

        return adj



class GC_Block(nn.Module):
    def __init__(self, in_features, out_features, init=True, init_domain = True,p_dropout = 0.5, bias=True, node_n = 129):
        """Define a residual block of GCN."""
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init = init
        self.init_domain = init_domain

        self.gc1 = GraphConvolution(in_features, out_features, bias=bias, init=self.init,init_domain=self.init_domain)
        self.gc2 = GraphConvolution(out_features, out_features, bias=bias, init=self.init,init_domain=self.init_domain)

        self.bn1 = nn.BatchNorm1d(node_n * out_features)
        self.bn2 = nn.BatchNorm1d(node_n * out_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x, adj):
        """Forward step of GC Block."""
        y = self.gc1(x, adj)
        b, n, f = y.shape
        if b != 1:
            y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y, adj)
        b, n, f = y.shape
        if b != 1:
            y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        if self.in_features == self.out_features:
            return y + x
        else: return y


class AttentionBlock(nn.Module):
    def __init__(self,in_features,nb_features,depth,in_channels,kernel_size,isPreConv):
        super(AttentionBlock, self).__init__()
        self.in_features = in_features
        self.nb_features = nb_features
        self.depth = depth
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.isPreConv = isPreConv

        self.convs = nn.ModuleList([CNN1DTBlock(in_channels=self.in_features if i == 0 else self.nb_features,
                                               out_channels=self.nb_features,
                                                kernel_size=self.kernel_size) for i in range(self.depth)])

        self.seb = SEBlock(nb_channels=self.in_channels,nb_features=self.nb_features)

        if self.isPreConv:
            self.sab = SelfAttentionBlock(nb_features=self.nb_features)
        else:
            self.sab = SelfAttentionBlock(nb_features=self.in_features)

    def forward(self,x):
        if self.isPreConv:
            for i in range(self.depth):
                x = self.convs[i](x)
        # x, scale = self.seb(x)
        x, scale = self.sab(x)
        scale = torch.sigmoid(scale)
        return scale # (bs, nb_channels)



class GCNAdj(nn.Module):
    def __init__(self, input_shape,output_shape,path,
                 saveModel_suffix='',manipulate_parak=None,threshold=500,init=True,init_domain=True):
        super(GCNAdj, self).__init__()

        self.input_shape = input_shape # (n_electrodes,n_timepoints)
        self.output_shape = output_shape
        self.nb_electrodes = self.input_shape[0]
        self.nb_timepoints = self.input_shape[1]

        self.nb_features = 64
        self.depth = 6
        self.kernel_size = 32

        self.depth_gc = 12
        self.adj = self.get_original_adj()

        self.saveModel_suffix = saveModel_suffix
        self.path = path

        self.manipulate_parak = manipulate_parak
        self.threshold = threshold
        self.init = init
        self.init_domain = init_domain
        self.threshold_dropelec = 0.5

        self.using_rank = False
        if self.manipulate_parak is None:
            self.using_rank = True

        self.use_pos_embedding = False
        self.depth_pos_emb = 6
        self.isPreConv = True

        self.prepred_scale = False


        # logging
        logging.info(f"-----PARAMETERS FOR GCNAdj-----")
        logging.info(f"\tSave Model Suffix      : {self.saveModel_suffix}")
        logging.info(f"\tInput Shape            : {self.input_shape}")
        logging.info(f"\tOutput Shape           : {self.output_shape}")

        logging.info(f"\tManipulate K           : {self.manipulate_parak}")

        logging.info(f"\tNumber of features for Conv Layer  : {self.nb_features}")
        logging.info(f"\tDepth for Conv Layer               : {self.depth}")
        logging.info(f"\tKernel Size for Conv Layer         : {self.kernel_size}")
        logging.info(f"\tDepth for GC Blocks    : {self.depth_gc}")

        logging.info(f"\tOriginal Adjacency Matrix          : {self.adj}")
        logging.info(f"\tUsing Rank Method                  : {self.using_rank}")
        logging.info(f"\tManipulate Parameter k             : {self.manipulate_parak}")
        logging.info(f"\tThreshold              : {self.threshold}")
        logging.info(f"\tInitialize the Adjacency Matrix    : {self.init}")
        logging.info(f"\tInitialize with Domain Knowledge (random init otherwise)   : {self.init_domain}")
        logging.info(f"\tThreshold for drop electrodes      : {self.threshold_dropelec}")

        logging.info(f"\tUsing Positional Embedding         : {self.use_pos_embedding}")
        logging.info(f"\tDepth for Positional Embedding     : {self.depth_pos_emb}")

        logging.info(f"\tUsing Pre-Conv Block               : {self.isPreConv}")

        logging.info(f"\tUsing Scale before Pred. Layer     : {self.prepred_scale}")

        # end of logging
        self.pos_encoding = self.positional_encoding()
        self.linear_before_pos_encoding = nn.Linear(1, self.depth_pos_emb)

        dct_matrices,_ = self.get_dct_matrix(self.nb_timepoints)
        if torch.cuda.is_available():
            self.dct_matrix = torch.from_numpy(dct_matrices).type(torch.float32).cuda()
        else:
            self.dct_matrix = torch.from_numpy(dct_matrices).type(torch.float32)

        self.attention_block = AttentionBlock(in_features=self.nb_timepoints*self.depth_pos_emb if self.use_pos_embedding else self.nb_timepoints,
                                              nb_features=self.nb_features,
                                              depth=self.depth,
                                              in_channels=self.nb_electrodes,
                                              kernel_size = self.kernel_size,
                                              isPreConv = self.isPreConv)

        self.gcbs = nn.ModuleList([GC_Block(in_features=self.nb_timepoints if i == 0 else self.nb_features,
                                            out_features=self.nb_features,init=self.init,init_domain=self.init_domain) for i in range(self.depth_gc)])

        self.prediction = PredictionLayer(input_shape=(self.nb_electrodes,self.nb_features),
                                          output_shape=self.output_shape,
                                          hiden_layers_depth = self.nb_features)


    def forward(self, x):
        # x = torch.matmul(torch.permute(x,(0,2,1)),self.dct_matrix[: self.nb_timepoints, :])
        # x = torch.permute(x,(0,2,1))

        if self.use_pos_embedding:
            bs = x.shape[0]
            out = torch.reshape(x,(bs,self.nb_timepoints,self.nb_electrodes,1))

            out = self.linear_before_pos_encoding(out)  # (batch_size, num_timepoints, num_electrodes ,depth)
            out = nn.functional.relu(out)  # ReLU

            # positional encoding
            out = torch.add(out, self.pos_encoding)
            out = nn.functional.relu(out)  # ReLU
            # (batch_size, num_timepoints, num_electrodes ,depth)
            out = torch.permute(out,(0,1,3,2))
            # (batch_size, num_timepoints, depth,num_electrodes)

            out = torch.reshape(out,(bs,-1,self.nb_electrodes))
        else:
            out = x

        # out shape: (bs, timepoints, num_electrodes)
        scale_for_nodes = self.attention_block(out)


        # get the updated adjacency matrix or the mask
        updated_adj, dropout_elec = self.adjust_adj(scale_for_nodes,k=self.manipulate_parak,threshold=self.threshold_dropelec) # (bs, nb_nodes, nb_nodes)
        # updated_adj = torch.tensor(updated_adj)
        if self.init:
            in_ = dropout_elec # (bs, k)
        else:
            in_ = updated_adj # (bs, node_n, node_n)

        x = torch.permute(x, (0, 2, 1))
        for i in range(self.depth_gc):
            x = self.gcbs[i](x,in_) # (bs, nb_nodes, nb_features)

        # x: (bs, nb_nodes, nb_features)
        if self.prepred_scale:
            x = torch.mul(x,torch.unsqueeze(scale_for_nodes,-1))


        if self.init:
            out = self.prediction(x,in_)
        else: out = self.prediction(x)

        reversed_sigmoid_scale = 1 - scale_for_nodes  # (bs,num_electrodes)
        # reversed, because we need to detect the ones has the lowest scale and throw them

        return out,scale_for_nodes

    def forward2(self, x, isPredict = False):
        bs = x.shape[0]


        # get the updated adj
        updated_adj = []
        for i in range(bs):
            # detect problematic electrode from x / random pick some others
            dropout_elecs = self.detect_outlier(x[i],k=self.manipulate_parak, isPredict=isPredict)

            adj = self.drop_tops(self.adj, dropout_elecs)  # (num_nodes,num_nodes)
            updated_adj.append(adj)
        updated_adj = torch.stack(updated_adj).float()
        if torch.cuda.is_available():
            updated_adj = updated_adj.cuda()


        x = torch.permute(x, (0, 2, 1))
        for i in range(self.depth_gc):
            x = self.gcbs[i](x,updated_adj) # (bs, nb_nodes, nb_features)

        # x = torch.flatten(x,start_dim=1)
        out = self.prediction(x)
        return out, updated_adj


    def detect_outlier(self, signal, k=None, isPredict=False):
        # signal shape: (500,129)
        if k is None:
            return None
        threshold = self.threshold

        max_abs_value = torch.max(torch.abs(signal), axis=0).values # (129,)
        above_threshold_elec = (max_abs_value > threshold).nonzero().detach().cpu().numpy().flatten()

        if isPredict:
            indexes = list(above_threshold_elec) # when predicting, only remove ourliers
        else:
            if len(above_threshold_elec) < k:
                random_select = random.sample(set(np.arange(0, 129, 1)) - set(above_threshold_elec),
                                              k - len(above_threshold_elec))
                indexes = list(random_select)+above_threshold_elec.tolist()
            else:
                indexes = list(above_threshold_elec[:k])

        indexes = torch.tensor(indexes).long()
        return indexes

    def adjust_adj(self,scales,k=8,threshold=None):
        '''
        scales: shape (bs,num_nodes)
        '''
        using_k = self.using_rank
        bs = scales.shape[0]

        if self.manipulate_parak is None:
            updated_adj = [self.adj for _ in range(bs)]
            updated_adj = torch.stack(updated_adj).float()
            if torch.cuda.is_available():
                updated_adj = updated_adj.cuda()
            return updated_adj, None

        if using_k:
            tops = torch.topk(torch.abs(scales), k=k,dim=1,largest=False,sorted=True) # (bs, k)

            updated_adj = []
            for i in range(bs):
                adj = self.drop_tops(self.adj,tops[1][i,:]) # (num_nodes,num_nodes)
                scale_ = torch.unsqueeze(scales[i],dim=0)
                adj = adj.clone() * scale_
                updated_adj.append(adj)
            updated_adj = torch.stack(updated_adj).float()
            if torch.cuda.is_available():
                updated_adj = updated_adj.cuda()
                # tops = tops.cuda()
            return updated_adj, tops[1]

        else:
            # using threshold
            if threshold is None:
                raise Exception('threshold is None')
            mask = scales < threshold
            drop_electrodes = []
            for i in range(bs):
                drop_electrodes.append((mask[i] == True).nonzero(as_tuple=True)[0])
            # drop_electrodes = torch.tensor(drop_electrodes).float()
            return None, drop_electrodes


    def drop_tops(self, adj, tops):
        '''
        adj: shape (nb_nodes, nb_nodes)
        tops: shape (k)
        '''
        # assert max(tops) < adj.shape[0]
        if tops is None:
            return adj

        adj[tops, :] = 0
        adj[:, tops] = 0
        # tmp = adj.detach().numpy()
        return adj

    def get_original_adj(self):
        # adj = np.eye(N = self.nb_electrodes)
        if torch.cuda.is_available():
            dir = config['data_dir'] + 'elec_pos/'
            # dir = '/cluster/work/hilliges/niweng/deepeye/data/elec_pos/'
        else:
            dir = 'D:\\ninavv\\master\\thesis\\data\\elec_pos\\'
        electrodePositions = sio.loadmat(dir + "lay129_head.mat")['lay129_head']['pos'][0][0][3:132]

        from scipy.spatial import distance_matrix
        dis_matrix = distance_matrix(x=electrodePositions,y=electrodePositions)
        longest_distance = np.max(dis_matrix)

        threshold_1 = 0.125 * longest_distance
        # threshold_2 = 0.875 * longest_distance
        # mean_ = np.mean(dis_matrix)

        adj_1 = torch.tensor((dis_matrix < threshold_1)*1)
        # adj_2 = torch.tensor((dis_matrix > threshold_2) * (-1))
        adj = adj_1 #+ adj_2
        # adj = torch.tensor(mean_ - dis_matrix)
        if torch.cuda.is_available():
            adj = adj.cuda()

        return adj

    def save(self):
        ckpt_dir = self.path + 'GCNAdj' + \
            '_nb_{}_{}'.format(0,self.saveModel_suffix) + '.pth'
        torch.save(self.state_dict(), ckpt_dir)
        logging.info(f"Saved new best model (on validation data) to ckpt_dir")

    def draw_adj(self):
        plt.figure(figsize=(8,8),dpi=100)
        # draw the position of electrodes first
        if torch.cuda.is_available():
            dir = config['data_dir'] + 'elec_pos/'
            # dir = '/cluster/work/hilliges/niweng/deepeye/data/elec_pos/'
        else:
            dir = 'D:\\ninavv\\master\\thesis\\data\\elec_pos\\'
        electrodePositions = sio.loadmat(dir + "lay129_head.mat")['lay129_head']['pos'][0][0][3:132]

        plt.scatter(electrodePositions[:, 0], electrodePositions[:, 1], s=150, c='white',edgecolors='black',linewidths=3,zorder=5)
        # for i in range(self.nb_electrodes):
        #     plt.text(electrodePositions[i, 0]+0.01, electrodePositions[i, 1]+0.01, s=str(i + 1),zorder=5)

        # selected_node = 40-1
        # plt.text(electrodePositions[selected_node, 0] + 0.01,
        #          electrodePositions[selected_node, 1] + 0.01, s=str(selected_node + 1), zorder=5)

        adj = self.adj.cpu().detach().numpy()
        vmin_, vmax_ = np.min(adj), np.max(adj)
        print('min and max: {}, {}'.format(vmin_, vmax_))
        import matplotlib
        import matplotlib.cm as cm
        norm = matplotlib.colors.Normalize(vmin=vmin_, vmax=vmax_, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
        for i in range(self.nb_electrodes):
            for j in range(self.nb_electrodes):
                if i < j:
                    plt.plot([electrodePositions[i, 0], electrodePositions[j, 0]],
                                          [electrodePositions[i,1],electrodePositions[j,1]],
                                          c=mapper.to_rgba(adj[i][j]),zorder=0,alpha=0.3)
                # if self.adj[i][j] == 1:
                #     # if i == selected_node:
                #     #     color_name = 'mediumvioletred'
                #     #
                #     # else:
                #     #     color_name = 'slateblue'
                #     # plt.scatter(electrodePositions[i,0],electrodePositions[i,1],c='purple')
                #     # plt.scatter(electrodePositions[j,0],electrodePositions[j,1], c='purple')
                #     plt.plot([electrodePositions[i,0],electrodePositions[j,0]],
                #              [electrodePositions[i,1],electrodePositions[j,1]],
                #              c='crimson',zorder=0)
                # if self.adj[i][j] == -1:
                #     plt.plot([electrodePositions[i,0],electrodePositions[j,0]],
                #              [electrodePositions[i,1],electrodePositions[j,1]],
                #              c='slateblue',zorder=0)


        # for i in range(self.nb_electrodes):
        #     for j in range(self.nb_electrodes):
        #         if i == selected_node and self.adj[i][j] == 1:
        #             print(i,j)
        #             plt.scatter(electrodePositions[j, 0], electrodePositions[j, 1], s=70, c='mediumvioletred',zorder=10)
        #             plt.plot([electrodePositions[i, 0], electrodePositions[j, 0]],
        #                      [electrodePositions[i, 1], electrodePositions[j, 1]],
        #                      c='mediumvioletred', zorder=5)
        # plt.scatter(electrodePositions[selected_node, 0], electrodePositions[selected_node, 1], s=70, c='darkred',zorder=10)

        # plt.title('Adjacency matrix in plot')
        plt.axis('off')
        plt.show()
        return None

    def get_dct_matrix(self,N):
        """Output n*n matrix of DCT (Discrete Cosinus Transform) coefficients."""
        dct_m = np.eye(N)
        for k in np.arange(N):
            for i in np.arange(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
        idct_m = np.linalg.inv(dct_m)
        return dct_m, idct_m

    def save_adj_matrix(self):
        adj_m = self.gcbs[-1].gc2.att.cpu().detach().numpy()
        ckpt_dir = self.path + 'adj_matrix.npz'
        np.savez(ckpt_dir,adj_m = adj_m)
        logging.info(f"Save the adjacency matrix.")


    def positional_encoding(self):
        '''
        calculate the positional encoding given the window length
        :return: positional encoding (1, window_length, 1, d_model)
        '''
        angle_rads = self.get_angles(np.arange(self.nb_timepoints)[:, np.newaxis], np.arange(self.depth_pos_emb)[np.newaxis, :])

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, :, np.newaxis, :]

        pos_encoding = torch.from_numpy(pos_encoding).float()

        if torch.cuda.is_available():
            pos_encoding = pos_encoding.cuda()

        return  pos_encoding # (1, seq_len, 1, d_model)

    def get_angles(self, pos, i):
        '''
        calculate the angles givin postion and i for the positional encoding formula
        :param pos: pos in the formula
        :param i: i in the formula
        :return: angle rad
        '''
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(self.depth_pos_emb))
        return pos * angle_rates


if __name__ == '__main__':
    # 'input_shape':input_shape,'output_shape':output_shape,'path':path,'manipulate_parak':manipulate_parak
    gcn = GCNAdj(input_shape=(129,500),output_shape=2,path='')
    # drop_electrodes = [1,39,45,108,115,121]
    # drop_electrodes_index = [each-1 for each in drop_electrodes]
    # gcn.adj = gcn.drop_tops(gcn.adj,drop_electrodes_index)
    gcn.draw_adj()