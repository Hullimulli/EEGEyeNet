import torch
import numpy as np


def zero_softmax(x,epsilon=0.1):
    ori_shape = x.shape
    x_flatten = torch.flatten(x,start_dim=1)
    x_new = torch.pow(torch.exp(x_flatten) - 1,2)
    sum_ = torch.sum(x_new,dim=1) + epsilon
    sum_ = torch.unsqueeze(sum_,dim=-1)
    x_new = x_new/sum_
    x_new = torch.reshape(x_new,ori_shape)


    return x_new


def normalized_adj_m(A):
    '''
    A : (bs, nb_node, nb_node), torch tensor
    '''
    assert A.shape[1] == A.shape[2]
    nb_node = A.shape[1]
    bs = A.shape[0]

    I = torch.eye(nb_node)
    if torch.cuda.is_available():
        I = I.cuda()
    A = A + I

    d = torch.sum(torch.abs(A), axis = -1)
    # print(f'd:{d}')
    d = 1 / torch.sqrt(d)
    D = torch.stack([torch.diag(d[i]) for i in range(bs)])
    # print(f'D:{D}')
    return D@A@D


if __name__ == '__main__':
    A = np.array([[0.5,0.8,-5],
                  [1,0,1],
                  [-0.5,0,0.4]])
    # A = np.abs(A)
    print(f'A:{A}')
    print(f'normalized A:{normalized_adj_m(A)}')