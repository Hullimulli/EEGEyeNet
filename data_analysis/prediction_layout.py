import torch
from DL_Models.torch_models.CNN.CNNMultiTask import CNNMultiTask



if __name__ == '__main__':
    # load model
    input_shape = (129,500)
    output_shape = 2
    path = ''
    multitask = False
    use_SEB = False
    use_self_attention = True

    models = {
        'CNNMultiTask': [CNNMultiTask,
                         {'input_shape': input_shape, 'output_shape': output_shape, 'depth': 12, 'mode': '1DT',
                          'path': path, 'multitask': multitask,
                          'use_SEB': use_SEB, 'use_self_attention': use_self_attention}],
        # 'CNN': [CNN, {'model_name': 'CNN', 'path': path, 'loss':'mse', 'model_number':0, 'batch_size': 32, 'input_shape': input_shape,
        #              'output_shape' : 2, 'kernel_size': 64, 'epochs' : 3, 'nb_filters' : 16, 'use_residual' : True, 'depth' : 12}],
        # 'Autoencoder': [Autoencoder,{'input_shape':input_shape,'output_shape':output_shape,'n_rep':n_rep,'path':path,'using_bn': False,
        #                              'use_SEB':use_SEB}],
        # 'GCNAdj':[GCNAdj,{'input_shape':input_shape,'output_shape':output_shape,'path':path,
        #                   'manipulate_parak':manipulate_parak,'threshold':threshold,'init':GCN_init,
        #                   'init_domain':GCN_init_domain}]
    }


    model_name = models.keys()[0]
    model = models[model_name][0](**models[model_name][1])


    # load in dataset - test part


    # predict

    # plot