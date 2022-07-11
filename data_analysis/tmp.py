import numpy as np
from explainability.visualization.topoPlot import topoPlot
import os
import matplotlib
import matplotlib.cm as cm

from experiment import split,converting_ylabels,data_normalization
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_data(file_path):
    print(f"-----Load data from: {file_path}")
    np_data = np.load(file_path, allow_pickle=True)
    np_eeg = np_data['EEG']
    np_labels = np_data['labels']
    print(f"-----Load finish.")
    return np_eeg,np_labels



if __name__ == '__main__':

    dir = 'D:\\ninavv\\master\\thesis\\runs\\'
    experiment_id = '1655838572_Direction_task_dots_min_CNN_angle_SA_no2\\'
    #'1652344843_Position_task_dots_min_GCN_coeff0\\'
    # 1652347651_Position_task_dots_min_GCN_across\\
    # 1652346196_Position_task_dots_min_CNN_scross_sa\\
    npz_file = dir + experiment_id+ 'test_scale.npz'

    data = np.load(npz_file,allow_pickle=True)
    scale = data['scale']

    # scale = 1/(1 + np.exp(-scale))






    data_folder ='D:\\ninavv\\master\\thesis\\data\\'
    file_name = 'Direction_task_with_dots_min_prep_synch.npz'

    file_path = os.path.join(data_folder, file_name)
    trainX, trainY = load_data(file_path=file_path)
    batch_size = 32

    splitMethod = 'acrossSubject'  # 'acrossSubject'
    removeOutlier = 'no'
    threshold =500

    if splitMethod == 'acrossSubject':
        ids = trainY[:, 0]
        train, val, test = split(ids, 0.7, 0.15, 0.15, removeOutlier)
    elif splitMethod == 'withinSubject':
        dir = 'D:\\ninavv\\master\\thesis\\data\\'
        # dir = '/cluster/work/hilliges/niweng/deepeye/data/'
        if removeOutlier == 'yes':
            split_npz_filepath = os.path.join(dir, 'split/split_ind_' + str(threshold) + '.npz')
        else:
            split_npz_filepath = os.path.join(dir, 'split/split_ind_ori_' + str(threshold) + '.npz')

        npz_data = np.load(split_npz_filepath)
        train, val = npz_data['train'], npz_data['val']  # ,npz_data['test']

        split_npz_filepath = os.path.join(dir, 'split/split_ind_ori_' + str(threshold) + '.npz')
        npz_data = np.load(split_npz_filepath)
        test = npz_data['test']  # should not remove the data in test set

    isClassification = False
    if isClassification:
        trainY = converting_ylabels(trainY)

    # X_train, y_train = trainX[train], trainY[train]
    # X_val, y_val = trainX[val], trainY[val]
    # test[:] = False
    X_test, y_test = trainX[test], trainY[test]
    isNormalized = False
    if isNormalized:
        # X_train = data_normalization(X_train,data_name='X_train')
        # X_val = data_normalization(X_val,data_name='X_val')
        X_test = data_normalization(X_test, data_name='X_test')

    if splitMethod == 'withinSubject':
        selected_examples_id = [19, 81, 152, 619, 1034, 1148]
    elif splitMethod == 'acrossSubject':
        selected_examples_id = [53,424,641,1104]

    for each in selected_examples_id:
        scale_each = scale[each,:]
        vmin = np.min(scale_each)
        vmax = np.max(scale_each)
        topoPlot(scale_each,title='Example {}'.format(each),vmin=vmin,vmax=vmax)

    for each_id in selected_examples_id:
        plt.figure(figsize=(10,6),dpi=100)
        eeg_signal = X_test[each_id]
        this_scale = scale[each_id]

        minima = np.min(this_scale)
        maxima = np.max(this_scale)
        norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.OrRd) # plasma
        for i in range(129):
            plt.plot(np.arange(0,500,1),eeg_signal[:,i],color = mapper.to_rgba(this_scale[i]),alpha=0.5)
        plt.title('Subject {}'.format(each_id))
        plt.xlim(0,500)
        plt.show()