import numpy as np
import os
import matplotlib.pyplot as plt




def plot_psd_electrodes(data,title='',vmin=0,vmax=3):
    plt.figure(figsize=(6,6))
    print(data.shape)
    print('mean and std: {:.4f}+-{:.4f}'.format(np.mean(data),np.std(data)))
    print('min and max: {}, {}'.format(np.min(data),np.max(data)))
    plt.pcolormesh(np.arange(data.shape[0]), np.arange(data.shape[1]),np.transpose(data),vmin=vmin,vmax=vmax) #,shading='gouraud')
    plt.xlabel('Time')
    plt.ylabel('frequency stacked by electrodes')
    plt.yticks([])
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    data_folder = 'D:\\ninavv\master\\thesis\\data\\'
    # data_folder = '/cluster/work/hilliges/niweng/deepeye/data/'
    input_npz_file_name = 'Position_task_with_dots_synchronised_min_sample_0.1.npz'  #_stft_3.npz'

    input_data = np.load(os.path.join(data_folder, input_npz_file_name))

    input_data_eeg = input_data['EEG']

    print('-' * 20)
    print('ALL DATA')
    print('mean and std: {:.4f}+-{:.4f}'.format(np.mean(input_data_eeg), np.std(input_data_eeg)))
    print('min and max: {}, {}'.format(np.min(input_data_eeg), np.max(input_data_eeg)))
    print('quantile 25% 50% and 75%: {}, {}, {}'.format(np.quantile(input_data_eeg,0.25),
                                                        np.quantile(input_data_eeg,0.5),
                                                        np.quantile(input_data_eeg,0.75)))
    print('-'*20)
    vmin = np.quantile(input_data_eeg,0.1)
    vmax = np.quantile(input_data_eeg,0.9)
    print('quantile 10% and 90%: {:.4f}, {:.4f}'.format(vmin,vmax))

    sample_index = 365

    one_sample = input_data_eeg[sample_index,:,:]

    plot_psd_electrodes(one_sample,title='Sample index: {}'.format(sample_index),vmin=vmin,vmax=vmax)

    # one_ele = one_sample[:,0]
    # plt.plot(np.arange(len(one_ele)),one_ele)
    # plt.show()
