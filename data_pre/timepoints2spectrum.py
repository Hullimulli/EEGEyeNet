import numpy as np
import os
from scipy import signal
from mne.time_frequency import psd_array_multitaper
import math
from tqdm import tqdm


def get_boundary_index(arr, min_value, max_value):
    len_ = len(arr)
    min_index, max_index = 0, len_ - 1
    for i in range(len_):
        if arr[i] < min_value:
            min_index += 1
        else:
            continue
    for i in range(len_):
        if arr[len_ - 1 - i] > max_value:
            max_index -= 1
        else:
            continue

    return min_index, max_index


def timepoints2psd(input_data,sf=500,method = 'welch',electrode_chosen='all',max_allow_value=None):
    num_samples = input_data.shape[0]
    len_timepoints = input_data.shape[1]
    num_electrodes = input_data.shape[2]


    new_data = []
    for i in tqdm(range(num_samples)):
        ele_data = []
        for j in range(num_electrodes):
            eeg_signal = input_data[i,:,j]
            if method == 'multitaper':
                psd, freqs = psd_array_multitaper(eeg_signal, sf, adaptive=True,
                                                  normalization='full', verbose=0)
                psd = psd[:50]
            elif method == 'welch':
                freqs, psd = signal.welch(eeg_signal, sf, nperseg=sf)
                psd = psd[:50]

            elif method == 'stft':
                f, t, Zxx = signal.stft(eeg_signal, sf, nperseg=64, noverlap=64-14)
                fmin,fmax = 0, 50
                fmin_index, fmax_index = get_boundary_index(f, fmin, fmax)
                # psd = np.abs(Zxx)[fmin_index:fmax_index,:].flatten()
                psd = np.abs(Zxx)[fmin_index:fmax_index, :]
                num_freq = len(f)
                num_timepoints = len(t)
            else:
                raise Exception("undefined method.")

            ele_data.append(psd)
        new_data.append(ele_data)

    new_data = np.array(new_data)
    if method == 'stft':
        # shape of new data: (num_samples,num_electrodes, num_freq, num_timepoints)
        if electrode_chosen == '3':
             electrode_index = [0, 16, 31]
        elif electrode_chosen == 'all':
            electrode_index = np.arange(0,129)
        else:
            raise Exception ('not defined electrode_chosen')

        array_list = []
        for ind in electrode_index:
            array_list.append(new_data[:,ind,:,:])
        # new_data = new_data[:,electrode_index,:,:]
        new_data = np.concatenate(array_list,axis = 1)
        # (num_samples,num_electrodes_selected * num_freq, num_timepoints_new)

    elif electrode_chosen == 'all' and method != 'stft':
        # shape of new data: (num_samples,num_electrodes, psd_len)
        new_data = new_data

    else:
        raise  Exception ('undefined situation.')

    new_data = np.transpose(new_data,(0,2,1))
    if max_allow_value != None:
        new_data = np.clip(new_data,a_min = 0,a_max = max_allow_value)
    return new_data



if __name__ == '__main__':
    data_folder = 'D:\\ninavv\master\\thesis\\data\\'
    # data_folder = '/cluster/work/hilliges/niweng/deepeye/data/'
    input_npz_file_name = 'Position_task_with_dots_synchronised_min_sample_0.1.npz'
    method = 'stft'
    electrode_chosen = '3'
    max_allow_value = 7.71
    output_npz_file_name = input_npz_file_name.split('.npz')[0]+'_'+method+'_'+electrode_chosen+'.npz'
    input_data = np.load(os.path.join(data_folder,input_npz_file_name))

    input_data_eeg = input_data['EEG']
    input_data_labels = input_data['labels']

    output_data_eeg = timepoints2psd(input_data_eeg,method=method,
                                     electrode_chosen = electrode_chosen,max_allow_value = max_allow_value)
    # output_data_eeg = np.concatenate((input_data_eeg,output_data_eeg),axis=1)
    # if electrode_chosen == '3':
    #     electrode_index = [0,16,31]
    #     output_data_eeg = output_data_eeg[:,:,electrode_index]
    # elif electrode_chosen == 'all':
    #     output_data_eeg = output_data_eeg
    # else:
    #     raise Exception('not defined electrode set.')

    np.savez(os.path.join(data_folder,output_npz_file_name),
             EEG = output_data_eeg,labels = input_data_labels) # labels remians the same
    print('Converting is done. File is stored in {}'.format(os.path.join(data_folder,output_npz_file_name)))