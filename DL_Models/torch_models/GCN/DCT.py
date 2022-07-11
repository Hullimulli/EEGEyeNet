import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.fft

def get_dct_matrix(N):
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

def load_data(file_path):
    print(f"-----Load data from: {file_path}")
    np_data = np.load(file_path, allow_pickle=True)
    np_eeg = np_data['EEG']
    np_labels = np_data['labels']
    print(f"-----Load finish.")
    return np_eeg,np_labels

def plot_eeg(eeg_data,title=''):
    # assert eeg_data.shape[0] == 500
    assert eeg_data.shape[1] == 129

    cmap = plt.get_cmap('copper')
    colors = cmap(np.linspace(0, 1, 129))

    plt.figure(figsize=(8,6))
    for i in range(129):
        plt.plot(eeg_data[:,i],linewidth=1,c=colors[i])
        # break
    plt.ylim(-200,200)
    plt.xlim(0,500)
    plt.title(f'EEG data, {title}')
    plt.show()


if __name__ == '__main__':
    # load in data:

    data_folder = 'D:\\ninavv\\master\\thesis\\data\\'
    file_name = 'Position_task_with_dots_min_prep_synch_testset.npz'
    file_path = os.path.join(data_folder, file_name)
    trainX, trainY = load_data(file_path=file_path)

    sample_ind = 14
    print('selected index: {}'.format(sample_ind))

    sample_eeg = trainX[sample_ind]
    sample_label = trainY[sample_ind]

    plot_eeg(sample_eeg)

    nb_timepoints = 500
    limited_coeff = 100
    dct_n = 100

    dct_matrix,idct_matrix = get_dct_matrix(nb_timepoints)

    plt.figure(figsize=(8,8),dpi=100)
    plt.imshow(dct_matrix)
    plt.show()


    dct_eeg = np.matmul(sample_eeg.T,dct_matrix[:nb_timepoints, :])[:,:limited_coeff]
    # dct_eeg = np.matmul(sample_eeg.T[:,:dct_n], dct_matrix[:dct_n, :])
    dct_eeg = dct_eeg.T
    plot_eeg(dct_eeg,title = 'dct')


    dct_eeg_2 = scipy.fft.dct(sample_eeg,axis=0,type=3,norm="ortho")
    plot_eeg(dct_eeg_2, title='dct_2')

    inverse_dct = np.matmul(dct_eeg.T,idct_matrix[:limited_coeff, :])
    inverse_dct = inverse_dct.T
    plot_eeg(inverse_dct, title=f'inverse_dct,limited_coeff={limited_coeff}')