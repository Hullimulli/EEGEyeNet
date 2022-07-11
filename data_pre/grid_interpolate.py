import numpy as np
import mne
import os
from scipy.interpolate import griddata
from tqdm import tqdm

def convert2d(pos_array):
    assert pos_array.shape[1] == 3
    assert pos_array.shape[0] == 129

    ref_pos = pos_array[-1]

    assert ref_pos[0] == 0
    assert ref_pos[1] == 0

    center = (0,0)
    ref_height = ref_pos[-1]

    pos_array[:,2] -= ref_height
    for i in range(len(pos_array)-1):
        dis_1 = np.sqrt(np.sum(np.square(pos_array[i,:])))
        dis_2 = np.sqrt(np.sum(np.square(pos_array[i,:2])))
        ratio = dis_1/dis_2
        pos_array[i,0] *= ratio
        pos_array[i,1] *= ratio


    return pos_array

def visualize_single_case():
    poselec = mne.channels.make_standard_montage('GSN-HydroCel-129')
    tmp = poselec.get_positions()['ch_pos']
    pos_list = []
    for ele_name in tmp.keys():
        pos_list.append(tmp[ele_name])

    pos_array = np.array(pos_list)
    pos_array = convert2d(pos_array)
    pos_array = pos_array[:-1,:2]

    edge_scale = 1
    x_min,x_max = np.min(pos_array[:,0])*(edge_scale),np.max(pos_array[:,0])*(edge_scale)
    y_min,y_max = np.min(pos_array[:,1])*(edge_scale),np.max(pos_array[:,1])*(edge_scale)


    grid_x, grid_y = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

    # load EEG signals

    data_folder ='D:\\ninavv\\master\\thesis\\data\\'
    file_name = 'Position_task_with_dots_synchronised_min_testset.npz'

    np_data = np.load(os.path.join(data_folder,file_name),allow_pickle=True)
    np_eeg = np_data['EEG']
    np_labels = np_data['labels']
    print('load data from: {}.\nData size: {}.'.format(os.path.join(data_folder,file_name),np_eeg.shape))


    choose_ind = 4036
    choose_timep = 450
    np_eeg_case = np_eeg[choose_ind,choose_timep,:-1]

    print('np eeg case shape:{}'.format(np_eeg_case.shape))

    grid_z0 = griddata(points=pos_array, values=np_eeg_case, xi=(grid_x, grid_y), method='nearest')
    grid_z1 = griddata(points=pos_array, values=np_eeg_case, xi=(grid_x, grid_y), method='linear',fill_value=0.)
    grid_z2 = griddata(points=pos_array, values=np_eeg_case, xi=(grid_x, grid_y),  method='cubic')

    import matplotlib.pyplot as plt
    plt.subplot(221)
    # plt.imshow(np_eeg_case, extent=(0,1,0,1), origin='lower')
    plt.plot(pos_array[:,0], pos_array[:,1], 'k.', ms=1)
    plt.title('Original')
    plt.subplot(222)
    plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower',cmap=plt.cm.bwr)
    plt.title('Nearest')
    plt.subplot(223)
    shw1 = plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower',cmap=plt.cm.bwr)
    bar1 = plt.colorbar(shw1)
    bar1.set_label('ColorBar')
    plt.clim(-50,50)
    plt.title('Linear')
    plt.subplot(224)
    plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower',cmap=plt.cm.bwr)
    plt.title('Cubic')
    plt.gcf().set_size_inches(6, 6)
    plt.show()


def create_new_dataset(fromdatafile,todatafile):
    print(f"-----Load data from: {fromdatafile}")
    np_data = np.load(fromdatafile, allow_pickle=True)
    np_eeg = np_data['EEG']
    np_labels = np_data['labels']
    print(f"-----Load finish.")

    print('-----Start Converting...-----')
    # averaging from timepoint 200 - 300
    np_eeg = np_eeg[:,200:300,:]
    assert np_eeg.shape[1] == 100
    np_eeg = np.mean(np_eeg,axis=1)
    assert len(np_eeg.shape) == 2

    # get electrode position
    poselec = mne.channels.make_standard_montage('GSN-HydroCel-129')
    tmp = poselec.get_positions()['ch_pos']
    pos_list = []
    for ele_name in tmp.keys():
        pos_list.append(tmp[ele_name])

    pos_array = np.array(pos_list)
    pos_array = convert2d(pos_array)
    pos_array = pos_array[:-1, :2]

    edge_scale = 1.
    x_min, x_max = np.min(pos_array[:, 0]) * (edge_scale), np.max(pos_array[:, 0]) * (edge_scale)
    y_min, y_max = np.min(pos_array[:, 1]) * (edge_scale), np.max(pos_array[:, 1]) * (edge_scale)

    grid_x, grid_y = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

    grid_data = []

    for i in tqdm(range(len(np_eeg))):
        np_eeg_case = np_eeg[i,:-1]
        grid_z1 = griddata(points=pos_array, values=np_eeg_case, xi=(grid_x, grid_y), method='linear', fill_value=0.)
        grid_data.append(grid_z1)

        import matplotlib.pyplot as plt
        plt.subplot(121)
        # plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
        plt.plot(pos_array[:, 0], pos_array[:, 1], 'k.', ms=1)
        plt.title('Original')
        plt.subplot(122)
        shw1 = plt.imshow(grid_z1.T, extent=(0, 1, 0, 1), origin='lower', cmap=plt.cm.bwr)
        bar1 = plt.colorbar(shw1)
        bar1.set_label('ColorBar')
        plt.clim(-50, 50)
        plt.title('{}'.format(np_labels[i]))
        plt.show()

    grid_data = np.array(grid_data)
    print('-----Finish Converting.-----')
    np.savez(todatafile,EEG=grid_data,labels=np_labels)

if __name__ == '__main__':
    # data_folder =  'D:\\ninavv\\master\\thesis\\data\\'
    # #'/cluster/work/hilliges/niweng/deepeye/data/'
    # file_name = 'Position_task_with_dots_synchronised_min_sample_0.1.npz'
    # output_file_name = file_name.split('.npz')[0] + '_image.npz'
    #
    # fromdatafile = os.path.join(data_folder,file_name)
    # todatafile = os.path.join(data_folder,output_file_name)
    # create_new_dataset(fromdatafile=fromdatafile,todatafile=todatafile)
    visualize_single_case()