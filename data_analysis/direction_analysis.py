import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

def get_npz_data(data_dir,data_file_name):
    with np.load(data_dir + data_file_name, allow_pickle=True) as f:
        # X = f['EEG']
        y = f['labels']
    return None, y

if __name__ == '__main__':
    # load in data
    data_dir = 'D:/ninavv/master/thesis/data/'
    data_file_name = 'Direction_task_with_dots_synchronised_min_extended_labels.npz'
    # data_file_name = 'Direction_task_with_dots_min_prep_synch.npz'
    trainX, trainY = get_npz_data(data_dir, data_file_name)

    del trainX

    # angle distribution (min/max value, distribution, any clusters)
    y_angle = trainY[:, -2]
    y_angle = y_angle.astype(np.float32)
    print('--------ANGLE---------')
    min_,max_ = np.min(y_angle),np.max(y_angle)
    print(f'range: {min_:.4f} ~ {max_:.4f}')
    avg,std = np.mean(y_angle),np.std(y_angle)
    print(f'mean: {avg:.4f} \tstd: {std:.4f}')

    plt.figure(figsize=(10,8),dpi=300)
    plt.hist(y_angle, bins=100,color='steelblue')
    plt.title('Angle Distribution in Direction Task')
    plt.show()



    # amplitude distribution
    y_amp = trainY[:, -3]
    y_amp = y_amp.astype(np.float32)
    print('--------AMPLITUDE---------')
    min_, max_ = np.min(y_amp), np.max(y_amp)
    print(f'range: {min_:.4f} ~ {max_:.4f}')
    avg, std = np.mean(y_amp), np.std(y_amp)
    print(f'mean: {avg:.4f} \tstd: {std:.4f}')

    plt.figure(figsize=(10, 8), dpi=300)
    plt.hist(y_amp, bins=100, color='steelblue')
    plt.title('Amplitude Distribution in Direction Task')
    plt.show()

    # shift distribution

    y_row = trainY[:, 9:11].astype(np.float32)
    y = copy.deepcopy(y_row)
    y[:, 0] = np.cos(y_row[:, 1]) * y_row[:, 0]
    y[:, 1] = np.sin(y_row[:, 1]) * y_row[:, 0]

    fixation_1 = trainY[:, 5:7].astype(np.float32) # 5:7
    fixation_2 = trainY[:, 7:9].astype(np.float32) # 7:9
    y = fixation_2 - fixation_1

    print('--------SHIFT---------')
    euclidean_distance = np.linalg.norm(y, axis=1)
    min_, max_ = np.min(euclidean_distance), np.max(euclidean_distance)
    print(f'range of euclidean_distance: {min_:.4f} ~ {max_:.4f}')
    avg, std = np.mean(euclidean_distance), np.std(euclidean_distance)
    print(f'mean of euclidean_distance: {avg:.4f} \tstd of euclidean_distance: {std:.4f}')

    # plt.figure(figsize=(10, 8), dpi=300)
    # starting_point = [0,0]
    # for i in tqdm(range(len(y))):
    #     xs = [0, y[i, 0]]
    #     ys = [0, y[i, 1]]
    #     plt.plot(xs,ys,color='steelblue',alpha=0.5)
    # plt.title('Shift Distribution in Direction Task')
    # plt.show()

    print('--------SACCADE---------')
    y_saccade_1 = trainY[:, 1:3].astype(np.float32)
    y_saccade_2 = trainY[:, 3:5].astype(np.float32)

    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(y_saccade_1[:,0],y_saccade_1[:,1],color='limegreen', alpha=0.5)
    plt.xlim(0,800)
    plt.ylim(0,600)
    plt.title('Saccade-Start Distribution in Direction Task')
    plt.show()


    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(y_saccade_2[:,0],y_saccade_2[:,1], color='seagreen', alpha=0.5)
    plt.xlim(0, 800)
    plt.ylim(0, 600)
    plt.title('Saccade-End Distribution in Direction Task')
    plt.show()

    print('--------FIXATION---------')
    y_fixation_1 = trainY[:, 5:7].astype(np.float32)
    y_fixation_2 = trainY[:, 7:9].astype(np.float32)

    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(y_fixation_1[:, 0], y_fixation_1[:, 1], color='limegreen', alpha=0.5)
    plt.xlim(0, 800)
    plt.ylim(0, 600)
    plt.title('Fixation-Start Distribution in Direction Task')
    plt.show()

    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(y_fixation_2[:, 0], y_fixation_2[:, 1], color='seagreen', alpha=0.5)
    plt.xlim(0, 800)
    plt.ylim(0, 600)
    plt.title('Fixation-End Distribution in Direction Task')
    plt.show()