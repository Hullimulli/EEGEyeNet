import numpy as np
import os
import matplotlib.pyplot as plt
from experiment import converting_ylabels


if __name__ == '__main__':
    # load in the data
    runs_folder = 'D:\\ninavv\\master\\thesis\\runs\\'
    experiment_id = '1657198211_Advanced_Direction_task_dots_min_CNN_SA_SHIFT'
    pred_file_name = 'pred_y.npz'

    data = np.load(os.path.join(runs_folder,experiment_id,pred_file_name))
    pred_y = data['pred_y']
    gt_y = data['y']

    # get a part of the data
    cla_y = converting_ylabels(gt_y)[:,1:]
    choose_dot = 9
    scoring_func = (lambda y, pred_y: np.linalg.norm(y - pred_y, axis=1).mean())  # euclidean_distance

    plt.figure(figsize=(8, 6))
    # plt.scatter(gt_y[:, 1], gt_y[:, 2], color='grey', alpha=0.4)
    plt.scatter(pred_y[:, 0], pred_y[:, 1], color='olive', alpha=0.4)

    plt.xlim(0, 800)
    plt.ylim(0, 600)
    plt.show()

    for choose_dot in range(25):
        indexes = np.where(cla_y==choose_dot)[0]

        if len(indexes) == 0:
            continue

        score = scoring_func(gt_y[indexes,1:],pred_y[indexes])

        plt.figure(figsize=(8,6))
        plt.scatter(gt_y[indexes, 1], gt_y[indexes, 2], color='grey', alpha=0.5)
        plt.scatter(pred_y[indexes,0],pred_y[indexes,1],color='olive',alpha=0.5)

        plt.xlim(0,800)
        plt.ylim(0,600)
        labels = ['1,19,26', 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                  15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 27]
        plt.title('True label as: {}\nscore: {:.4f}\nnumbver of trails:{}'.format(labels[choose_dot],score,len(indexes)))
        plt.show()

