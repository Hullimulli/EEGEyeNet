import numpy as np
from data_analysis.withinsubject_analysis import load_data
import os
import random

if __name__ == '__main__':
    random_seed = 2022

    threshold = 500
    removeoutlier = True

    dropout_ind = np.load('../data_analysis/dropout_trail_index_'+str(threshold)+'.npz')['record']
    # trails_ind_left = set(trails_ind_total) - set(dropout_ind)

    data_folder = 'D:\\ninavv\\master\\thesis\\data\\'
    # file_name = 'Position_task_with_dots_synchronised_min.npz'
    file_name = 'Position_task_with_dots_min_prep_synch.npz'
    file_path = os.path.join(data_folder, file_name)

    _, y = load_data(file_path=file_path)

    number_trails_total = len(y)
    number_of_subjects = len(np.unique(y[:,0]))
    subject_ids = np.arange(1, number_of_subjects + 1)
    trails_ind_total = np.arange(number_trails_total)

    train_ind_set, val_ind_set, test_ind_set = [],[],[]
    for each_id in subject_ids:
        indexes = np.where(y[:,0] == each_id)[0]
        indexes = set(indexes)
        if removeoutlier:
            indexes = set(indexes) - set(dropout_ind)
        num_total = len(indexes)
        print('Subject {}\ttotal trails(remove outliers): {}'.format(int(each_id),num_total))

        num_train = int(0.7*num_total)
        num_val = int(0.15*num_total)
        num_test = num_total - num_val - num_train
        print('\tnum_train:{}\tnum_val:{}\tnum_test:{}'.format(num_train,num_val,num_test))

        random.seed(random_seed)
        train_ind = random.sample(indexes, num_train)
        train_ind_set.extend(train_ind)
        indexes_left = indexes - set(train_ind)

        random.seed(random_seed)
        val_ind = random.sample(indexes_left, num_val)
        val_ind_set.extend(val_ind)
        indexes_left = set(indexes_left) - set(val_ind)

        test_ind_set.extend(indexes_left)

    print('num of trails in train: {}\tval: {}\ttest: {}'.format(len(train_ind_set),len(val_ind_set),len(test_ind_set)))

    if removeoutlier:
        np.savez('split_ind_' + str(threshold) + '.npz', train=np.array(train_ind_set),
                 val=np.array(val_ind_set),
                 test=np.array(test_ind_set))
    else:
        np.savez('split_ind_ori_' + str(threshold) + '.npz', train=np.array(train_ind_set),
                 val=np.array(val_ind_set),
                 test=np.array(test_ind_set))
    print('DONE')






