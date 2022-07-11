import numpy as np

from data_analysis.withinsubject_analysis import load_data
import os

if __name__ == '__main__':
    # load in data
    data_folder = 'D:\\ninavv\\master\\thesis\\data\\'
    # file_name = 'Position_task_with_dots_synchronised_min.npz'
    file_name = 'Position_task_with_dots_min_prep_synch.npz' #_minprep_synch
    file_path = os.path.join(data_folder, file_name)

    X_test, y_test = load_data(file_path=file_path)

    # # only examine interest subject
    # interested_subject_ids = [ 1.,  7., 11., 38., 44., 46., 52., 56.]
    #
    # record = []
    #
    # for each_id in interested_subject_ids:
    #     mask = (y_test[:,0] == each_id)
    #     eeg_signal_each = X_test[mask, :,:]
    #
    #     for e_ind,each_trail in enumerate(eeg_signal_each):
    #         if each_trail.std() >= 50:
    #             print('S{}T{}\t mean:{:.4f}\tstd:{:.4f}'.format(int(each_id),e_ind,each_trail.mean(),
    #                                                         each_trail.std()))
    #         record.append([each_id,e_ind,each_trail.mean(),each_trail.std()])

    boundary = 200
    record = []
    for trail_ind,each_trail in enumerate(X_test):
        max_ = each_trail.max()
        min_ = each_trail.min()
        if max_ > boundary or min_ < -(boundary):
            record.append(trail_ind)
            print('T{}\tmax:{:.2f}\tmin:{:.2f}\tstd:{:.2f}'.format(trail_ind,max_,min_,
                                                                  each_trail.std()))



    print('Total dropout trail: {}'.format(len(record)))
    record = np.array(record)
    np.savez('dropout_trail_index_'+str(boundary)+'.npz',record= record)
    print('DONE')