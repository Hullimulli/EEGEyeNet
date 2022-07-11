import os
from data_analysis.withinsubject_analysis import load_data
import numpy as np

from sklearn.cluster import AgglomerativeClustering

if __name__ == '__main__':
    # load in data
    data_folder = 'D:\\ninavv\\master\\thesis\\data\\'
    # file_name = 'Position_task_with_dots_synchronised_min.npz'
    file_name = 'Position_task_with_dots_min_prep_synch.npz'  # _minprep_synch
    file_path = os.path.join(data_folder, file_name)

    X,y = load_data(file_path=file_path)

    # load the list of outliers
    threshold = 500
    outlier_npz_file = f'dropout_trail_index_{threshold}.npz'
    outlier_ind = np.load(outlier_npz_file)['record']

    X_outlier = X[outlier_ind]
    y_outlier = y[outlier_ind]

    np.savez(data_folder+f'Position_task_with_dots_min_prep_synch_outliers_thres={threshold}.npz',EEG=X_outlier,labels=y_outlier)

    nb_outlier = len(X_outlier)
    X_outlier = X_outlier.reshape(nb_outlier,-1)

    clustering = AgglomerativeClustering().fit(X_outlier)
    nb_cluster = len(np.unique(clustering))
    print(f'number of clusters: {nb_cluster}')

    for i in range(nb_cluster):
        print(f'\tCluster {i+1}:{np.sum(clustering==i)}')





