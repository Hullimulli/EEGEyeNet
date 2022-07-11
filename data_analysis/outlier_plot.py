import numpy as np
from data_analysis.withinsubject_analysis import load_data
import os
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    record = np.load('dropout_trail_index_500.npz')['record']
    record = np.load('../data_pre/split_ind.npz')['test']
    print('number of outlier trails: {}'.format(len(record)))

    data_folder = 'D:\\ninavv\\master\\thesis\\data\\'
    file_name = 'Position_task_with_dots_synchronised_min.npz'
    file_path = os.path.join(data_folder, file_name)

    X_test, y_test = load_data(file_path=file_path)

    y_test_drop = y_test[record]
    print(y_test_drop[:,1].mean())
    print(y_test_drop[:, 2].mean())

    plt.figure(figsize=(8,6))
    plt.scatter(y_test_drop[:,1],y_test_drop[:,2],alpha=0.5,color='coral')
    plt.xlim(0,800)
    plt.ylim(0,600)
    plt.show()

    y_test_drop = pd.DataFrame(y_test_drop,columns=['id','x','y'])
    y_test_drop_gb = y_test_drop.groupby(by='id')
    print(y_test_drop_gb.count())

    y_test = pd.DataFrame(y_test,columns=['id','x','y'])
    y_test_gb = y_test.groupby(by='id')
    print(y_test_gb.count())

    print(y_test_gb.groups.keys())

    for each_id in list(y_test_gb.groups.keys())[30:]:
        print(each_id)
        plt.figure(figsize=(8, 6))
        tmp = y_test_gb.get_group(each_id)['x']
        plt.scatter(y_test_gb.get_group(each_id)['x'], y_test_gb.get_group(each_id)['y'], alpha=0.5, color='coral')
        plt.xlim(0, 800)
        plt.ylim(0, 600)
        plt.title('S{}'.format(each_id))
        plt.show()

    plt.figure(figsize=(12, 6))
    labels = ['S{}'.format(int(each)) for each in y_test_gb.groups.keys()]
    X_axis = np.arange(len(y_test_drop_gb))
    plt.bar(X_axis - 0.2, y_test_drop_gb.count()['x'], 0.4, label='outlier count')
    plt.bar(X_axis + 0.2, y_test_gb.count()['x'], 0.4, label='total count')
    plt.xticks(X_axis, labels,rotation=90)

    plt.legend()

    plt.show()

