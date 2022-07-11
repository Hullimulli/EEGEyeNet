import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # load in the data
    npz_file = './outlier_electrodes.npz'
    outlier_elec = np.load(npz_file,allow_pickle=True)['record']

    # count number of wired electrodes in each trails
    cnt_ = [len(each) for each in outlier_elec]
    print(max(cnt_))

    plt.figure(figsize=(8,6))
    plt.hist(cnt_,bins=range(5,max(cnt_)))
    plt.show()

    # count each electrodes' show-up time
    cnt_2 = [0 for i in range(129)]
    for each in outlier_elec:
        for each_e in each:
            cnt_2[each_e] += 1

    plt.figure(figsize=(8, 6))
    plt.bar(range(0,129), cnt_2)
    plt.show()

    cnt_2 = np.array(cnt_2)
    print(np.argsort(cnt_2)[::-1])
    print(np.sort(cnt_2)[::-1])

    # get the index of 120+ electrodes with maximum value over 150
    print(np.where((np.array(cnt_)>20) & (np.array(cnt_)<30))[0])

