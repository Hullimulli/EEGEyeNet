import numpy as np
import os
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # load in the data
    runs_folder = 'D:\\ninavv\\master\\thesis\\runs\\'
    experiment_id = '1650985396_Position_task_dots_min'
    pred_file_name = 'pred_y.npz'

    data = np.load(os.path.join(runs_folder,experiment_id,pred_file_name))
    pred_y = data['pred_y']
    gt_y = data['y']

    pred_y_final = np.argmax(pred_y,axis=1)
    cm = confusion_matrix(gt_y[:,1],pred_y_final)
    print(cm)

    plt.figure(figsize=(10,10),dpi=300)
    labels = ['1,19,26',2,3,4,5,6,7,8,9,10,11,12,13,14,
              15,16,17,18,20,21,22,23,24,25,27]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    disp.plot()
    plt.show()

    for i in range(len(cm)):
        pred_collection = cm[i,:]
        N = 3
        topN = pred_collection.argsort()[-N:][::-1]
        print('For True Label: {}\n\tPredicted Label Top {}:'.format(labels[i],N))
        for j in range(N):
            print('{}({:.2f}%),'.format(labels[topN[j]],
                                        (pred_collection[topN[j]]/np.sum(pred_collection))*100),end='')
        print('\n')

    # get the topN accuracy
    cnt = 0
    N = 3
    for i in range(len(pred_y)):
        each_pred_y = pred_y[i]
        topN = each_pred_y.argsort()[-N:][::-1]
        if gt_y[i,1] in topN:
            cnt+=1
    print('Top{} accuracy:{:.4f}'.format(N,cnt/len(gt_y)))

    id = 6000
    for id in range(0,1000):
        if gt_y[id,1] == 24:
            print('{}\tgt:{},pred:{}'.format(id,gt_y[id],np.argmax(pred_y[id])))

