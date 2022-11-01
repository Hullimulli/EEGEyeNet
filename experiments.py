from Joels_Files.TwoDArchitecture.Train import method
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Process Experiment Parameters')
parser.add_argument('--seed', type=int, default=0, help='seed of the experiment')
parser.add_argument('--convDimension', type=int, default=1, help='type of convolution, shapes input features')
parser.add_argument('--name', type=str, default='CNN', help='architecture type')
parser.add_argument('--task', type=str, default='angle', help='task type')
parser.add_argument('--dataset', type=str, default='max', help='Preprocessing')
parser.add_argument('--electrodeConf', type=str, default='All', help='electrode configuration')
args = parser.parse_args()
if args.electrodeConf == "Top2":
    electrodes = np.array([125,128])
elif args.electrodeConf == "Top3":
    electrodes = np.array([17,125,128])
elif args.electrodeConf == "Top8":
    electrodes = np.array([1, 17, 32, 38, 121, 125, 128, 129])
elif args.electrodeConf == "SideFronts":
    electrodes = np.array([1,4, 5,6,8,12,14,17, 19,21,25,32, 33,38, 43,120,121, 122,125, 126,127,128, 129])
elif args.electrodeConf == "All":
    electrodes = np.arange(1,130)
elif args.electrodeConf == "TopBack":
    electrodes = np.array([59, 60, 65, 66, 70, 83, 84, 85, 90, 91, 129])
elif args.electrodeConf == "Back":
    electrodes = np.array([50, 58, 59, 60, 65, 66, 70, 83, 84, 85, 90, 91, 96, 101, 129])
elif args.electrodeConf == "TopFront":
    electrodes = np.array([1, 8, 14, 17, 21, 25, 32, 38, 43, 120, 121, 125, 126, 127, 128])
elif args.electrodeConf == "Front":
    electrodes = np.array([1, 2, 8, 9, 14, 17, 21, 22, 25, 26, 27, 32, 33, 38, 39, 43, 115, 120, 121, 122, 123, 125, 126, 127, 128])
elif args.electrodeConf == "FrontAndBack":
    electrodes = np.array([1, 8, 14, 17, 21, 25, 32, 38, 43, 59, 60, 65, 66, 70, 83, 84, 85, 90, 91, 120, 121, 125, 126, 127, 128, 129])
elif args.electrodeConf == "Top40":
    electrodes = np.array([1, 2, 8, 9, 14, 17, 21, 22, 25, 26, 27, 32, 33, 38, 39, 43, 50, 58, 59, 60, 65, 66, 70, 83, 84, 85, 90, 91, 96,101, 115, 120, 121, 122, 123, 125, 126, 127, 128, 129])

######################## Your Parameters ##########################
modelDirectory = './MultiDNet' # Here you will save your model

######################## How to run #####################
# use “python experiments.py --name CNN --seed 1“
# for name you can use CNN, EEGNet, PyramidalCNN, Xception and InceptionTime
# Seed is for the random initialization. I recommend to use a different integer for each run
# When you run experiments.py, only one network is run and the score is calculated. Therefore, you must
# run each model separately, but this allows for parallelization :)



task = method(task=args.task, directory=modelDirectory, electrodes = electrodes, batchSize=64,
              wandbProject='Neurips_Sept2022',name=args.name, seed=args.seed, convDimension=args.convDimension,
              dataPostFix=args.dataset, memoryEfficientBool=False)
# task = method(name='PyramidalCNN',task='lr',dataPostFix="min",directory='/Users/Hullimulli/Documents/ETH/SA2/localRuns', seed=args.seed,
#               convDimension=1, wandbProject='',batchSize=64, memoryEfficientBool=True)
task.fit(nrOfEpochs=50, saveBool=True, loadNpzBool=True)