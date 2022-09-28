from Joels_Files.TwoDArchitecture.Train import method
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Process Experiment Parameters')
parser.add_argument('--seed', type=int, default=0, help='seed of the experiment')
parser.add_argument('--convDimension', type=int, default=1, help='type of convolution, shapes input features')
parser.add_argument('--name', type=str, default='CNN', help='architecture type')
parser.add_argument('--task', type=str, default='angle', help='task type')
parser.add_argument('--dataset', type=str, default='min', help='Preprocessing')
parser.add_argument('--electrodeConf', type=str, default='All', help='electrode configuration')
args = parser.parse_args()
if args.electrodeConf == "Top2":
    electrodes = np.array([125,128])
elif args.electrodeConf == "Top3":
    electrodes = np.array([17,125,128])
elif args.electrodeConf == "All":
    electrodes = np.arange(1,130)
task = method(task=args.task, directory='./MultiDNet', electrodes = electrodes, batchSize=64, wandbProject='Neurips_Sept2022',name=args.name, seed=args.seed, convDimension=args.convDimension, dataPostFix=args.dataset)
#task = method(name='PyramidalCNN',task='lr',dataPostFix="max",directory='/Users/Hullimulli/Documents/ETH/SA2/localRuns', seed=args.seed, electrodes=np.arange(1,130),convDimension=1, wandbProject='',batchSize=64)
task.fit(nrOfEpochs=50, saveBool=True)