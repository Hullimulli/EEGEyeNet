from Joels_Files.TwoDArchitecture.Train import method
import argparse


parser = argparse.ArgumentParser(description='Process Experiment Parameters')
parser.add_argument('--seed', type=int, default=0, help='seed of the experiment')
parser.add_argument('--convDimension', type=int, default=1, help='type of convolution, shapes input features')
parser.add_argument('--name', type=str, default='CNN', help='architecture type')

args = parser.parse_args()

#task = method(task='angle', directory='./MultiDNet', batchSize=32, wandbProject='eegeye',name=args.name, seed=args.seed, convDimension=args.convDimension)
task = method(name='CNN_1D',directory='/Users/Hullimulli/Documents/ETH/SA2/localRuns', seed=args.seed, convDimension=1, wandbProject='eegeye',batchSize=32, task='angle')
task.fit(nrOfEpochs=1, saveBool=False)