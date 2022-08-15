from Joels_Files.TwoDArchitecture.Train import method
import argparse


parser = argparse.ArgumentParser(description='Process Experiment Parameters')
parser.add_argument('--seed', type=int, default=0, help='seed of the experiment')
parser.add_argument('--convDimension', type=int, default=1, help='type of convolution, shapes input features')
parser.add_argument('--name', type=str, default='CNN', help='architecture type')

args = parser.parse_args()

task = method(name=args.name, seed=args.seed, convDimension=args.convDimension, directory='./MultiDNet', batchSize=32, wandbProject='eegeye')
#task = method(name='CNN_1D',directory='/Users/Hullimulli/Documents/ETH/SA2/localRuns', seed=args.seed,nrOfEpochs=1, convDimension=1,wandbProject='',batchSize=32)
task.fit(nrOfEpochs=50, saveBool=False)