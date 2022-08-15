from Joels_Files.TwoDArchitecture.Train import method
import argparse


parser = argparse.ArgumentParser(description='Process Experiment Parameters')
parser.add_argument('--seed', type=int, default=0, help='an integer for the accumulator')

args = parser.parse_args()

task = method(name='ResCNN_N', seed=args.seed, convDimension=0, directory='./MultiDNet', batchSize=32, wandbProject='eegeye')
#task = method(name='CNN_1D',directory='/Users/Hullimulli/Documents/ETH/SA2/localRuns', seed=args.seed,nrOfEpochs=1, convDimension=1,wandbProject='',batchSize=32)
task.fit(nrOfEpochs=50, saveBool=False)