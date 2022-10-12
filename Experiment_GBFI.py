import numpy as np
from Joels_Files.TwoDArchitecture.dataLoader import loadData, split
from Joels_Files.mathFunctions.electrode_math import gradientBasedFI
from Joels_Files.plotFunctions.electrode_plots import electrodeBarPlot,topoPlot, colourCode, electrodePlot
from config import config
from Joels_Files.plotFunctions.attention_visualisations import saliencyMap,plotSaliencyMap
from tensorflow import keras
############################
loadBool = False
task = 'lr'
dataPostFix = 'max'
colours = "Reds"
colour = "red"
datafilePath = "" # Path to npz file
############################################################# PFI ##########################################
with np.load(config['data_dir'] + datafilePath) as f:
    inputs = f[config['trainX_variable']]
    targets = f[config['trainY_variable']]

loss = 'mse'
if task == 'angle':
    loss = 'angle-loss'
elif task == 'lr':
    loss = "bce"


if task == 'amplitude':
    targetIndex = 1
    outputShape = 1
    outputActivation = "linear"
    taskSet = "direction"
elif task == 'angle':
    targetIndex = 2
    outputShape = 1
    outputActivation = "linear"
    taskSet = "direction"
elif task == "lr":
    targetIndex = 1
    outputShape = 1
    outputActivation = "sigmoid"
    taskSet = "lr"
elif task == "position":
    targetIndex = [1, 2]
    outputShape = 2
    outputActivation = "linear"
    taskSet = "position"
else:
    print("Not a valid task.")
    targetIndex = [1, 2]
    outputShape = 2
    outputActivation = "linear"
    taskSet = ""

if loss == 'angle-loss':
    lossName = loss
elif loss == "bce":
    lossName = 'binary_crossentropy'
else:
    lossName = 'mse'

# This commented part is for my specific way of saving the data, you may ignore it
# inputPath = config['data_dir'] + "{}_{}/".format(taskSet, dataPostFix) + 'X.npy'
# targetPath = config['data_dir'] + "{}_{}/".format(taskSet, dataPostFix) + 'Y.npy'
#
# inputs, targets = loadData(inputPath, targetPath)
trainIndices, valIndices, testIndices = split(targets[:, 0], 0.7, 0.15, 0.15)
targets = targets[:, targetIndex]

inputs = inputs[valIndices]
targets = targets[valIndices]

####### Adjust paths to models here as well as save locations
modelPaths = [
    'EEGNet_1D_{}_129_{}_1'.format(task, dataPostFix),
    'EEGNet_1D_{}_129_{}_2'.format(task, dataPostFix),
    'EEGNet_1D_{}_129_{}_3'.format(task, dataPostFix),
    'EEGNet_1D_{}_129_{}_4'.format(task, dataPostFix),
    'EEGNet_1D_{}_129_{}_5'.format(task, dataPostFix),
    'CNN_1D_{}_129_{}_1'.format(task,dataPostFix),
    'CNN_1D_{}_129_{}_2'.format(task,dataPostFix),
    'CNN_1D_{}_129_{}_3'.format(task,dataPostFix),
    'CNN_1D_{}_129_{}_4'.format(task,dataPostFix),
    'CNN_1D_{}_129_{}_5'.format(task,dataPostFix),
    'PyramidalCNN_1D_{}_129_{}_1'.format(task,dataPostFix),
    'PyramidalCNN_1D_{}_129_{}_2'.format(task, dataPostFix),
    'PyramidalCNN_1D_{}_129_{}_3'.format(task, dataPostFix),
    'PyramidalCNN_1D_{}_129_{}_4'.format(task, dataPostFix),
    'PyramidalCNN_1D_{}_129_{}_5'.format(task, dataPostFix),
    'Xception_1D_{}_129_{}_1'.format(task, dataPostFix),
    'Xception_1D_{}_129_{}_2'.format(task, dataPostFix),
    'Xception_1D_{}_129_{}_3'.format(task, dataPostFix),
    'Xception_1D_{}_129_{}_4'.format(task, dataPostFix),
    'Xception_1D_{}_129_{}_5'.format(task, dataPostFix),
    'InceptionTime_1D_{}_129_{}_1'.format(task, dataPostFix),
    'InceptionTime_1D_{}_129_{}_2'.format(task, dataPostFix),
    'InceptionTime_1D_{}_129_{}_3'.format(task, dataPostFix),
    'InceptionTime_1D_{}_129_{}_4'.format(task, dataPostFix),
    'InceptionTime_1D_{}_129_{}_5'.format(task, dataPostFix),

]

modelPaths = ["/Users/Hullimulli/Documents/ETH/SA2/00Neurips2022/MultiDNet/"+m+"/last" for m in modelPaths]
if loadBool:
    grads = np.genfromtxt("/Users/Hullimulli/Documents/ETH/SA2/00Neurips2022/Results/{}_{}_PFI.csv".format(task,dataPostFix), delimiter=',')[1:,1]
else:
    grads = gradientBasedFI(inputSignals=inputs,groundTruth=targets,modelPaths=modelPaths,loss = loss,
                    directory="/Users/Hullimulli/Documents/ETH/SA2/00Neurips2022/Results",
                    filename="{}_{}_PFI".format(task,dataPostFix))
electrodeBarPlot(grads,"/Users/Hullimulli/Documents/ETH/SA2/00Neurips2022/Results",yAxisLabel="Input Gradients",
                 filename="{}_{}_Bar".format(task,dataPostFix),colour=colour)
coloursEl = colourCode(grads,colourMap=colours)
electrodePlot(coloursEl,directory="/Users/Hullimulli/Documents/ETH/SA2/00Neurips2022/Results",filename="{}_{}_El".format(task,dataPostFix),alpha=1)
topoPlot(grads,directory="/Users/Hullimulli/Documents/ETH/SA2/00Neurips2022/Results",filename="{}_{}_Topo".format(task,dataPostFix),cmap=colours,
         valueType="Norm. Gradient Value")

############################################################# Saliency ##########################################
########## Again, adjust paths accordingly
#Specific Subject Sample
inputs = inputs[50:52]
targets = targets[50:52]

# For Single Sample, neat trick -> Second brackets
# inputs = inputs[[50]]
# targets = targets[[50]]
model = keras.models.load_model("PathToDesiredModel", compile=False)
electrodesToPlot = np.array([90])
savePath = ""

grads = saliencyMap(model,inputs,targets,loss,includeInputBool=False, absoluteValueBool=True)

plotSaliencyMap(inputs,targets,grads, directory=savePath,
                electrodesUsedForTraining=np.arange(1,130),
                electrodesToPlot=electrodesToPlot,saveBool=True)