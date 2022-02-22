from Joels_Files.AnalEyeZor import AnalEyeZor
import numpy as np
import pandas as pd
#This script tries to give an intuition on how to use AnalEyeZor.py. This class is designed to be used directly
#with the existing EEGEyeNet Code. Note that few changes were done to the original code:
# -Benchmark does not overwrite the amplitude network with the angle network.
# -The default network load command is modified s.t. the model can only predict, no access to its loss function is needed.

#First, make sure to have the correct path in config['data_dir'].
#This script exclusively uses tensorflow -> config['framework'] = 'tensorflow'

def main():
    #Some Functions store their calculations. Set to True, if calculations have to be done.
    trainModels = True
    calculatePFI = True
    datasetPrediction = True
    pcaCalculation = True

    #Per dataset and electrode configuration, we create a new experiment. Each class instance belongs to a folder originally generated by benchmark.

    #We first train a model for the direction task.
    if trainModels:
        print("Training Models for 1st Task.")
        experimentOne = AnalEyeZor(task="Direction_task", dataset="dots", preprocessing="min", trainBool=True
                                   ,models=["PyramidalCNN"])
        experimentOne.moveModels(newFolderName="Sample_Direction_PyramidalCNN_PFI", originalPath=experimentOne.currentFolderPath)
    else:
        print("Loading Models for 1st Task.")
        experimentOne = AnalEyeZor(task="Direction_task", dataset="dots", preprocessing="min", trainBool=False
                                   ,models=["PyramidalCNN"],path="Sample_Direction_PyramidalCNN_PFI/")

    #PFI Analysis for the angle task. Plots are stored in the corresponding folder within runs.
    if calculatePFI:
        print("Calculating PFI.")
        experimentOne.PFI(saveTrail="_angle",nameSuffix="_PyramidalCNN_angle",iterations=1)

    print("Visualizing PFI.")
    lossValues = pd.read_csv(experimentOne.currentFolderPath + "PFI_PyramidalCNN_angle.csv", usecols=["PyramidalCNN"]).to_numpy()
    experimentOne.topoPlot(lossValues,cmap="Blues",filename="Sample_PFI_TopoPlot",epsilon=0.01)
    experimentOne.electrodeBarPlot(lossValues, colour="blue", filename="Sample_PFI_ElectrodeBarPlot")

    #We expect to find 3 regions of interest. We summerize them with 3 electrodes.
    print("Visualizing Top 3 Electrode Configuration.")
    electrodesOfInterest = np.array([1,17,32])
    electrodeRating = np.zeros(129)
    electrodeRating[electrodesOfInterest-1] = 50
    experimentOne.electrodePlot(colourValues=experimentOne.colourCode(values=electrodeRating, colourMap='Reds'),
                       filename='Sample_Top3_Configuration', alpha=0.4)



    #Next we train a model for the direction task with only 3 electrodes.
    del experimentOne
    if trainModels:
        print("Training Models for 2nd Task.")
        experimentTwo = AnalEyeZor(task="Direction_task", dataset="dots", preprocessing="min", trainBool=True
                                   ,models=["PyramidalCNN"],electrodes=np.array([1,17,32]))
        experimentTwo.moveModels(newFolderName="Sample_Direction_PyramidalCNN", originalPath=experimentTwo.currentFolderPath)
    else:
        print("Loading Models for 2nd Task.")
        experimentTwo = AnalEyeZor(task="Direction_task", dataset="dots", preprocessing="min", trainBool=False
                                   ,models=["PyramidalCNN"],electrodes=np.array([1,17,32]),path="Sample_Direction_PyramidalCNN/")

    #We do some prediction visualisation.
    print("Visualizing Prediction with the Direction Task Specific Function.")
    experimentTwo.visualizePredictionDirection(modelNames=["PyramidalCNN"], nrOfPoints=9, filename="Sample_Top3_PredictionVis")

    #We can predict the entire data set and select indices of interest. Here we want only left saccades.
    print("Plotting some Signals.")
    if datasetPrediction:
        experimentTwo.predictAll(postfix="Top3")
    lookLeftIndices = experimentTwo.findDataPoints(type="LeftOnly",model="PyramidalCNN",postfix="Top3")
    experimentTwo.plotSignal('PyramidalCNN', electrodes = np.array([32]), splitAngAmpBool=True, filename="Sample_Top3_LeftSaccadeAmpSignal", run=1,
                    specificDataIndices=lookLeftIndices, nrOfPoints=20000, nrOfLevels=6,
                    percentageThresh=3, maxValue=100)

    #Do the angular split up plot with pca modified data.
    if pcaCalculation:
        experimentTwo.pca()
    experimentTwo.plotSignal('PyramidalCNN', electrodes = np.array([1,17,32]), splitAngAmpBool=True, filename="Sample_Top3_Signals",
                    run=1, nrOfPoints=20000, nrOfLevels=8,
                    percentageThresh=3, maxValue=100,componentAnalysis="PCA",dimensions=10)
    if trainModels:
        #Here we train our custom simple regressor for the angle task.
        #This function does not save the network.
        print("Training the New Method for the Angle Task (and Amplitude Task).")
        experimentTwo.simpleDirectionRegressor("Support Vector Machine",plotBool=False)

    #Lastly, we have a look at the L&R task. Note we need here the dataset with the 129 electrodes plus the 3 eye tracker inputs
    #First eyetracker input is at index 129 and corresponds to the movement on the x axis.
    del experimentTwo
    if trainModels:
        print("Training Models for 3nd Task.")
        experimentThree = AnalEyeZor(task="LR_task", dataset="antisaccade", preprocessing="min", trainBool=True
                                   ,models=["PyramidalCNN"],electrodes=np.array([1,32]))
        experimentThree.moveModels(newFolderName="Sample_LR_PyramidalCNN", originalPath=experimentThree.currentFolderPath)
    else:
        print("Loading Models for 3nd Task.")
        experimentThree = AnalEyeZor(task="LR_task", dataset="antisaccade", preprocessing="min", trainBool=False
                                   ,models=["PyramidalCNN"],electrodes=np.array([1,32]),path="Sample_LR_PyramidalCNN/")

    print("Plotting Some Signals.")
    experimentThree.plotSignal('PyramidalCNN', electrodes = np.array([1,32]), filename="Sample_LR_AvgSig",
                    run=1, nrOfPoints=20000, meanBool=True, maxValue=100)

    #We want to analyze 5 misclassified signals
    if datasetPrediction:
        experimentThree.predictAll(postfix="LR")
    misclassIndices = experimentThree.findDataPoints(type="Missclassified", model="PyramidalCNN", postfix="LR", lossThresh=0.1)[:5]
    experimentThree.attentionVisualization("PyramidalCNN", filename="Sample_LR_ActVis",method="Saliency", run=1,
                                           dataIndices=misclassIndices)
    experimentThree.plotSignal('PyramidalCNN', electrodes = np.array([1,32]), filename="Sample_LR_Misclass",specificDataIndices=misclassIndices,
                    run=1, nrOfPoints=20000, plotMovementBool=False,plotSignalsSeperatelyBool=True, meanBool=False, maxValue=100)
    experimentThree.plotSignal('PyramidalCNN', electrodes = np.array([1,32]), filename="Sample_LR_Misclass",specificDataIndices=misclassIndices,
                    run=1, nrOfPoints=20000, plotMovementBool=True,plotSignalsSeperatelyBool=True, meanBool=False, maxValue=100)

    #We take an artifical signal and feed it to the network.
    print("Analyzing Behaviour for a noisy step function.")
    experimentThree.customSignal("Step", amplitude=30, turnPoint=100,postfix="_200ms",noiseStd=20)
    experimentThree.attentionVisualization("PyramidalCNN", filename="Sample_LR_ActVis_Step",method="Saliency", run=1,
                                           dataType="Step",postfix="_200ms",artificialTruths=np.array([0,1]))
    experimentThree.plotSignal('PyramidalCNN', electrodes = np.array([1,32]), filename="Sample_LR_Step",
                    run=1, nrOfPoints=20000, plotMovementBool=False, plotSignalsSeperatelyBool=True,meanBool=False, maxValue=100,
                    dataType="Step",postfix="_200ms")
if __name__ == '__main__':
     main()