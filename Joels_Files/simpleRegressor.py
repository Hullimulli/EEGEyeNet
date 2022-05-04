import numpy as np
from config import config
from utils import IOHelper
from scipy.ndimage import convolve1d
from benchmark import split
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import BayesianRidge
from Joels_Files.plotFunctions import prediction_visualisations
import time
from sklearn.metrics import mean_squared_error

def simpleDirectionRegressor(electrodes,regressor="SupportVectorMachine", nrOfRuns=1, findZeroCrossingBool=False,
                         movingAverageFilterLength=50, defaultCrossingValue=250):
    """

    Trains a Random Forest for the Direction Task, returns the score and visualizes the result.
    @param regressor: Which regressor is used. Can be "Bayesian Ridge", "Support Vector Machine". Defaults to Random Forest.
    @type regressor: String
    @param nrOfPoints: How many points have to be visualized.
    @type nrOfPoints: Integer
    @param nrOfRuns: Over how many runs the result should be averaged.
    @type nrOfRuns: Integer
    @param plotBool: If True, Visualizations are generated.
    @type plotBool: Bool
    @param findZeroCrossingBool: If the zerocrossing should be found manually. WARNING: This is implemented very inefficiently.
    @type findZeroCrossingBool: Bool
    @param movingAverageFilterLength: If zerocrossings are found manually, the signal is smooth with an moving average filter.
    This is long the filger is.
    @type movingAverageFilterLength: Integer
    @param defaultCrossingValue: If no zerocrossing is found, this is where the signal is split.
    @type defaultCrossingValue: Integer
    @return:
    @rtype:
    """
    if not config['task'] == 'Direction_task':
        print("Function only works for Direction Task.")
        return
    tempY = IOHelper.get_npz_data(config['data_dir'], verbose=True)[1]
    ids = tempY[:, 0]
    trainIndices, valIndices, testIndices = split(ids, 0.7, 0.15, 0.15)
    tempX = IOHelper.get_npz_data(config['data_dir'], verbose=True)[0][:, :, electrodes.astype(np.int) - 1]
    dataX = np.zeros([tempX.shape[0], 2 * tempX.shape[2]])

    if findZeroCrossingBool:
        # Moving Average
        tempXAvg = convolve1d(tempX, np.ones(movingAverageFilterLength) / movingAverageFilterLength, axis=1)

        for i in range(tempX.shape[2]):
            for j in range(tempX.shape[0]):
                zerosCrossing = (np.where(np.diff(np.sign(tempXAvg[j, :, i] - np.mean(tempXAvg[j, :, i]))))[0])
                if zerosCrossing.size != 0:
                    zerosCrossing = zerosCrossing[np.argmin(np.absolute(zerosCrossing - defaultCrossingValue))]
                else:
                    zerosCrossing = defaultCrossingValue
                if zerosCrossing == 0:
                    zerosCrossing = defaultCrossingValue
                dataX[j, 0 + i * 2] = np.mean(tempX[j, :zerosCrossing, i])
                dataX[j, 1 + i * 2] = np.mean(tempX[j, zerosCrossing:, i])
        del tempXAvg
    else:
        for i in range(tempX.shape[2]):
            dataX[:, 0 + i * 2] = np.mean(tempX[:, :250, i], axis=1)
            dataX[:, 1 + i * 2] = np.mean(tempX[:, 250:, i], axis=1)
    dataY = np.zeros(tempY.shape)
    dataY[:, 0] = tempY[:, 1]
    dataY[:, 1] = np.cos(tempY[:, 2])
    dataY[:, 2] = np.sin(tempY[:, 2])
    del tempX
    errorsAmp = np.zeros(nrOfRuns)
    errorsAng = np.zeros(nrOfRuns)
    trainingTime = np.zeros(nrOfRuns)
    for i in tqdm(range(nrOfRuns)):
        tic = time.time()
        if regressor == "BayesianRidge" or regressor == "Bayesian Ridge":
            regressor = "Bayesian Ridge"
            regrAmp = BayesianRidge()
            regrAngOne = BayesianRidge()
            regrAngTwo = BayesianRidge()
        elif regressor == "SupportVectorMachine" or regressor == "Support Vector Machine":
            regressor = "Support Vector Machine"
            regrAmp = svm.SVR(kernel="rbf")
            regrAngOne = svm.SVR(kernel="rbf")
            regrAngTwo = svm.SVR(kernel="rbf")
        else:
            regressor = "Random Forest"
            regrAmp = RandomForestRegressor()
            regrAngOne = RandomForestRegressor()
            regrAngTwo = RandomForestRegressor()
        regrAmp.fit(dataX[trainIndices], dataY[trainIndices, 0])
        regrAngOne.fit(dataX[trainIndices], dataY[trainIndices, 1])
        regrAngTwo.fit(dataX[trainIndices], dataY[trainIndices, 2])
        trainingTime[i] = time.time() - tic
        predictionAmp = regrAmp.predict(dataX[testIndices])
        predictionAngOne = regrAngOne.predict(dataX[testIndices])
        predictionAngTwo = regrAngTwo.predict(dataX[testIndices])
        predictionAng = np.arctan2(predictionAngTwo, predictionAngOne)
        errorsAmp[i] = meanSquareError(dataY[testIndices, 0], predictionAmp)
        errorsAng[i] = angleError(tempY[testIndices, 2], predictionAng) / np.pi * 180

        prediction_visualisations.visualizePredictionAmplitude(groundTruth=(dataY[testIndices, 0])[:10],
                                                           prediction=np.expand_dims(predictionAmp[:10],axis=(0,1)),directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/",modelNames=["SVM"],filename="SVM_Amp_Vis")
        prediction_visualisations.visualizePredictionAngle(groundTruth=(tempY[testIndices, 2])[:10],
                                                           prediction=np.expand_dims(predictionAng[:10],axis=(0,1)),directory="/Users/Hullimulli/Documents/ETH/SA2/debugFolder/",modelNames=["SVM"],filename="SVM_Ang_Vis")

    print(
        "The Average Amplitude Error is {}\u00B1{}px using {}. Training time was {}\u00B1{}s".format(np.mean(errorsAmp),
                                                                                                     np.std(errorsAmp),
                                                                                                     regressor, np.mean(
                trainingTime), np.std(trainingTime)))
    print("The Average Angle Error is {}°\u00B1{}° using {}. Training time was {}\u00B1{}s".format(np.mean(errorsAng),
                                                                                                   np.std(errorsAng),
                                                                                                   regressor, np.mean(
            trainingTime), np.std(trainingTime)))

def meanSquareError(y,yPred):
    return np.sqrt(mean_squared_error(y, yPred.ravel()))

def angleError(y,yPred,noMeanBool=False):
    difference = y - yPred.ravel()
    if noMeanBool:
        return np.absolute(np.arctan2(np.sin(difference), np.cos(difference)))
    return np.sqrt(np.mean(np.square(np.arctan2(np.sin(difference), np.cos(difference)))))