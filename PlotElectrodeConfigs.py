import numpy as np
from Joels_Files.plotFunctions.electrode_plots import colourCode, electrodePlot

top2 = np.array([125,128])
top3 = np.array([17,125,128])
top8 = np.array([1, 17, 32, 38, 121, 125, 128, 129])
top23 = np.array([1,4, 5,6,8,12,14,17, 19,21,25,32, 33,38, 43,120,121, 122,125, 126,127,128, 129])
#top32 = np.array([1,2,8,9,14,17,21,22,25,26,32,33,38,43,59,60,65,66,70,83,84,85,90,91,120,121,122,125,126,127,128,129])
#top65 = np.array([1,2,7,8,9,10,14,15,17,18,21,22,24,25,26,27,28,32,33,34,35,38,39,43,44,45,46,47,50,53,57,58,59,60,65,66,70,83,84,85,86,90,91,96,98,100,101,102,106,108,110,114,115,116,117,120,121,122,123,124,125,126,127,128,129])

topBack = np.array([59,60,65,66,70,83,84,85,90,91,129])
back = np.array([50,58,59,60,65,66,70,83,84,85,90,91,96,101,129])
topFront = np.array([1,8,14,17,21,25,32,38,43,120,121,125,126,127,128])
Front = np.array([1,2,8,9,14,17,21,22,25,26,27,32,33,38,39,43,115,120,121,122,123,125,126,127,128])
top40 = np.array([1,2,8,9,14,17,21,22,25,26,27,32,33,38,39,43,50,58,59,60,65,66,70,83,84,85,90,91,96,101,115,120,121,122,123,125,126,127,128,129])

scores = np.zeros(129)
scores[top2-1] += 1
scores[top3-1] += 1
scores[top8-1] += 1
scores[top23-1] += 1
zeros = np.argwhere(scores==0)
colours = colourCode(scores,colourMap='gist_rainbow')
colours[zeros] = np.array([255,255,255])
electrodePlot(colours,alpha=0.5,directory='/Users/Hullimulli/Documents/ETH/SA2/00Neurips2022/Results',filename="Minimally_Clusters")

scores = np.zeros(129)
scores[topFront-1] += 1
scores[Front-1] += 1
scores[top40-1] += 3
scores[topBack-1] -= 1
scores[back-1] -= 1
zeros = np.argwhere(scores==0)
colours = colourCode(scores,colourMap='gist_rainbow')
colours[zeros] = np.array([255,255,255])
electrodePlot(colours,alpha=0.5,directory='/Users/Hullimulli/Documents/ETH/SA2/00Neurips2022/Results',filename="Maximally_Clusters")