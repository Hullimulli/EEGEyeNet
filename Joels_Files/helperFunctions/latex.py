import numpy as np
from texttable import Texttable
import latextable
import pandas as pd
import os

def generateTable(file , directory,filename='tableLatex' ,caption="Performance of each Network",
                  nrOfDigits=2 ,scale=1 ,transposed=True):
    """
    Returns the latex code for a table saved in a csv file.
    @param file: The csv filename with path from which the table has to be generated.
    @type file: String
    @param filename: Name of the txt output file.
    @type filename: String
    @param addNrOfParams: If True, the #paramaters of the network will be in the table.
    @type addNrOfParams: Bool
    @param caption: Title of the table.
    @type caption: String
    @param nrOfDigits: To what digit the entries are rounded.
    @type nrOfDigits: Integer
    @param scale: With what float all data points have to be multiplied.
    @type scale: Float
    @param transposed: If True, the header is a column.
    @type transposed: Bool
    @return:
    @rtype:
    """
    data = pd.read_csv(file ,header=None)
    data = data.astype(str).values.tolist()
    dataCopy = data
    j=0
    deviationIndices = list()
    if transposed:
        while (j<len ( data)):
            # Checks if header entry has a corresponding standard deviation on the subsequent header entry
            if not j == len(data) - 1 and 'std' in data[j+1][ 0 ].lower():
                # Put standard deviation as plus/minus
                for i in range(1,len( data[0])):
                    dataCopy[j][i] = '$'+str ( round(scale*float(data[j][i]), nrOfDigits))+' \pm '+ str(round(scale*float(data[j+1][ i ]), nrOfDigits))+'$'
                deviationIndices.append(j+1)
                j+=2
            else:
                for i in range(1, len(data[0])):
                    dataCopy[j][i] = data[j][i]
                j+=1
        deviationIndices = np.asarray(deviationIndices,dtype=int)[::-1]

        # Remove lines with standard deviation
        for j in deviationIndices:
            for i in np.arange(len(dataCopy[j]))[::-1]:
                del dataCopy[j][i]
            del dataCopy[j]
        data = dataCopy
    else:
        while (j < len(data[0])):
            if not j == len(data[0]) - 1 and 'std' in data[0][j + 1].lower():
                for i in range(1, len(data)):
                    dataCopy[i][j] = '$' + str(round(scale * float(data[i][j]), nrOfDigits)) + ' \pm ' + str(round(scale * float(data[i][j+1]) , nrOfDigits)) + '$'
                deviationIndices.append(j + 1)
                j += 2
            else:
                for i in range(1, len(data)):
                    dataCopy[i][j] = data[i][j]
                j += 1
        deviationIndices = np.asarray(deviationIndices, dtype=int)[::-1]

        for i in np.arange(len(dataCopy))[::-1]:
            for j in deviationIndices:
                del dataCopy[i][j]
        data = dataCopy

    table = Texttable()
    table.set_cols_align(["c"] * len(data[0]))
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(data)
    with open(os.path.join(directory,filename+'.txt'), 'w') as f:
        f.write(latextable.draw_latex(table, caption=caption))