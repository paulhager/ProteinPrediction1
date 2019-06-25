from matplotlib.colors import ListedColormap
import matplotlib
import matplotlib.cm as matcm
import matplotlib.pyplot as plt
import numpy as np

statPath = 'teststatistics.txt'
resultsPath = 'testresults.txt'
def loadStats(Path):
    # stats:
    # TP, FP, TN, FN, FPR, Precision, Recall/TPR, F1-score, MCC
    statsDict = {}
    with open(Path, 'r') as statfile:
        for line in statfile:
            val = line.split('\t')
            val[0] = val[0].strip(':')
            val[1] = val[1].strip('\n')
            statsDict[val[0]] = val[1]
    statfile.close()
    return statsDict

def loadResults(Path):
    # Format:
    # Position_label, Predicted_value, Result
    # example:
    # [0, 0.232442, TN]
    resultsList = []
    with open(Path, 'r') as results:
        for line in results:
            line = line.strip('\n')
            val = line.split('\t')
            resultsList.append(val)
    results.close()
    return resultsList

def makeConfMat(statsDict):
    cm = [[int(statsDict['TP']), int(statsDict['FP'])], [int(statsDict['FN']), int(statsDict['TN'])]]
    plt.clf()
    plt.imshow(cm, cmap=matcm.get_cmap('tab20b'), norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    classNames = ['Binding', 'Non-Binding']
    plt.title('Confusion Matrix - Test Data')
    plt.ylabel('Predicted')
    plt.xlabel('True')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i][j])
    plt.show()


statsDict = loadStats(statPath)
resultsList = loadResults(resultsPath)
makeConfMat(statsDict)