import ast
import sys
import pickle
import simplejson
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,8)

from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score


#function to act as converter when default json converter fails
def convert(o):
    if isinstance(o, np.int64): 
        return int(o) 
    elif isinstance(o, np.bool_):
        if o == True:
            return True
        return False
    elif isinstance(o, pd.Timestamp):
        if o.hour == 0 and o.minute == 0 and o.second == 0:
            return o.strftime("%d-%m-%Y")
        else:
            return str(o)


class RegressionScore:
  def __init__(self, ytrue, ypred, metric='explained_variance_score', sample_weight=None, multioutput='uniform_average'):
    self.ytrue = ytrue
    self.ypred = ypred
    self.metric = metric
    self.sample_weight = sample_weight
    self.multioutput = multioutput

  def explained_variance_score(self):
    return explained_variance_score(self.ytrue, self.ypred, self.sample_weight, self.multioutput) 

  def max_error(self):
    return max_error(self.ytrue, self.ypred)

  def mean_absolute_error(self):
    return mean_absolute_error(self.ytrue, self.ypred, self.sample_weight, self.multioutput)
  
  def mean_squared_error(self):
    return mean_squared_error(self.ytrue, self.ypred, self.sample_weight, self.multioutput)

  def mean_squared_log_error(self):
    return mean_squared_log_error(self.ytrue, self.ypred, self.sample_weight, self.multioutput)
  
  def median_absolute_error(self):
    return median_absolute_error(self.ytrue, self.ypred)

  def r2_score(self):
    return r2_score(self.ytrue, self.ypred, self.sample_weight, self.multioutput)

  def evaluate_score(self):
    if self.metric == "explained_variance_score":
      return self.explained_variance_score()
    
    elif self.metric == "max_error":
      return self.max_error()

    elif self.metric == "mean_absolute_error":
      return self.mean_absolute_error()
    
    elif self.metric == "mean_squared_error":
      return self.mean_squared_error()

    elif self.metric == "mean_squared_log_error":
      return self.mean_squared_log_error()
    
    elif self.metric == "median_absolute_error":
      return self.median_absolute_error()

    elif self.metric == "r2_score":
      return self.r2_score()


# This function will return yTrue and yPred after reading the files
# and generating predictions using pickled model
def loadModel(pklFilePath, dataPath, trueLabel, columnNames, flag):
    
    with open(pklFilePath,'rb') as file:
        model = pickle.load(file)

    if columnNames == "None":
        testDf = pd.read_csv(dataPath)
    else:
        testDf = pd.read_csv(dataPath, usecols=columnNames)

    if flag == "Yes": # trueLabel == "Name of a columns in dataPath"
        yTrueSeries = testDf[trueLabel]
        testDf.drop(columns=trueLabel, inplace=True)
        arr0 = yTrueSeries.values
    
    else: # trueLabel == "path of ytrue.csv"
        yTrueDf = pd.read_csv(trueLabel)
        arr0 = yTrueDf.iloc[:,0].to_numpy()
    
    if arr0.dtype == "object":  
        yTrue = []             
        for l in arr0:
            yTrue.append(ast.literal_eval(l))
        yTrue = np.array(yTrue)
    else:
        yTrue = arr0

    yPred = model.predict(testDf)

    # when yTrue.shape == (N, 1) and yPred.shape == (N, )
    if yTrue.shape != yPred.shape:
        yTrue = yTrue.reshape(yTrue.shape[0], )

    return yPred, yTrue


def plot(pred, dimension, outputFolder, residuals, plotName="residualPlot", xlabel="Predictions"):
    plt.figure()
    if dimension == 0:
        plt.scatter(pred, residuals, alpha=0.5)
    else:
        plt.scatter(pred[:,dimension], residuals[:,dimension], alpha=0.5)
    plt.ylabel("Residuals")
    plt.xlabel(xlabel)
    plt.title("Prediction Residual plot")
    plt.savefig(f"{outputFolder}{plotName}.png")
    plt.close() 


# This function will loop through the available possile metrics
# and generate those mteric scores for yTrue and yPred
def generateMetrics(yTrue, yPred, sample_weight, multioutput):

    if multioutput == "variance_weighted":
        multipleOutputMetrics = ['explained_variance_score', 'r2_score']
    else:
        multipleOutputMetrics = ['explained_variance_score', 'mean_absolute_error', 
           'mean_squared_error', 'median_absolute_error', 'r2_score', 'mean_squared_log_error']

    errorMetrics = {}

    if yTrue.ndim == 1:
        metric = "max_error"
        rs = RegressionScore(yTrue, yPred, metric=metric)
        errorMetrics[metric] = np.round(rs.evaluate_score(), 2)

        for metric in multipleOutputMetrics:
            try:
                rs = RegressionScore(yTrue, yPred, metric=metric, sample_weight=sample_weight, multioutput=multioutput)
                errorMetrics[metric] = np.round(rs.evaluate_score(), 2)
            except Exception as e:
                pass
            
    elif yTrue.ndim > 1:
        for metric in multipleOutputMetrics:
            try:
                rs = RegressionScore(yTrue, yPred, metric=metric, sample_weight=sample_weight, multioutput=multioutput)
                errorMetrics[metric] = np.round(rs.evaluate_score(), 2)
            except Exception as e:
                pass

    return errorMetrics


if __name__ == "__main__":

    flag = sys.argv[1]  # "Yes" or "No"
    trueLabel = sys.argv[2]  # trueLabel == "Name of cols in dataPath" if flag=="Yes" else "ytrue.csv"
    dataPath = sys.argv[3]
    sample_weight = sys.argv[4] # None or path of CSV file
    multioutput = sys.argv[5]
    outputFolder = sys.argv[6]
    pklFilePath = sys.argv[7]
    columnNames = sys.argv[8]

    if not outputFolder.endswith("/"):
        outputFolder += "/"

    try:
        if sample_weight == "None":
            sample_weight = None
        else:
            weightsDf = pd.read_csv(sample_weight)
            arr0 = weightsDf.iloc[:,0].to_numpy()
    
            # List of list in pd series have object dtype
            # Converting the object to list of lists
            if arr0.dtype == "object":  
                sample_weight = []             
                for l in arr0:
                    sample_weight.append(ast.literal_eval(l))
                sample_weight = np.array(sample_weight)
            else:
                sample_weight = arr0

    except Exception as e:
        sample_weight = None

    if flag == "Yes":
        trueLabel = ast.literal_eval(trueLabel)

    if columnNames != "None" and type(columnNames) == str:
        columnNames = ast.literal_eval(columnNames)
        columnNames.extend(trueLabel)if flag == "Yes" else "do nothing"
        
    yPred, yTrue = loadModel(pklFilePath, dataPath, trueLabel, columnNames, flag)

    # If sample weight shape is not equal to predictions shape and sample weight dim > 1
    if sample_weight is not None:
        if yPred.shape != sample_weight.shape and sample_weight.ndim != 1:
            sample_weight = None

    # If multioutput input is a list
    try:
        multioutput = ast.literal_eval(multioutput)
        if yPred.ndim == 1 or len(multioutput) < 2:
            multioutput = 'uniform_average'
        elif yPred.shape[1] != len(multioutput):
            multioutput = 'uniform_average'
    except Exception as e:
        pass

    res = generateMetrics(yTrue, yPred, sample_weight, multioutput)

    with open(f"{outputFolder}regressionMetrics.json", "w") as write_file:
        simplejson.dump(res, write_file, default=convert, ignore_nan=True)

    # Plotting
    residuals = yTrue - yPred

    try:
        residualsDim = len(residuals[0])
    except Exception as e:
        residualsDim = 1

    if residualsDim == 1:
        plot(yPred, 0, outputFolder, residuals, plotName="residualPlot_dimension0")

    elif residualsDim in (2,3):
        for i in range(residualsDim):
            plotName = f"residualPlot_dimension{i}"
            xlabel = f"Prediction dimension {i}"
            plot(yPred, i, outputFolder, residuals, plotName=plotName, xlabel=xlabel)