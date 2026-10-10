import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import GenericUnivariateSelect, f_regression
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator

class LinearRegressionGD(BaseEstimator):
    def __init__(self, learning_rate, iterations, convergence_tolerance):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.convergence_tolerance = convergence_tolerance

        self.costs = []
        self.theta = []

    def fit(self, x_data, y_data):
        self.theta = np.zeros(x_data.shape[1] + 1)

        for i in range(self.iterations):
            # the linear regression equation
            y_predict = self.theta[0]
            for j in range(x_data.shape[1]):
                y_predict += self.theta[j + 1] * x_data[:, j]

            # get costs and save them
            m = y_predict.shape[0]
            self.costs.append(np.sum((y_predict - y_data) ** 2) / (2.0 * m))

            # check if predictions and true values are starting to converge through little cost change
            if i > 1 and abs(self.costs[i - 1] - self.costs[i]) < self.convergence_tolerance:
                break

            # update theta values, gradient descent
            self.theta[0] = self.theta[0] - (self.learning_rate / m) * np.sum(y_predict - y_data)
            for j in range(x_data.shape[1]):
                self.theta[j + 1] = self.theta[j + 1] - (self.learning_rate / m) * np.sum((y_predict - y_data) * x_data[:, j])

    def predict(self, test_data):
        # the linear regression equation, this time with the final theta values after running fit first
        test_predict = self.theta[0]
        for i in range(test_data.shape[1]):
            test_predict += self.theta[i + 1] * test_data[:, i]
        return test_predict

# get data
trainData = pd.read_csv("data/train.csv")
testData = pd.read_csv("data/test_features.csv")

# print number of missing values in every column
print("Number of missing values of each column:")
print(trainData.isna().sum())

# print number of rows with at least one missing value
rowsWithMissing = trainData.isnull().any(axis=1).sum()
print("Number of rows with at least one missing value:", rowsWithMissing, "\n")

# data clean, fixing unusual values
trainData.loc[trainData["Hours Studied"] > 9, "Hours Studied"] /= 10 # all unusual values were multiples of 10 greater than 10
trainData.loc[trainData["Previous Scores"] > 100, "Previous Scores"] = np.nan # all unusual values were greater than 100
trainData["Sleep Hours"] = trainData["Sleep Hours"].abs() # all unusual values were negative hours within range if they were positive

# data clean, filling in the missing values with the mean of the column it's in, for every column with missing values
meanValue = trainData["Weekly Study Hours"].mean()
trainData["Weekly Study Hours"] = trainData["Weekly Study Hours"].fillna(meanValue)
meanValue = trainData["Previous Scores"].mean()
trainData["Previous Scores"] = trainData["Previous Scores"].fillna(meanValue)
meanValue = trainData["Sleep Hours"].mean()
trainData["Sleep Hours"] = trainData["Sleep Hours"].fillna(meanValue)
meanValue = trainData["Sample Question Papers Practiced"].mean()
trainData["Sample Question Papers Practiced"] = trainData["Sample Question Papers Practiced"].fillna(meanValue)

# print mean of every column
print("Mean of each column:")
print(trainData.mean(), "\n")

# print the max of every column
print("Max of each column:")
print(trainData.max(), "\n")

# print the min of every column
print("Min of each column:")
print(trainData.min(), "\n")

# initialize the data
xRawTrainData = trainData[["Hours Studied", "Weekly Study Hours", "Previous Scores", "Extracurricular Activities", "Sleep Hours",
               "Sample Question Papers Practiced", "Commute Minutes"]].to_numpy()
yRawTrainData = trainData["Performance Index"].to_numpy()
test = testData[["Hours Studied", "Weekly Study Hours", "Previous Scores", "Extracurricular Activities", "Sleep Hours",
               "Sample Question Papers Practiced", "Commute Minutes"]].to_numpy()

# split then standardize
xTrain, xDev, yTrain, yDev = train_test_split(xRawTrainData, yRawTrainData, test_size=0.3, shuffle=True, random_state=42)
scaler = StandardScaler()
xTrainScaled = scaler.fit_transform(xTrain)
xDevScaled = scaler.transform(xDev)
xTestScaled = scaler.transform(test)

# training the model on the train data for the dev validation and predictions file, 0.03 and 1500
linearRegression = LinearRegressionGD(0.03, 1500, 0.00001)
linearRegression.fit(xTrainScaled, yTrain)

# performance check of the model using dev set
predictions = linearRegression.predict(xDevScaled)
print("Dev data MSE:", mean_squared_error(yDev, predictions))
print("Dev data RMSE:", np.sqrt(mean_squared_error(yDev, predictions)))
print("Number of iterations with dev data:", len(linearRegression.costs))
print("The last cost from dev data:", linearRegression.costs[-1])

# predict test data and write to predictions.csv
predictions = np.round(linearRegression.predict(xTestScaled), decimals=1)# round to one decimal place as shown from sample output from exam PDF
predictionsFile = pd.DataFrame({"ID" : testData["ID"], "Performance Index" : predictions})
predictionsFile.to_csv("data/predictions.csv", index=False)

featureNames = trainData.columns[:-1]
# below are two methods I've chosen for finding the three most important features
# first is using Generic Univariate Select from sci-kit learn to determine how much each feature contributes to the model, getting the top ones
findTopFeatures = GenericUnivariateSelect(score_func=f_regression, mode="percentile", param=50)
findTopFeatures.fit(xTrainScaled, yTrain)
findTopFeatures.transform(xDevScaled)
topFeatures = findTopFeatures.get_support()
topFeaturesScores = findTopFeatures.scores_
sortedFeatures = sorted(zip(featureNames, topFeaturesScores), key=lambda sort_index: sort_index[1], reverse=True)
print("\nThe top important features based on Generic Univariate Select:")
for featureIndex in range(3):
    # the score represents how strong of a relationship the feature has with the target, or performance index
    print(sortedFeatures[featureIndex][0], sortedFeatures[featureIndex][1])

# second is Permutation Importance to determine how much the model relies on each feature, getting the top ones using a for loop
permutationImportance = permutation_importance(linearRegression, xDevScaled, yDev, scoring="neg_mean_squared_error", n_repeats=10, random_state=42)
importances = permutationImportance.importances_mean
sortedFeatures = sorted(zip(featureNames, importances), key=lambda sort_index: sort_index[1], reverse=True)
print("\nThe top important features based on Permutation Importance:")
for featureIndex2 in range(3):# the "importance" represents how much shuffling a feature's values increases the MSE
    print(sortedFeatures[featureIndex2][0], sortedFeatures[featureIndex2][1])

#one prediction
values = np.array([[4,30,82,0,4,2,16]])
valuesScaled = scaler.transform(values)
thisPrediction = linearRegression.theta[0]
for valuesIndex in range(7):
    thisPrediction += linearRegression.theta[valuesIndex + 1] * valuesScaled[0][valuesIndex]
print(f"\n{thisPrediction}")