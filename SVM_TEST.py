import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
from sklearn import metrics
import matplotlib.pyplot as plt


# >> Check the correlations for the features. If there is any feature that has correlation>= 0.9 then it remove it from the dataset<< #
def featuresFiltering_correlation(X):
    corr_threshold = 0.9
    corr = X.corr()  # Calculate the corr for the features
    drop_columns = np.full(corr.shape[0], False, dtype=bool)  # create a list with 15 False
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= corr_threshold:
                drop_columns[j] = True
    columns_dropped = X.columns[drop_columns]
    X.drop(columns_dropped, axis=1, inplace=True)
    return columns_dropped


# >> Check the pvalue for the features. If there is any feature that has pvalue> 0.5 then it remove it from the dataset<< #
def features_with_less_significant(X, Y):
    alpha = 0.05
    regression_ols = None
    columns_dropped = np.array([])
    for itr in range(0, len(X.columns)):
        ## performing the regression and fitting the model using Ordinary least squares
        regression_ols = sm.OLS(Y, X).fit()
        # to get the feature that has the hights pvalue
        max_feature_col = regression_ols.pvalues.idxmax()
        max_p_val = regression_ols.pvalues.max()
        if max_p_val > alpha:
            X.drop(max_feature_col, axis='columns', inplace=True)
            columns_dropped = np.append(columns_dropped, [max_feature_col])
        else:
            break
    regression_ols.summary()
    # return the removed features
    return columns_dropped


# Calculate the cost function
def calc_cost(W, X, Y):
    # calculate hinge loss
    N = X.shape[0]
    # the reason why we add a column with 1 (w*x+b)
    max_distances = 1 - Y * (np.dot(X, W))
    max_distances[max_distances < 0] = 0  # equivalent to max(0, distance) to get the max distance
    hinge_loss = C * (np.sum(max_distances) / N)

    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost


# Calculate the gradient cost
def calc_gradient_cost(W, X_batch, Y_batch):

    # if only one example is passed
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])  # gives multidimensional array

    distance = 1 - (Y_batch * np.dot(X_batch, W))
    sum = np.zeros(len(W))

    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (C * Y_batch[ind] * X_batch[ind])
        sum += di

    sum = sum / len(Y_batch)
    # return the gradient of the cost function
    return sum


def stochastic_gradient_descent(features, outputs):
    max_epochs = 5000
    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.01  # in percent
    # stochastic gradient descent
    for epoch in range(1, max_epochs):
        # shuffle to prevent repeating update cycles
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            gradient = calc_gradient_cost(weights, x, Y[ind])
            weights = weights - (learning_rate * gradient)

        # convergence check on 2^nth epoch
        if epoch == 2 ** nth or epoch == max_epochs - 1:
            cost = calc_cost(weights, features, outputs)
            #print("Epoch is: {} and Cost is: {}".format(epoch, cost))
            # stoppage criterion
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = cost
            nth += 1
    return weights


def start():
    #print("reading dataset...")

    # read data in pandas (pd) data frame
    data = pd.read_csv(r'C:\Users\hebam\Desktop\cancerdata.csv')

    #print("applying feature engineering...")

    # Replace the data to numbers
    lung_cancer = {'YES': 1.0, 'NO': -1.0}
    data['LUNG_CANCER'] = data['LUNG_CANCER'].map(lung_cancer)
    gender = {'F': 0.0, 'M': 1.0}
    data['GENDER'] = data['GENDER'].map(gender)

    # put features & outputs in different data frames
    Y = data.loc[:, 'LUNG_CANCER']
    X = data.iloc[:, 0:-1]

    # filter features
    featuresFiltering_correlation(X)
    features_with_less_significant(X, Y)

    # normalize data for better convergence and to prevent overflow
    X_normalized = MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(X_normalized)

    # insert 1 in every row for intercept b
    X.insert(loc=len(X.columns), column='intercept', value=1)
    # print(X)
    # print(X.shape[0])
    # split data into train and test set
    #print("splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=42)

    # train the model
    print("training started...")
    W = stochastic_gradient_descent(X_train.to_numpy(), y_train.to_numpy())
    print("training finished.")
    print("weights are: {}".format(W))

    # testing the model
    print("testing the model...")
    y_train_predicted = np.array([])
    for i in range(X_train.shape[0]):
        yp = np.sign(np.dot(X_train.to_numpy()[i], W))
        y_train_predicted = np.append(y_train_predicted, yp)

    predicted_answer = np.array([])
    for i in range(X_test.shape[0]):
        trans = np.sign(np.dot(X_test.to_numpy()[i], W))
        predicted_answer = np.append(predicted_answer, trans)

## To plot confusion matrix
    """confusion_matrix = metrics.confusion_matrix(y_test, predicted_answer)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["No", "Yes"])

    cm_display.plot()
    plt.title('Confusion Matrix for SVM')
    plt.show()"""


    l=len(predicted_answer)
    correct=0
    for i in range(l):
        RightAnswar = y_test.to_numpy()[i]
        predictedAnswar= predicted_answer[i]

        if RightAnswar == predictedAnswar:
            correct = correct + 1
    acc= correct/l*100

    print("Accuracy =  {}".format(acc))



    return acc

# set hyper-parameters and call init
C = 10000
learning_rate = 0.000001
start()