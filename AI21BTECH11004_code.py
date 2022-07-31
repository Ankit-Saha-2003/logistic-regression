# Binary Classification using Logistic Regression

# Name: Ankit Saha
# Roll number: AI21BTECH11004

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the dataset file
df = pd.read_csv('Epoch Hackathon - Dataset.csv')
print(df, '\n')
total_count = len(df)
print(f'Size of the data set = {total_count}\n')
print(df.describe(), '\n')
print((df.describe())['quality'], '\n')

## Exploratory Data Analysis

# Finding the distibution of good and bad quality wines
good_count = np.sum(df['quality'])
bad_count = total_count - good_count
print(f"Number of good quality wines = {good_count}")
print(f"Number of bad quality wines = {bad_count}")
print(f"Good : Bad = {np.around(good_count * 100 / total_count, 3)}% : {np.around(bad_count * 100 / total_count, 3)}%\n")

# Plotting scatter plots and histograms to visualize the pairwise relations between sets of 4 variables and our target variable 'quality'
ls1 = list(df)[:4]
ls1.append('quality')
sc1 = sns.pairplot(df.loc[:, ls1], hue='quality')
plt.savefig('figs/fig-1.png')
plt.show()

ls2 = list(df)[4:8]
ls2.append('quality')
sc2 = sns.pairplot(df.loc[:, ls2], hue='quality')
plt.savefig('figs/fig-2.png')
plt.show()

ls3 = ['pH', 'sulphates', 'alcohol', 'percentage_free_sulphur', 'quality']
sc3 = sns.pairplot(df.loc[:, ls3], hue='quality')
plt.savefig('figs/fig-3.png')
plt.show()

ls4 = ['k_value', 'l_value', 'm_value', 'n_value', 'quality']
sc4 = sns.pairplot(df.loc[:, ls4], hue='quality')
plt.savefig('figs/fig-4.png')
plt.show() # k_value is a linear function (twice) of l_value

# Plotting a correlation heatmap to identify relations between the variables
sns.heatmap(df.corr())
plt.savefig('figs/heatmap.png')
plt.show()
# From the heatmap, we see a strong correlation between the following:
# (fixed acidity, k_value), (fixed acidity, l_value), (k_value, l_value), (residual sugar, n_value), (sulphates, m_value)

ls5 = ['fixed acidity', 'k_value', 'l_value', 'quality']
sc5 = sns.pairplot(df.loc[:, ls5], hue='quality')
plt.savefig('figs/fig-5.png')
plt.show() # fixed acidity, k_value and l_value are seemingly linear functions of each other
print('k_value vs l_value\n', (df['k_value'] / df['l_value']).describe(), '\n')
print('fixed acidity vs k_value\n', (df['fixed acidity'] / df['k_value']).describe(), '\n')
print('fixed acidity vs l_value\n', (df['fixed acidity'] / df['l_value']).describe(), '\n')

sc6 = df.plot(kind='scatter', x='residual sugar', y='n_value')
plt.savefig('figs/fig-6')
plt.show() # Fairly linear
print('residual sugar vs n_value\n', (df['residual sugar'] / df['n_value']).describe(), '\n')

sc7 = df.plot(kind='scatter', x='sulphates', y='m_value')
plt.savefig('figs/fig-7')
plt.show() # Fairly linear
print('sulphates vs m_value\n', (df['sulphates'] / df['m_value']).describe(), '\n')

## Feature Engineering

# Since k_value, l_value, m_value, n_value all have other variables that they are strongly correlated with, we can drop them from our data
df = df.drop(columns=['k_value', 'l_value', 'm_value', 'n_value'])
print(df, '\n')

## Training the model

# Splitting the dataset into training and test sets (80:20)
training_indices = random.sample(range(total_count), int(0.8 * total_count))
test_indices = [i for i in range(total_count) if i not in training_indices]
training_set = df.iloc[training_indices].drop(columns='quality')
test_set = df.iloc[test_indices].drop(columns='quality')
training_labels = df.iloc[training_indices]['quality']
test_labels = df.iloc[test_indices]['quality']
print(f'Size of the training data set = {len(training_set)}')
print(f'Size of the test data set = {len(test_set)}')

def sigmoid(z):
    """ Returns the sigmoid of z, i.e., 1/(1 + exp(-z)) """
    return 1 / (1 + np.exp(-z))

def minus_log_likelihood(X, y, W):
    """ Returns the (negative of the log-likelihood function) which is to be minimized
    :param X: Input vector
    :param y: Labels
    :param W: Weight vector
    """
    z = np.dot(X, W)
    J = np.sum(y * -np.log1p(np.exp(-z)) + (1 - y) * -np.log1p(np.exp(z))) / len(X) 
    # log1p is being used for precision reasons when exp(-z) or exp(z) is very small
    return -J

def gradient(X, y, W):
    """ Returns the gradient of the log-likelihood function with parameters X, y, W """
    z = np.dot(X, W)
    return np.dot(X.T, sigmoid(z) - y) / len(X)

def steepest_descent(X, y):
    """ Returns the weight vector obtained after training the model """
    W = np.zeros(len(X.columns)) # Initial guess
    epsilon = 1e-6 # To check if norm(grad(W)) has obtained stability
    step_size = 1e-2 # To determine how much to reduce our guess by at each iteration
    g_prev = gradient(X, y, W)
    W -= step_size * g_prev
    g_curr = gradient(X, y, W)

    while (abs(np.linalg.norm(g_curr) - np.linalg.norm(g_prev)) > epsilon): 
        # Keep repeating until the gradient is sufficiently small (approaches zero => minima)
        g_prev = g_curr
        W -= step_size * g_curr
        g_curr = gradient(X, y, W)

    return W

# Performing logistic regression to get the weight vector
weight_vector = steepest_descent(training_set, training_labels)
print(f'\nWeight vector obtained after training =\n {weight_vector}')

## Testing and Evaluation

# Testing the model on the test data set
predicted_output = np.dot(test_set, weight_vector)
predicted_labels = []
for output in predicted_output:
    if (sigmoid(output) >= 0.5):
        predicted_labels.append(1)
    else:
        predicted_labels.append(0)

# Evaluating the model
TP = 0 # True positives
FP = 0 # False positives
TN = 0 # True negatives
FN = 0 # False negatives

for i in range(len(test_labels)):
    if test_labels.iloc[i] and predicted_labels[i]:
        TP += 1
    elif not test_labels.iloc[i] and predicted_labels[i]:
        FP += 1
    elif not test_labels.iloc[i] and not predicted_labels[i]:
        TN += 1
    else:
        FN += 1

print(TP, FP, TN, FN)
print(f'\nAccuracy = {(TP + TN) / len(test_labels)}')
print(f'Precision = {TP / (TP + FP)}')
print(f'Recall/Sensitivity = {TP / (TP + FN)}')
print(f'Specificity = {TN / (TN + FP)}')
