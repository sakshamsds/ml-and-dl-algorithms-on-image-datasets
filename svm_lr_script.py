import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    X = np.hstack((np.ones((n_data, 1)), train_data))
    w = initialWeights.reshape(n_features + 1, 1)

    prediction = sigmoid(X.dot(w))
    error_func = labeli * np.log(prediction) + (1.0 - labeli) * np.log(1.0 - prediction)
    error = (- 1.0 * np.sum(error_func)) / n_data

    error_grad = np.sum((prediction - labeli) * X, axis=0) / n_data
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    n_data = data.shape[0]
    X = np.hstack((np.ones((n_data, 1)), data))
    prob = sigmoid(X.dot(W))
    return np.argmax(prob, axis=1).reshape((n_data, 1))


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    X = np.hstack((np.ones((n_data, 1)), train_data))
    W = params.reshape((n_feature + 1, n_class))

    num = np.exp(X.dot(W))

    den = np.sum(num, axis=1)
    den = den.reshape(den.shape[0], 1)

    prediction = num / den

    innerSum = np.sum(Y * np.log(prediction))
    error = -np.sum(innerSum)

    error_grad = X.T.dot(prediction - labeli)
    error_grad = error_grad.flatten()

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    n_data   = data.shape[0]
    X = np.hstack((np.ones((n_data, 1)), data))
    t = np.sum(np.exp(X.dot(W)), axis=1)
    t = t.reshape(t.shape[0], 1)
    theta_value = np.exp(X.dot(W)) / t
    return np.argmax(theta_value, axis=1).reshape(n_data, 1)


# """
# Script for Logistic Regression
# """

print('\nBINARY LOGISTIC REGRESSION\n')

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    # print('iteration: ', i)
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""
##################
# YOUR CODE HERE #
##################

print('\nSUPPORT VECTOR MACHINE\n')


random_indices = np.random.randint(train_data.shape[0], size=10000)
training_data = train_data[random_indices, :]            
training_label = train_label[random_indices, :].flatten()

# Linear SVM
print('\nLinear Kernel')
linear_svm = svm.SVC(kernel='linear')
linear_svm.fit(training_data, training_label)
print('Training Accuracy: ' + str(round(linear_svm.score(training_data, training_label) * 100, 2)) + '%')
print('Validation Accuracy: ' + str(round(linear_svm.score(validation_data, validation_label) * 100, 2)) + '%')
print('Test Accuracy: ' + str(round(linear_svm.score(test_data, test_label) * 100, 2)) + '%')

# RBF with Gamma = 1
print('\nRBF Kernel with Gamma = 1')
rbf_svm = svm.SVC(kernel='rbf', gamma=1)
rbf_svm.fit(training_data, training_label)
print('Training Accuracy: ' + str(round(rbf_svm.score(training_data, training_label) * 100, 2)) + '%')
print('Validation Accuracy: ' + str(round(rbf_svm.score(validation_data, validation_label) * 100, 2)) + '%')
print('Test Accuracy: ' + str(round(rbf_svm.score(test_data, test_label) * 100, 2)) + '%')

# RBF with default Gamma = 'scale'
print('\nRBF Kernel with default Gamma = \'scale\' ')
rbf_svm_2 = svm.SVC(kernel='rbf', gamma='scale')
rbf_svm_2.fit(training_data, training_label)
print('Training Accuracy: ' + str(round(rbf_svm_2.score(training_data, training_label) * 100, 2)) + '%')
print('Validation Accuracy: ' + str(round(rbf_svm_2.score(validation_data, validation_label) * 100, 2)) + '%')
print('Test Accuracy: ' + str(round(rbf_svm_2.score(test_data, test_label) * 100, 2)) + '%')


# RBF with default gamma and variable C
accuracies = []
# c_values = np.arange(1, 10, 1)
c_values = np.arange(0, 101, 10)
c_values[0] = 1

# iterating c values
for c in c_values:
    print('\nRBF Kernel with C: ', c)
    rbf_model = svm.SVC(kernel='rbf', C=c, gamma='scale')
    rbf_model.fit(training_data, training_label)
    training_accuracy = round(rbf_model.score(training_data, training_label) * 100, 2)
    validation_accuracy = round(rbf_model.score(validation_data, validation_label) * 100, 2)
    test_accuracy = round(rbf_model.score(test_data, test_label) * 100, 2)

    print('Training Accuracy: ' + str(training_accuracy) + '%')
    print('Validation Accuracy: ' + str(validation_accuracy) + '%')
    print('Test Accuracy: ' + str(test_accuracy) + '%')

    accuracies.append([training_accuracy, validation_accuracy, test_accuracy])


# Whole dataset training
accuracies = np.array(accuracies)
optimal_c = c_values[np.argmax(accuracies[:, 1])]
optimal_rbf_model = svm.SVC(kernel='rbf', C=optimal_c, gamma='scale')
optimal_rbf_model.fit(train_data, train_label.flatten())
print('\nRBF with FULL training set with the optimal C : ')
print('Training Accuracy: ' + str(round(optimal_rbf_model.score(training_data, training_label) * 100, 2)) + '%')
print('Validation Accuracy: ' + str(round(optimal_rbf_model.score(validation_data, validation_label) * 100, 2)) + '%')
print('Test Accuracy: ' + str(round(optimal_rbf_model.score(test_data, test_label) * 100, 2)) + '%')


# Matlab Plot
plt.figure(figsize=(12, 8))
plt.plot(c_values, accuracies[:, 0], color='r')
plt.plot(c_values, accuracies[:, 1], color='g')
plt.plot(c_values, accuracies[:, 2], color='b')
plt.xlabel('C values', fontsize=16)
plt.ylabel('Accuracy', size=16)
plt.title('SVM Accuracy vs C', fontsize=20)
plt.xticks(c_values, fontsize=16)
plt.yticks(np.arange(95, 100), fontsize=16)
plt.legend(['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'], fontsize=20)


"""
Script for Extra Credit Part
"""

print('\nMULTI CLASS LOGISTIC REGRESSION\n')

# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
