import pickle
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
import matplotlib.pyplot as plt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1/(1 + np.exp(-z))

# sigmoid(1)


def preprocess():
    """ Input:
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

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # # visualize single data sample
    # mat = loadmat('mnist_all.mat')
    # plt.imshow(mat['test8'][11].reshape((28, 28)))
    # plt.xticks([], [])
    # plt.yticks([], [])

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples.
    # Your code here.

    # image size 28*28

    trl = []
    tl = []
    for i in range(10):
        train_mat = mat['train' + str(i)]
        mat_labeled_train = np.concatenate(
            (train_mat, np.full((train_mat.shape[0], 1), i)), axis=1)
        trl.append(mat_labeled_train)

    for i in range(10):
        test_mat = mat['test' + str(i)]
        mat_labeled_test = np.concatenate(
            (test_mat, np.full((test_mat.shape[0], 1), i)), axis=1)
        tl.append(mat_labeled_test)

    labeled_train_full = np.vstack(
        (trl[0], trl[1], trl[2], trl[3], trl[4], trl[5], trl[6], trl[7], trl[8], trl[9]))

    np.random.shuffle(labeled_train_full)

    labeled_train = labeled_train_full[0:50000, :]
    train_data = labeled_train[:, 0:784]/255
    train_label = labeled_train[:, 784]
    labeled_validation = labeled_train_full[50000:60000, :]
    validation_data = labeled_validation[:, 0:784]/255
    validation_label = labeled_validation[:, 784]

    labeled_test_full = np.vstack(
        (tl[0], tl[1], tl[2], tl[3], tl[4], tl[5], tl[6], tl[7], tl[8], tl[9]))

    np.random.shuffle(labeled_test_full)

    test_data = labeled_test_full[:, 0:784]/255
    test_label = labeled_test_full[:, 784]

    # Selecting Features
    combined = np.vstack((train_data, validation_data))
    boolean_value_columns = np.all(combined == combined[0, :], axis=0)

    featureCount = 0
    global featureIndices
    featureIndices = []
    for n in range(len(boolean_value_columns)):
        if boolean_value_columns[n] == False:
            featureCount += 1
            featureIndices.append(n)
            # print(n, end=" ")
    print("\nNumber of selected features : ", featureCount)
    inverted_bool = ~boolean_value_columns
    final = combined[:, inverted_bool]
    traindata_shape = train_data.shape[0]
    train_data = final[0:traindata_shape, :]
    validation_data = final[traindata_shape:, :]
    test_data = test_data[:, inverted_bool]

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here

    # FORWARD PASS
    # for every training data bias value = 1
    input_shape = training_data.shape[0]
    input_plus_bias = np.concatenate(
        (np.ones((input_shape, 1)), training_data), axis=1)
    hidden_layer = sigmoid(np.dot(input_plus_bias, w1.T))
    hidden_layer_shape = hidden_layer.shape[0]

    hidden_layer_plus_bias = np.concatenate(
        (np.ones((hidden_layer_shape, 1)), hidden_layer), axis=1)
    outputs = sigmoid(np.dot(hidden_layer_plus_bias, w2.T))

    # ERROR
    true_labels = np.zeros((input_shape, n_class))

    # one hot encoding of true labels
    for i in range(input_shape):
        y_label = training_label[i]
        true_labels[i][y_label] = 1

    one_minus_true_labels = 1-true_labels
    one_minus_output_labels = 1-outputs 

    log_predicted = np.log(outputs)
    log_one_minus_predicted = np.log(one_minus_output_labels)
    neg_log_likelihood = -1 * np.sum(np.multiply(one_minus_true_labels, log_one_minus_predicted) +
                                     np.multiply(true_labels, log_predicted))/input_shape

    # BACKPROPAGATION
    label_error = outputs - true_labels
    w2_gradient = np.dot(label_error.T, hidden_layer_plus_bias)

    product_firstbias = hidden_layer_plus_bias * (1-hidden_layer_plus_bias)
    temp = np.dot(label_error, w2) * product_firstbias
    w1_gradient = np.dot(temp.T, input_plus_bias)
    w1_gradient = w1_gradient[1:, :]

    # REGULARIZATION
    reg = lambdaval * (np.sum(np.square(w1)) +
                       np.sum(np.square(w2))) / (2*input_shape)
    obj_val = neg_log_likelihood + reg

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    reg_gradient_w1 = (w1_gradient + lambdaval * w1)/input_shape
    reg_gradient_w2 = (w2_gradient + lambdaval * w2)/input_shape
    obj_grad = np.concatenate(
        (reg_gradient_w1.flatten(), reg_gradient_w2.flatten()), 0)

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image

    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here

    # feed forward
    input_plus_bias = np.concatenate(
        (np.ones((data.shape[0], 1)), data), axis=1)
    hidden_layer = sigmoid(np.dot(input_plus_bias, w1.T))

    hidden_layer_plus_bias = np.concatenate(
        (np.ones((hidden_layer.shape[0], 1)), hidden_layer), axis=1)
    outputs = sigmoid(np.dot(hidden_layer_plus_bias, w2.T))

    labels = np.argmax(outputs, axis=1)

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# hyperparameters
num_hidden_units = np.arange(7, 50, 7)      # 7 -> 49
lambda_values = np.arange(0, 61, 10)        # 0 -> 60

# 4*7 = 28
# 7*7 = 49

# training using multiple hyperparamets and choose the best based on the validation test accuracy
def train(n_hidden, lambdaval):
    # set the number of nodes in output unit
    n_class = 10

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate(
        (initial_w1.flatten(), initial_w2.flatten()), 0)

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True,
                         args=args, method='CG', options=opts)

    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden *
                     (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1))                     :].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters
    # find the accuracy on Training Dataset
    predicted_label = nnPredict(w1, w2, train_data)
    training_set_accuracy = round(100 *
                                  np.mean((predicted_label == train_label).astype(float)), 2)
    print('Training set Accuracy:' + str(training_set_accuracy) + '%')

    # find the accuracy on Validation Dataset
    predicted_label = nnPredict(w1, w2, validation_data)
    validation_set_accuracy = round(100 *
                                    np.mean((predicted_label == validation_label).astype(float)), 2)
    print('Validation set Accuracy:' + str(validation_set_accuracy) + '%')

    # find the accuracy on Test Dataset
    predicted_label = nnPredict(w1, w2, test_data)
    test_set_accuracy = round(100 *
                              np.mean((predicted_label == test_label).astype(float)), 2)
    print('Test set Accuracy:' + str(test_set_accuracy) + '%')

    return [training_set_accuracy, validation_set_accuracy, test_set_accuracy, w1, w2]


# all training
training_meta_data = []     # n_hidden * lambda * 4

for n_hidden in num_hidden_units:
    n_hidden_training_data = []
    for lambdaval in lambda_values:
        print("Hidden units: ", n_hidden, "| lambda val: ", lambdaval)
        start = time.time()
        single_training_meta_data = train(n_hidden, lambdaval)
        training_time = round(time.time()-start, 2)
        print("Training time: %s seconds \n" % training_time)
        single_training_meta_data.append(training_time)
        n_hidden_training_data.append(single_training_meta_data)

    training_meta_data.append(n_hidden_training_data)


max_acc = 0
optimal_hidden_nodes = 0
optimal_lambda = 0
optimal_w1 = None
optimal_w2 = None
optimal_test_accuracy = None

# get the parameters for best validation accuracy
for i in range(len(training_meta_data)):
    for j in range(len(training_meta_data[i])):
        val_acc = training_meta_data[i][j][1]
        if val_acc > max_acc:
            max_acc = val_acc
            optimal_hidden_nodes = num_hidden_units[i]
            optimal_lambda = lambda_values[j]
            optimal_w1 = training_meta_data[i][j][3]
            optimal_w2 = training_meta_data[i][j][4]
            optimal_test_accuracy = training_meta_data[i][j][2]


print("Best validation accuracy: ", max_acc, "\nFor hyperparameters: ",
      "\n\t Number of hidden nodes = ", optimal_hidden_nodes,
      "\n\t Regularization constant = ", optimal_lambda,
      "\n\t Accuracy on unseen data = ", optimal_test_accuracy)


# save data into pickle file

params = {}
params['λ'] = optimal_lambda
params['n_hidden'] = optimal_hidden_nodes
params['selected_features'] = featureIndices
params['w1'] = optimal_w1
params['w2'] = optimal_w2
pickle.dump(params, open('params.pickle', 'wb'))
load_pickle = pickle.load(file=open('params.pickle', 'rb'))

# save training metadata
# pickle.dump(training_meta_data, open('training_meta_data.pickle', 'wb'))
# training_meta_data = pickle.load(file=open('training_meta_data.pickle', 'rb'))


'''
# PLOTS

# optimal_hidden_nodes = 42
# optimal_lambda = 10
node_index = np.where(num_hidden_units == optimal_hidden_nodes)[0][0]
lambda_index = np.where(lambda_values == optimal_lambda)[0][0]

training_meta_data[node_index][lambda_index][0]

# GRAPH 1
graph1_training_accuraccies = []
graph1_validation_accuraccies = []
graph1_test_accuraccies = []
for i in range(len(num_hidden_units)):
    graph1_training_accuraccies.append(training_meta_data[i][lambda_index][0])
    graph1_validation_accuraccies.append(training_meta_data[i][lambda_index][1])
    graph1_test_accuraccies.append(training_meta_data[i][lambda_index][2])

plt.plot(num_hidden_units, graph1_training_accuraccies, color='r')
plt.plot(num_hidden_units, graph1_validation_accuraccies, color='g')
plt.plot(num_hidden_units, graph1_test_accuraccies, color='b')
plt.xlabel('# Hidden nodes')
plt.ylabel('Accuracy')
plt.xticks(num_hidden_units)
plt.title('Accuracy vs # Hidden nodes | λ=' + str(optimal_lambda))
plt.grid(linewidth=0.5, which='both', linestyle='--')
# plt.minorticks_on()
plt.legend(labels=['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])



# GRAPH 2
graph2_training_accuraccies = []
graph2_validation_accuraccies = []
graph2_test_accuraccies = []
for i in range(len(lambda_values)):
    graph2_training_accuraccies.append(training_meta_data[node_index][i][0])
    graph2_validation_accuraccies.append(training_meta_data[node_index][i][1])
    graph2_test_accuraccies.append(training_meta_data[node_index][i][2])

plt.plot(lambda_values, graph2_training_accuraccies, color='r')
plt.plot(lambda_values, graph2_validation_accuraccies, color='g')
plt.plot(lambda_values, graph2_test_accuraccies, color='b')
plt.xlabel('λ')
plt.ylabel('Accuracy')
plt.xticks(lambda_values)
plt.title('Accuracy vs λ | # Hidden nodes = ' + str(optimal_hidden_nodes))
plt.grid(linewidth=0.5, linestyle='--')
plt.legend(labels=['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])


# GRAPH 3
graph3_training_time = []
for i in range(len(num_hidden_units)):
    graph3_training_time.append(training_meta_data[i][lambda_index][-1])

plt.plot(num_hidden_units, graph3_training_time, color='#59C1BD')
plt.xticks(num_hidden_units)
plt.xlabel('# Hidden nodes')
plt.ylabel('Training Time')
plt.title('Training Time vs # Hidden nodes | λ=' + str(optimal_lambda))
plt.grid(linewidth=0.5, which='both', linestyle='--')


# GRAPH 4
graph4_training_time = []
for i in range(len(lambda_values)):
    graph4_training_time.append(training_meta_data[node_index][i][-1])

plt.plot(lambda_values, graph4_training_time, color='#59C1BD')
plt.xlabel('λ')
plt.ylabel('Training Time')
plt.title('Training Time vs λ | # Hidden nodes = ' + str(optimal_hidden_nodes))
plt.grid(linewidth=0.5, which='both', linestyle='--')
'''
