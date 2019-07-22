import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output


class Layer(object):
    def __init__(self, num_prev, num_neurons, activation_function='sigmoid'):
        '''
        Initialize weights randomly with He scaling.

        self.A -- [num_neurons, 1] Array of activations
        self.Z -- [num_neurons, 1] Array of linearly aggregated inputs
        self.W -- [num_neurons, num_prev] Array of weights applied to previous layer's activations A.
        self.b -- [num_neurons, 1] Array of biases
        '''
        self.n = num_neurons
        self.act = activation_function

        self.A = np.zeros((num_neurons,1))
        self.Z = np.zeros((num_neurons,1))
        self.b = np.zeros((self.n,1))

        # He initialization.
        if self.act == 'sigmoid':
            self.W = np.random.randn(num_neurons, num_prev) * np.sqrt(2.0/num_prev)
        # Xavier initialization
        elif self.act == 'tanh':
            self.W = np.random.randn(num_neurons, num_prev) * np.sqrt(1.0/num_prev)

        self.dA = np.zeros((num_neurons,1))
        self.dZ = np.zeros((num_neurons,1))
        self.db = np.zeros((self.n,1))
        self.dW = np.zeros((num_neurons, num_prev))

    def forward(self, A_prev):
        '''
        Description:
        Perform forward pass by computing Z, then A.

        Parameters:
        A_prev -- [num_prev, m_samples] Array of activations from previous layer.

        RETURNS:
        None
        '''
        print('before', self.Z.shape, self.W.shape, A_prev.shape, self.b.shape)
        self.Z = self.W @ A_prev + self.b
        print('during', self.Z.shape, self.W.shape, A_prev.shape, self.b.shape)

        if self.act == 'linear':
            self.A = self.Z

        elif self.act == 'sigmoid':
            self.A = 1 / (1 + np.exp(-self.Z))

        elif self.act == 'tanh':
            self.A = np.tanh(Z)

        print('after', self.Z.shape, self.W.shape, A_prev.shape, self.b.shape)


    def backward(self, W_next, dZ_next, A_prev):
        '''
        Description:
        Compute dA first, then dZ, then dW and db.

        Parameters:
        W_next -- [num_next, self.n] Array of weights between this layer and next layer.
        dZ_next -- [num_next, m_samples] Array of linear inputs to next layer.
        A_prev -- [num_prev,m_samples] Array of activations from previous layer.

        Computes:
        self.dA -- [self.n, m_samples] Gradient of cost w.r.t. activations.
        self.dZ -- [self.n, m_samples] Gradient of cost w.r.t. linear inputs.
        self.dW -- [self.n, num_prev] Gradient of cost w.r.t. weights.
        self.db -- [self.n, 1] Gradient of cost w.r.t. biases.
        '''
        m = self.A.shape[1]
        self.dA = W_next.T @ dZ_next

        if self.act == 'linear':
            self.dZ = self.dA * 1

        elif self.act == 'sigmoid':
            self.dZ = self.dA * self.A * (1 - self.A)

        elif self.act == 'tanh':
            self.dZ = self.dA * (1 - self.Z**2)

        self.dW = 1 / m * self.dZ @ A_prev.T
        self.db = 1 / m * np.sum(self.dZ, axis=1, keepdims=True)

    def update(self, learning_rate):
        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db


def arrayMemorySize(x, base=10):
    '''
    DESCRIPTION:
    Prints memory size of a numpy array.

    PARAMETERS:
    x -- (numpy array)
    base -- (int) Specifies base10 (for SI) or base2 (for IEC) computation.
    '''
    if base == 2:
        units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
        for i in range(5):
            if round(x.nbytes/2**(i * 10), 2) > 1:
                unit = units[i]
                power = i * 10

    else:
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        for i in range(5):
            if round(x.nbytes/10**(i * 3), 2) > 1:
                unit = units[i]
                power = i * 3

    print(np.shape(x), 'array uses', round(x.nbytes/base**power, 2), unit)


def buildNetwork(layout, num_features):
    '''
    Arguments:
    layout -- [num_layers, 2] Tuple where first column contains number of neurons in each layer
                and second column contains activation functions for each layer.

    Returns:
    network -- [num_layers,:] List of layers, each with number of neurons specified in 'network'.
    '''
    network = []
    network.append(Layer(num_features, layout[0][0], activation_function=layout[0][1]))
    print('Layer 1:', layout[0])

    for i in range(1, len(layout)):
        num_prev = layout[i-1][0]
        num_neurons = layout[i][0]
        activation_function = layout[i][1]
        network.append(Layer(num_prev, num_neurons, activation_function))
        print('Layer', i+1, '\b:', layout[i])

    return network


def costMSE(H, Y):
    '''
    Parameters:
    H -- [1, m] Predicted values.
    Y -- [1, m] Target values.

    Returns:
    J -- (float) Mean squared error of dataset.
    grad -- (float) Gradient of cost w.r.t. predicted values.
    '''
    if H.shape[0] > 1:
        print('ERROR: MSE cost function only excepts input with shape [1, m].')
    m = H.shape[1]
    J = 1 / (2 * m) * (H - Y) @ (H - Y).T
    grad = 1 / m * np.sum(H - Y, keepdims=True)
    return J, grad


def costLogistic(H, Y):
    '''
    Parameters:
    H -- [1, m] Predicted values in range (0, 1).
    Y -- [1, m] Target values, either 0 or 1.

    Returns:
    J -- (float) Logistic cost of dataset.
    grad -- (float) Gradient of cost w.r.t. predicted values.

    '''
    m = H.shape[1]
    J = -1 / m * (Y @ np.log(H.T) + (1 - Y) @ np.log(1 - H.T))
    grad = 1 / m * np.sum(np.divide(-Y, H) + np.divide(1 - Y, 1 - H), keepdims=True)
    return J, grad


def loadCSVData(filename, labelcol=-1, usecols=None, skiprow=0):
    '''
    DESCRIPTION:
    Loads data from CSV file into two arrays: one containing features and the other
    containing labels.

    PARAMETERS:
    filename -- (str) Location of CSV file to be processed.
    labelcol -- (int) Index of column containing labels.
    usecols -- (int, tuple) Indices of columns to be extracted from CSV.

    RETURNS:
    xdat -- [m,n](numpy array) Feature data.
    ydat -- [n,p](numpy array) Label data.
    '''
    data = np.loadtxt(filename, delimiter=',', skiprows=skiprow, usecols=usecols)
    np.random.shuffle(data)

    xdat = np.copy(data[:])
    xdat = np.delete(xdat, labelcol, axis=1)
    ydat = np.copy(data[:,labelcol])
    m = np.shape(ydat)[0]
    ydat = ydat.reshape(m,1)

    return xdat, ydat


def update_progress(progress, msg=''):
    '''
    DESCRIPTION:
    Displays progress bar and percent completion.

    PARAMETERS:
    progress -- (float) Decimal in range [0,1] representing percent completion.
    msg -- (str) Optional message to display with progress bar.
    '''
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    text += "\n" + msg
    print(text)


def computeCost(Y, network, costfunc='logistic'):
    # Compute cost
    if costfunc == 'logistic':
        cost, grad = costLogistic(network[-1].A, Y)
    elif costfunc == 'mse':
        cost, grad = costMSE(network[-1].A, Y)
    return cost, grad


def forwardprop(X, network):
    ###############
    # VERIFIED
    ###############
    L = len(network)
    network[0].forward(X)
    print('forward layer: 0')
    print('first A:', network[0].A.shape)

    for i in range(1, L):
        print('next A:', network[i-1].A.shape)
        network[i].forward(network[i-1].A)
        print('forward layer:', i)


def backprop(grad, X, network):
    L = len(network)
    W_output = np.eye(network[-1].A.shape[0], network[-1].A.shape[0])

    if L == 1:
        A_prev = X
    else:
        A_prev = network[L-2].A
    network[L-1].backward(W_output, grad, A_prev)
    #    print('Layer', L-1, 'backprop successful')

    for i in reversed(range(0, L - 1)):
        W_next = network[i+1].W
        dZ_next = network[i+1].dZ
        if i == 0:
            A_prev = X
        else:
            A_prev = network[i-1].A
        network[i].backward(W_next, dZ_next, A_prev)


def gradientDescent(X, Y, network, num_iterations, learning_rate,
                    costfunction='logistic', showprogress=True,
                    showpreds=False):
    L = len(network)
    costs = np.zeros((num_iterations))
    accs = np.zeros((num_iterations))

    for epoch in range(num_iterations):
        forwardprop(X, network)
        costs[epoch], grad = computeCost(Y, network, costfunc=costfunction)

        H = predict(X, network)
        accs[epoch], prec, rec = evaluateModel(H, Y)

        backprop(grad, X, network)

        # Perform one step of gradient descent
        for i in range(0, L):
            network[i].update(learning_rate)

        if showprogress == True:
            if showpreds == True:
                msg = H
            else: msg = ''
            update_progress(epoch/num_iterations, str(msg))


    if showprogress == True:
        update_progress(1)


    return network, costs, accs


def predict(X, network):
    '''
    RETURNS:
    predicted -- [num_target_classes, 1] Array of 1's and 0's representing
                 predicted values for each target class.
    '''
    # Forward propagation
    forwardprop(X, network)
    H = network[-1].A

    # Compute predictions
    predicted = np.zeros(network[-1].A.shape)
    if network[-1].act == 'sigmoid':
        predicted = 1*(H > 0.5)
    elif network[-1].act == 'tanh':
        predicted = 1*(H > 0)

    return predicted


def evaluateModel(H, Y, decimal=3):
    '''
    RETURNS:
    accuracy -- [num_classes, 1] Ratio of H predictions that match Y.
    precision -- [num_classes, 1] Ratio of true positives to flagged positives.
    recall -- [num_classes, 1] Ratio of flagged positives to total positives.
    '''
    k = Y.shape[1]
    eps = 1e-8

    tp = np.sum(np.logical_and(H, Y), axis=1, keepdims=True)
    fp = np.sum(np.logical_and(H, Y==0), axis=1, keepdims=True)
    tn = np.sum(np.logical_and(H==0, Y==0), axis=1, keepdims=True)
    fn = np.sum(np.logical_and(H==0, Y), axis=1, keepdims=True)

    accuracy = np.around(np.sum(H==Y, axis=1, keepdims=True) / k, decimal)
    precision = np.around(tp / (tp + fp + eps), decimal)
    recall = np.around(tp / (tp + fn + eps), decimal)

    return accuracy, precision, recall


def plot2DBoundary(X, Y, network):
    '''
    DESCRIPTION:
    Plots 2D class boundary as predicted by the last layer of a neural network.

    RETURN:
    None
    '''
    delta = 0.01
    x1_min, x1_max = X[0, :].min() - 1, X[0, :].max() + 1
    x2_min, x2_max = X[1, :].min() - 1, X[1, :].max() + 1
    x1 = np.arange(x1_min, x1_max, delta)
    x2 = np.arange(x2_min, x2_max, delta)
    a, b = np.meshgrid(x1, x2)

    c = np.c_[a.ravel(), b.ravel()].T
    Z = predict(c, network)
    Z = Z.reshape(a.shape)

    plt.scatter(X[[0], :], X[[1], :], c=Y.astype('uint8'), s=10, marker='.')
    plt.contour(a, b, Z, colors='black')
    plt.show()
