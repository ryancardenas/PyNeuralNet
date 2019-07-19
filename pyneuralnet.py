import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

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

    def forward(self, A_prev):
        '''
        Description:
        Perform forward pass by computing Z, then A.

        Parameters:
        A_prev -- [num_prev, m_samples] Array of activations from previous layer.

        Computes:
        self.A -- [self.n, m_samples] Array of activations for this layer.
        '''
        self.Z = self.W @ A_prev + self.b

        if self.act == 'linear':
            self.A = self.Z

        elif self.act == 'sigmoid':
            self.A = 1 / (1 + np.exp(-self.Z))

        return self.A

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
            self.dZ = self.dA

        elif self.act == 'sigmoid':
            self.dZ = self.A * (1 - self.A) * self.dA

        self.dW = 1 / m * self.dZ @ A_prev.T
        self.db = 1 / m * np.sum(self.dZ, axis=1, keepdims=True)

    def update(self, learning_rate):
        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db


def buildNetwork(layout, num_features):
    '''
    Arguments:
    layout -- [num_layers, 2] Tuple where first column contains number of neurons in each layer
                and second column contains activation functions for each layer.

    Returns:
    network -- [num_layers,:] List of layers, each with number of neurons specified in 'network'.
    '''
    network = []
    network.append(Layer(num_features, layout[0][0], layout[0][1]))
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
    grad = 1 / m * np.sum(H - Y)
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
    grad = 1 / m * np.sum(np.divide(-Y, H) + np.divide(1 - Y, 1 - H))
    return J, grad


def forwardprop(X, network):
    L = len(network)
    network[0].forward(X)

    for i in range(1, L):
        network[i].forward(network[i-1].A)

    return network


def computeCost(Y, network, costfunc='logistic'):
    # Compute cost
    if costfunc == 'logistic':
        cost, grad = costLogistic(network[-1].A, Y)
    elif costfunc == 'mse':
        cost, grad = costMSE(network[-1].A, Y)
#     print('Cost: ', costs)
    return cost, grad


def backprop(grad, network):
    L = len(network)
    W_output = np.eye(Y.shape[0], Y.shape[0])

    network[L-1].backward(W_output, grad, network[L-2].A)

    for i in reversed(range(0, L - 1)):
        W_next = network[i+1].W
        dZ_next = network[i+1].dZ
        A_prev = network[i-1].A
        network[i].backward(W_next, dZ_next, A_prev)

    return network


def gradientDescent(X, Y, network, num_iterations, learning_rate,
                    costfunc='logistic', showprogress='True'):
    L = len(network)
    costs = np.zeros((num_iterations,k))

    for epoch in range(num_iterations):
        network = forwardprop(X, network)
        costs[epoch], grad = computeCost(Y, network, costfunc)
        network = backprop(grad, network)

        # Perform one step of gradient descent
        for i in range(1, L):
            network[i].update(learning_rate)

        if showprogress == 'True':
            update_progress(epoch/num_iterations)

    if showprogress == 'True':
        update_progress(1)

    return network, costs


def predict(X, network):
    L = len(network)
    H = network[-1].A

    # Forward propagation
    network = forwardprop(X, network)

    # Compute predictions
    predicted = np.zeros(network[-1].A.shape)
    if network[-1].act == 'sigmoid':
        predicted = 1*(H > 0.5)
    elif network[-1].act == 'tanh':
        predicted = 1*(H > 0)

    return H


def evaluateModel(H, Y):
    k = Y.shape[0]

    tp = np.sum(np.logical_and(H, Y))
    fp = np.sum(np.logical_and(H, Y==0))
    tn = np.sum(np.logical_and(H==0, Y==0))
    fn = np.sum(np.logical_and(H==0, Y))

    accuracy = np.sum(H==Y) / k
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return accuracy, precision, recall
