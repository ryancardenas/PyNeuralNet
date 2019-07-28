import copy
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

class Layer(object):
    def __init__(self, num_prev, num_neurons, activation_function='sigmoid'):
        '''
        DESCRIPTION:
        Initialize weights randomly with appropriate scaling.

        self.A -- [num_neurons, 1] Array of activations
        self.Z -- [num_neurons, 1] Array of linearly aggregated inputs
        self.W -- [num_neurons, num_prev] Array of weights applied to previous layer's activations A.
        self.b -- [num_neurons, 1] Array of biases

        PARAMETERS:
        num_prev -- (float) Number of neurons in previous connected layer.
        num_neurons -- (float) Number of neurons in this layer.
        activation_function -- (str) Activation function of this layer.

        RETURNS:
        None
        '''
        self.n = num_neurons
        self.act = activation_function

        self.A = np.zeros((num_neurons,1))
        self.Z = np.zeros((num_neurons,1))
        self.b = np.zeros((self.n,1))

        # He initialization.
        if self.act == 'sigmoid':
            self.W = np.random.randn(num_neurons, num_prev)
            self.W = self.W * np.sqrt(2.0/num_prev)
        # Xavier initialization
        elif self.act == 'tanh':
            self.W = np.random.randn(num_neurons, num_prev)
            self.W = self.W * np.sqrt(1.0/num_prev)


        self.dA = np.zeros((num_neurons,1))
        self.dZ = np.zeros((num_neurons,1))
        self.db = np.zeros((self.n,1))
        self.dW = np.zeros((num_neurons, num_prev))

    def forward(self, A_prev):
        '''
        DESCRIPTION:
        Perform forward pass by computing Z, then A.

        PARAMETERS:
        A_prev -- [num_prev, m_samples] Array of activations from previous layer.

        RETURNS:
        None
        '''
        self.Z = self.W @ A_prev + self.b

        if self.act == 'linear':
            self.A = self.Z

        elif self.act == 'sigmoid':
            self.A = 1 / (1 + np.exp(-self.Z))

        elif self.act == 'tanh':
            self.A = np.tanh(self.Z)

    def backward(self, W_next, dZ_next, A_prev):
        '''
        DESCRIPTION:
        Compute dA first, then dZ, then dW and db.

        self.dA -- [self.n, m_samples] Gradient of cost w.r.t. activations.
        self.dZ -- [self.n, m_samples] Gradient of cost w.r.t. linear inputs.
        self.dW -- [self.n, num_prev] Gradient of cost w.r.t. weights.
        self.db -- [self.n, 1] Gradient of cost w.r.t. biases.

        PARAMETERS:
        W_next -- [num_next, self.n] Array of weights between this layer and
                    next layer.
        dZ_next -- [num_next, m_samples] Array of linear inputs to next layer.
        A_prev -- [num_prev,m_samples] Array of activations from previous layer.

        RETURNS:
        None
        '''
        m = self.A.shape[1]
        self.dA = W_next.T @ dZ_next

        if self.act == 'linear':
            self.dZ = self.dA * 1

        elif self.act == 'sigmoid':
            self.dZ = self.dA * self.A * (1 - self.A)

        elif self.act == 'tanh':
            self.dZ = self.dA * (1 - self.A**2)

        self.dW = 1 / m * self.dZ @ A_prev.T
        self.db = 1 / m * np.sum(self.dZ, axis=1, keepdims=True)

    def update(self, learning_rate):
        '''
        DESCRIPTION:
        Performs one step of gradient descent on this layer.

        PARAMETERS:
        learning_rate -- (float) Determines gradient step size.

        RETURNS:
        None7
        '''
        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db

class Network(list):
    def __init__(self, cost, reg):
        self.costfunction = cost
        self.regularization = reg

    def backprop(self, grad, X):
        '''
        DESCRIPTION:
        Loops through each layer of a network and calls that layer's self.backward()
        method. Starts with last layer in network (network[-1]) and passes 'grad' as
        its parameter. Last layer also receives an identity matrix 'W_output' as a
        parameter in order to give a one-to-one mapping for the last layer and the
        cost function. Each preceding layer receives the gradient of the following
        layer as a parameter.

        PARAMETERS:
        self -- [num_layers] List of neuron layers.
        grad -- [num_output_neurons, m_samples] Gradient of cost function.
        X -- [num_features, m_samples] Input data set.

        RETURNS:
        None
        '''
        L = len(self)
        W_output = np.eye(self[-1].A.shape[0], self[-1].A.shape[0])

        if L == 1:
            A_prev = X
        else:
            A_prev = self[L-2].A
        self[L-1].backward(W_output, grad, A_prev)

        for i in reversed(range(0, L - 1)):
            W_next = self[i+1].W
            dZ_next = self[i+1].dZ
            if i == 0:
                A_prev = X
            else:
                A_prev = self[i-1].A
            self[i].backward(W_next, dZ_next, A_prev)

    def computeCost(self, Y):
        '''
        DESCRIPTION:
        Selects cost function routine to execute.

        PARAMETERS:
        self -- [num_layers] List of neuron layers.
        Y -- [1, m_samples] Target labels.
        costfunc -- (str) Selects cost function.

        RETURNS:
        cost -- (float) Cost function output.
        grad -- [1, m_samples] Gradient of cost function.
        '''
        if self.costfunction == 'logistic':
            cost, grad = costLogistic(self[-1].A, Y)
        elif self.costfunction == 'mse':
            cost, grad = costMSE(self[-1].A, Y)
        return cost, grad

    def forwardprop(self, X):
        '''
        DESCRIPTION:
        Loops through each layer of a network and calls that layer's self.forward()
        method. Starts with first layer in network (network[0]) and passes data 'X'
        as its parameter. Each subsequent layer receives the activations of the
        previous layer as a parameter.

        PARAMETERS:
        self -- [num_layers] List of neuron layers.
        X -- [num_features, m_samples] Input data set.

        RETURNS:
        None
        '''
        L = len(self)
        self[0].forward(X)

        for i in range(1, L):
            self[i].forward(self[i-1].A)

    def gradientDescent(self, X, Y, num_iterations, learning_rate,
                        costfunction='logistic', showprogress=True,
                        debugmsg='', recording=False):
        '''
        DESCRIPTION:
        Performs gradientDescent on each layer of neural network. Computes cost and
        accuracy for graphing vs. epoch.

        PARAMETERS:
        self -- [num_layers] List of neuron layers.
        X -- [num_features, m_samples] Input data set.
        Y -- [num_target_classes, m_samples] Target labels.
        num_iterations -- (int) Number of epochs through which to iterate.
        learning_rate -- (float) Determines gradient descent step size.
        costfunction -- (str) Selects cost function.
        showprogress -- (bool) Toggles progress bar.
        debugmsg -- (str) Selects info to display with progress bar. For complete
                        list of options, see msgGradDesc().
        recording -- (bool) Toggles calculating accuracy during each epoch. Provides
                        interesting accuracy vs. epoch data, but increases run time.

        RETURNS:
        self -- [num_layers] List of neuron layers.
        costs -- [num_iterations] List of cost function outputs vs. epoch.
        accs -- [num_iterations] List of accuracies vs. epoch. Outputs a list of
                    zeros if 'recording' is set to 'False'.
        '''
        L = len(self)
        costs = np.zeros((num_iterations))
        accs = np.zeros((num_iterations))

        for epoch in range(num_iterations):
            # Prepare neuron layers for gradient descent step update.
            self.forwardprop( X)
            costs[epoch], grad = self.computeCost(Y)
            self.backprop(grad, X)

            # Record accuracy, precision, and recall of model in current state.
            if recording:
                H = self.predict(X)
                accs[epoch], prec, rec = evaluateModel(H, Y)

            # Perform one step of gradient descent.
            for i in range(0, L):
                self[i].update(learning_rate)

            # Display progress bar and a message if specified.
            if showprogress == True:
                msg = ''
                if debugmsg != '':
                    msg = self.msgGradDesc(debugmsg, grad, H, epoch,
                                      costs, accs)
                update_progress(epoch/num_iterations, str(msg))

        # Show completed progress bar.
        if showprogress == True:
            update_progress(1)

        return self, costs, accs

    def info(self):
        print('Cost Function: ', self.costfunction)
        print('Regularization:', self.regularization)
        for i in range(len(self)):
            print('Layer', i, '\b:', self.layout[i])

    def msgGradDesc(self, debugmsg, grad, H, epoch, costs, accs):
        '''
        DESCRIPTION:
        Interprets debugmsg keyword for gradientDescent() and outputs corresponding
        information as a string.

        PARAMETERS:
        Internal parameters of gradientDescent().

        RETURNS:
        msg -- (str) Debugging message to be displayed during gradient descent.
        '''
        if debugmsg == '':
            msg = ''
        elif debugmsg == 'grad':
            msg = grad
        elif debugmsg == 'H':
            msg = H
        elif debugmsg == 'cost':
            msg = costs[epoch]
        elif debugmsg == 'acc':
            msg = accs[epoch]
        elif debugmsg == 'last.A':
            msg = self[-1].A
        elif debugmsg == 'last.Z':
            msg = self[-1].Z
        elif debugmsg == 'last.W':
            msg = self[-1].W
        elif debugmsg == 'last.b':
            msg = self[-1].b
        elif debugmsg == 'last.dA':
            msg = self[-1].dA
        elif debugmsg == 'last.dZ':
            msg = self[-1].dZ
        elif debugmsg == 'last.dW':
            msg = self[-1].dW
        elif debugmsg == 'last.db':
            msg = self[-1].db
        elif debugmsg == 'first.A':
            msg = self[0].A
        elif debugmsg == 'first.Z':
            msg = self[0].Z
        elif debugmsg == 'first.W':
            msg = self[0].W
        elif debugmsg == 'first.b':
            msg = self[0].b
        elif debugmsg == 'first.dA':
            msg = self[0].dA
        elif debugmsg == 'first.dZ':
            msg = self[0].dZ
        elif debugmsg == 'first.dW':
            msg = self[0].dW
        elif debugmsg == 'first.db':
            msg = self[0].db
        return msg

    def predict(self, X):
        '''
        DESCRIPTION:
        Converts output of last layer of neural network to binary class predictions.

        PARAMETERS:
        self -- [num_layers] List of neuron layers.
        X -- [num_features, m_samples] Input data set.

        RETURNS:
        self.predicted -- [num_target_classes, m_samples] Array of 1's and 0's representing
                            predicted values for each target class.
        '''
        # Make copy of network that won't alter original network.
        network = copy.deepcopy(self)

        # Forward propagation
        network.forwardprop(X)
        H = network[-1].A

        # Compute predictions
        self.predicted = np.zeros(H.shape)
        if network[-1].act == 'sigmoid':
            self.predicted = 1*(H > 0.5)
        elif network[-1].act == 'tanh':
            self.predicted = 1*(H > 0)

        return self.predicted

def arrayMemorySize(x, base=10):
    '''
    DESCRIPTION:
    Prints memory size of a numpy array.

    PARAMETERS:
    x -- (numpy array)
    base -- (int) Specifies base10 (for SI) or base2 (for IEC) computation.

    RETURNS:
    None
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

def buildNetwork(layout, num_features, cost='logistic', reg='L2'):
    '''
    DESCRIPTION:
    Constructs list of neuron layers.

    PARAMETERS:
    layout -- [num_layers, 2] Tuple where first column contains number of neurons in each layer
                and second column contains activation functions for each layer.
    num_features -- (int) Number of features in the input data set.

    RETURNS:
    network -- [num_layers,:] List of layers, each with number of neurons specified in 'network'.
    '''
    network = Network(cost, reg)
    network.layout = layout
    network.append(Layer(num_features, layout[0][0], activation_function=layout[0][1]))

    for i in range(1, len(layout)):
        num_prev = layout[i-1][0]
        num_neurons = layout[i][0]
        activation_function = layout[i][1]
        network.append(Layer(num_prev, num_neurons, activation_function))

    network.info()
    return network

def costLogistic(H, Y):
    '''
    DESCRIPTION:
    Comptutes cross entropy 'logistic' cost and gradient.

    PARAMETERS:
    H -- [1, m_samples] Predicted values in range (0, 1).
    Y -- [1, m_samples] Target labels, either 0 or 1.

    RETURNS:
    J -- (float) Logistic cost of dataset.
    grad -- [1, m_samples] Gradient of cost w.r.t. predicted values.
                NOTE: grad must be [1, m] because backpropagation will sum over
                axis=1 and then divide by m_samples to get an average for
                updating network[?].W and network[?].b.

    '''
    m = Y.shape[1]
    Y = Y.astype('float32')
    J = -1 / m * (Y @ np.log(H.T) + (1 - Y) @ np.log(1 - H.T))
    grad = np.divide((H - Y), H * (1 - H))
    return J, grad

def costMSE(H, Y):
    '''
    DESCRIPTION:
    Computes mean squared error cost and gradient.

    PARAMETERS:
    H -- [1, m_samples] Predicted values.
    Y -- [1, m_samples] Target labels.

    RETURNS:
    J -- (float) Mean squared error of dataset.
    grad -- [1, m_samples] Gradient of cost w.r.t. predicted values.
    '''
    m = Y.shape[1]
    Y = Y.astype('float32')
    J = 1 / (2 * m) * (H - Y) @ (H - Y).T
    grad = 1 / m * (H - Y)
    return J, grad

def evaluateModel(H, Y, decimal=3):
    '''
    DESCRIPTION:
    Computes accuracy, precision, and recall model H and target Y.

    PARAMETERS:
    H -- [num_target_classes, m_samples] One-hot class predictions.
    Y -- [num_target_classes, m_samples] One-hot target labels.
    decimal -- (int) Specifies number of decimal places to round to.

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

def plot2DBoundary(X, Y, network, fill=False, lines=True,
                    color_map=plt.cm.Spectral, mark='o', marksize=20):
    '''
    DESCRIPTION:
    Plots 2D class boundary as predicted by the last layer of a neural network.

    PARAMETERS:
    X -- [num_features, m_samples] Input data set.
    Y -- [1, m_samples] Target labels, either 1 or 0.
    network -- [num_layers] List of neuron layers.
    fill -- (bool) Toggles filled contour plot.
    lines -- (bool) Toggles line contour plot.
    color_map -- (Numpy colormap) Specifies color scheme to use in plots.

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
    Z = network.predict(c)
    Z = Z.reshape(a.shape)

    if fill:
        plt.contourf(a, b, Z, cmap=color_map)
    if lines:
        plt.contour(a, b, Z, colors='black')
    plt.scatter(X[[0], :], X[[1], :], marker=mark, s=marksize,
                c=Y.astype('uint8'), cmap=color_map)
    plt.xlim(X[0].min(), X[0].max())
    plt.ylim(X[1].min(), X[1].max())
    plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def update_progress(progress, msg=''):
    '''
    DESCRIPTION:
    Displays progress bar and percent completion.

    PARAMETERS:
    progress -- (float) Decimal in range [0,1] representing percent completion.
    msg -- (str) Optional message to display with progress bar.

    RETURNS:
    None
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
